/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/opt/IOBarrierElimination.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::llvm
{

class IOBarrierElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * SweepTimerLabel_ = "SweepTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::IOBarrierElimination, sourceFile)
  {}

  void
  startMarkStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddTimer(MarkTimerLabel_).start();
  }

  void
  stopMarkStatistics() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  void
  startSweepStatistics() noexcept
  {
    AddTimer(SweepTimerLabel_).start();
  }

  void
  stopSweepStatistics(const rvsdg::Graph & graph) noexcept
  {
    GetTimer(SweepTimerLabel_).stop();
  }

  static std::unique_ptr<Statistics>
  create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

class IOBarrierElimination::Context
{
public:
  /**
   * Mark \p output as dereferenceable with size \p sizeInBytes.
   *
   * @return True, if the output was not already marked as dereferenceable, otherwise false.
   */
  bool
  markDereferenceable(const rvsdg::Output & output, const size_t sizeInBytes)
  {
    const auto it = dereferenceableOutputs_.find(&output);
    if (it == dereferenceableOutputs_.end())
    {
      dereferenceableOutputs_[&output] = sizeInBytes;
      return true;
    }

    if (it->second < sizeInBytes)
      dereferenceableOutputs_[&output] = sizeInBytes;

    return false;
  }

  /**
   * @return The size in bytes, if \p output is marked as dereferenceable, otherwise std::nullopt.
   */
  std::optional<size_t>
  isDereferenceable(const rvsdg::Output & output) const
  {
    const auto it = dereferenceableOutputs_.find(&output);
    if (it == dereferenceableOutputs_.end())
      return std::nullopt;

    return it->second;
  }

  static std::unique_ptr<Context>
  create()
  {
    return std::make_unique<Context>();
  }

private:
  std::unordered_map<const rvsdg::Output *, size_t> dereferenceableOutputs_{};
};

IOBarrierElimination::~IOBarrierElimination() = default;

IOBarrierElimination::IOBarrierElimination()
    : Transformation("IOBarrierElimination")
{}

void
IOBarrierElimination::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  auto & rvsdg = module.Rvsdg();

  context_ = Context::create();
  auto statistics = Statistics::create(module.SourceFilePath().value());

  statistics->startMarkStatistics(rvsdg);
  markRegion(rvsdg.GetRootRegion());
  statistics->stopMarkStatistics();

  statistics->startSweepStatistics();
  sweepRegion(rvsdg.GetRootRegion());
  statistics->stopSweepStatistics(rvsdg);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  context_.reset();
}

void
IOBarrierElimination::markRegion(rvsdg::Region & region)
{
  for (const auto node : rvsdg::TopDownTraverser(&region))
  {
    markNode(*node);
  }
}

void
IOBarrierElimination::markNode(const rvsdg::Node & node)
{
  rvsdg::MatchType(
      node.GetOperation(),
      [&](const rvsdg::PhiOperation &)
      {
        const auto phiNode = util::assertedCast<const rvsdg::PhiNode>(&node);
        markRegion(*phiNode->subregion());
      },
      [&](const LlvmLambdaOperation &)
      {
        const auto lambdaNode = util::assertedCast<const rvsdg::LambdaNode>(&node);
        markRegion(*lambdaNode->subregion());
      },
      [&](const LoadNonVolatileOperation & loadOperation)
      {
        const auto & addressOperand = *LoadOperation::AddressInput(node).origin();
        const auto sizeInBytes = GetTypeStoreSize(*loadOperation.GetLoadedType());

        auto [ioBarrierNode, ioBarrierOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(addressOperand);
        if (ioBarrierOp)
        {
          const auto & barredAddressOperand =
              *IOBarrierOperation::BarredInput(*ioBarrierNode).origin();
          if (const auto & ioStateInput = IOBarrierOperation::getIOStateInput(*ioBarrierNode);
              rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(*ioStateInput.origin()))
          {
            // If the IO state is directly connected to a function argument, we can eliminate the
            // IOBarrierOperation node as function inlining should reinsert a new IOBarrierOperation
            // node when inlining is performed.
            context_->markDereferenceable(barredAddressOperand, sizeInBytes);
          }
        }
        else
        {
          // The load node is not connected to a IOBarrierOperation node. Mark its address operand
          // as dereferenceable.
          context_->markDereferenceable(addressOperand, sizeInBytes);
        }
      });
}

void
IOBarrierElimination::sweepRegion(rvsdg::Region & region)
{
  for (const auto node : rvsdg::BottomUpTraverser(&region))
  {
    rvsdg::MatchType(
        node->GetOperation(),
        [&](const rvsdg::PhiOperation &)
        {
          const auto phiNode = util::assertedCast<const rvsdg::PhiNode>(node);
          sweepRegion(*phiNode->subregion());
        },
        [&](const LlvmLambdaOperation &)
        {
          const auto lambdaNode = util::assertedCast<const rvsdg::LambdaNode>(node);
          sweepRegion(*lambdaNode->subregion());
        },
        [&](const LoadNonVolatileOperation & loadOperation)
        {
          const auto & addressOperand = *LoadOperation::AddressInput(*node).origin();
          auto [ioBarrierNode, ioBarrierOp] =
              rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(addressOperand);
          if (!ioBarrierOp)
            return;

          const auto & barredAddressOperand =
              *IOBarrierOperation::BarredInput(*ioBarrierNode).origin();
          auto sizeOpt = context_->isDereferenceable(barredAddressOperand);
          const auto sizeInBytes = GetTypeStoreSize(*loadOperation.GetLoadedType());
          if (!sizeOpt.has_value() || sizeOpt.value() < sizeInBytes)
            return;

          removeIOBarrierNode(*ioBarrierNode);
        });
  }
}

void
IOBarrierElimination::removeIOBarrierNode(rvsdg::Node & node)
{
  JLM_ASSERT(is<IOBarrierOperation>(&node));

  const auto & barredInput = IOBarrierOperation::BarredInput(node);
  node.output(0)->divert_users(barredInput.origin());
  remove(&node);
}

}
