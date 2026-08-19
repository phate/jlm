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

  /**
   * Mark \p node eliminable.
   *
   * @param node An \ref IOBarrierOperation node
   * @return True, if the node was not already marked as eliminable, otherwise false.
   */
  bool
  markEliminable(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<IOBarrierOperation>(node.GetOperation()));
    return eliminableIOBarriers_.insert(&node);
  }

  bool
  isEliminable(const rvsdg::Node & node) const
  {
    // We only care about IOBarrierOperation nodes.
    if (!is<IOBarrierOperation>(node.GetOperation()))
      return false;

    // The IOBarrier node was directly marked for elimination
    if (eliminableIOBarriers_.Contains(&node))
      return true;

    // Its barred input was marked as dereferenceable
    const auto & barredInput = IOBarrierOperation::BarredInput(node);
    if (dereferenceableOutputs_.find(barredInput.origin()) != dereferenceableOutputs_.end())
      return true;

    return false;
  }

  static std::unique_ptr<Context>
  create()
  {
    return std::make_unique<Context>();
  }

private:
  std::unordered_map<const rvsdg::Output *, size_t> dereferenceableOutputs_{};
  util::HashSet<const rvsdg::Node *> eliminableIOBarriers_{};
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

  markRegion(rvsdg.GetRootRegion());
  sweepRegion(rvsdg.GetRootRegion());

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
