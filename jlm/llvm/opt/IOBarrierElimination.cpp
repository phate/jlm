/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/opt/IOBarrierElimination.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::llvm
{

class IOBarrierElimination::Statistics final : public util::Statistics
{
  const char * MarkTimerLabel_ = "MarkTime";
  const char * PropagateTimerLabel_ = "PropagateTime";
  const char * SweepTimerLabel_ = "SweepTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::IOBarrierElimination, sourceFile)
  {}

  void
  startMarkStatistics() noexcept
  {
    AddTimer(MarkTimerLabel_).start();
  }

  void
  stopMarkStatistics() noexcept
  {
    GetTimer(MarkTimerLabel_).stop();
  }

  void
  startPropagateStatistics() noexcept
  {
    AddTimer(PropagateTimerLabel_).start();
  }

  void
  stopPropagateStatistics() noexcept
  {
    GetTimer(PropagateTimerLabel_).stop();
  }

  void
  startSweepStatistics() noexcept
  {
    AddTimer(SweepTimerLabel_).start();
  }

  void
  stopSweepStatistics() noexcept
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

    dereferenceableOutputs_[&output] = std::max(it->second, sizeInBytes);
    return false;
  }

  /**
   * @return The size in bytes, if \p output is marked as dereferenceable, otherwise std::nullopt.
   */
  [[nodiscard]] std::optional<size_t>
  isDereferenceable(const rvsdg::Output & output) const
  {
    const auto it = dereferenceableOutputs_.find(&output);
    if (it == dereferenceableOutputs_.end())
      return std::nullopt;

    return it->second;
  }

  [[nodiscard]] size_t
  numDereferenceableOutputs() const
  {
    return dereferenceableOutputs_.size();
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

  statistics->startMarkStatistics();
  markOutputsDereferenceable(rvsdg.GetRootRegion());
  statistics->stopMarkStatistics();

  statistics->startPropagateStatistics();
  propagateDereferenceable(rvsdg);
  statistics->stopPropagateStatistics();

  statistics->startSweepStatistics();
  sweepRegion(rvsdg.GetRootRegion());
  statistics->stopSweepStatistics();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  context_.reset();
}

void
IOBarrierElimination::markOutputsDereferenceable(const rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        markOutputsDereferenceable(subregion);
      }
    }
    else
    {
      if (const auto loadOperation =
              dynamic_cast<const LoadNonVolatileOperation *>(&node.GetOperation()))
      {
        const auto & addressOperand = *LoadOperation::AddressInput(node).origin();
        const auto sizeInBytes = GetTypeStoreSize(*loadOperation->GetLoadedType());

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
      }
    }
  }
}

void
IOBarrierElimination::propagateDereferenceable(rvsdg::Graph & graph)
{
  std::function<void(rvsdg::Region &)> propagate = [&](rvsdg::Region & region)
  {
    for (auto node : rvsdg::TopDownTraverser(&region))
    {
      rvsdg::MatchTypeWithDefault(
          *node,
          [&](rvsdg::PhiNode & phiNode)
          {
            propagate(*phiNode.subregion());
          },
          [&](rvsdg::LambdaNode & lambdaNode)
          {
            propagate(*lambdaNode.subregion());
          },
          [&](rvsdg::GammaNode & gammaNode)
          {
            for (auto & [input, arguments] : gammaNode.GetEntryVars())
            {
              if (auto size = context_->isDereferenceable(*input->origin()))
              {
                for (const auto & argument : arguments)
                {
                  context_->markDereferenceable(*argument, size.value());
                }
              }
            }

            for (auto & subregion : gammaNode.Subregions())
              propagate(subregion);

            for (auto & [results, output] : gammaNode.GetExitVars())
            {
              bool allResultsAreDereferenceable = true;
              size_t sizeInBytes = std::numeric_limits<std::size_t>::max();
              for (const auto & result : results)
              {
                auto sizeOpt = context_->isDereferenceable(*result->origin());
                if (!sizeOpt)
                {
                  allResultsAreDereferenceable = false;
                  break;
                }

                sizeInBytes = std::min(sizeInBytes, sizeOpt.value());
              }
              if (allResultsAreDereferenceable)
                context_->markDereferenceable(*output, sizeInBytes);
            }
          },
          [&](rvsdg::ThetaNode & thetaNode)
          {
            // FIXME: This could be improved
            for (const auto & loopVar : thetaNode.GetLoopVars())
            {
              auto inputSizeOpt = context_->isDereferenceable(*loopVar.input->origin());
              auto resultSizeOpt = context_->isDereferenceable(*loopVar.post->origin());
              if (inputSizeOpt && resultSizeOpt)
              {
                auto sizeInBytes = std::min(inputSizeOpt.value(), resultSizeOpt.value());
                context_->markDereferenceable(*loopVar.output, sizeInBytes);
              }
            }

            propagate(*thetaNode.subregion());
          },
          [&](rvsdg::DeltaOperation &)
          {
            // Nothing needs to be done
          },
          [&](rvsdg::SimpleNode &)
          {
            // Nothing needs to be done
          },
          []()
          {
            throw std::logic_error(
                "Unhandled node type encountered during dereferenceable propagation.");
          });
    }
  };

  // FIXME: This is a simple fixpoint algorithm and can improved
  // FIXME: The algorithm is intra-procedural. There is no need to iterate over the entire graph
  // again. We could also just iterate over a function again.
  // FIXME: Counting the number of dereferenceable outputs is imprecise. It might be that we could
  // improve the result further as the size of an already marked output is widened. This is
  // currently not captured here.
  size_t numDereferenceableOutputs = 0;
  do
  {
    numDereferenceableOutputs = context_->numDereferenceableOutputs();
    propagate(graph.GetRootRegion());
  } while (numDereferenceableOutputs != context_->numDereferenceableOutputs());
}

void
IOBarrierElimination::sweepRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeWithDefault(
        node,
        [this](const rvsdg::PhiNode & phiNode)
        {
          sweepRegion(*phiNode.subregion());
        },
        [this](const rvsdg::LambdaNode & lambdaNode)
        {
          sweepRegion(*lambdaNode.subregion());
        },
        [](const rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [this](rvsdg::GammaNode & gammaNode)
        {
          for (auto & subregion : gammaNode.Subregions())
          {
            sweepRegion(subregion);
          }
        },
        [this](const rvsdg::ThetaNode & thetaNode)
        {
          sweepRegion(*thetaNode.subregion());
        },
        [this](const rvsdg::SimpleNode & simpleNode)
        {
          if (const auto loadOperation =
                  dynamic_cast<const LoadNonVolatileOperation *>(&simpleNode.GetOperation()))
          {
            auto & loadAddress = LoadOperation::AddressInput(simpleNode);
            auto [ioBarrierNode, ioBarrierOp] =
                rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(*loadAddress.origin());
            if (!ioBarrierOp)
              return;

            auto & barredAddressOperand = *IOBarrierOperation::BarredInput(*ioBarrierNode).origin();
            const auto sizeOpt = context_->isDereferenceable(barredAddressOperand);
            const auto sizeInBytes = GetTypeStoreSize(*loadOperation->GetLoadedType());
            if (!sizeOpt.has_value() || sizeOpt.value() < sizeInBytes)
              return;

            loadAddress.divert_to(&barredAddressOperand);
          }
        },
        []()
        {
          throw std::logic_error("Unsupported node type");
        });
  }

  region.prune(false);
}

}
