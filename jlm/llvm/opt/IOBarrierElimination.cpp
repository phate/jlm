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
  const char * NormalizationTimerLabel_ = "NormalizationTime";
  const char * MarkTimerLabel_ = "MarkTime";
  const char * PropagateTimerLabel_ = "PropagateTime";
  const char * SweepTimerLabel_ = "SweepTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::IOBarrierElimination, sourceFile)
  {}

  void
  startNormalizationStatistics() noexcept
  {
    AddTimer(NormalizationTimerLabel_).start();
  }

  void
  stopNormalizationStatistics() noexcept
  {
    GetTimer(NormalizationTimerLabel_).stop();
  }

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
   * Mark \p input as dereferenceable with size \p sizeInBytes.
   *
   * @return True, if the input was not already marked as dereferenceable, otherwise false.
   */
  bool
  markDereferenceable(const rvsdg::Input & input, const size_t sizeInBytes)
  {
    const auto it = dereferenceableInputs_.find(&input);
    if (it == dereferenceableInputs_.end())
    {
      dereferenceableInputs_[&input] = sizeInBytes;
      return true;
    }

    dereferenceableInputs_[&input] = std::max(it->second, sizeInBytes);
    return false;
  }

  bool
  markUsersDereferenceable(const rvsdg::Output & output, const size_t sizeInBytes)
  {
    bool wasMarked = false;
    for (auto & user : output.Users())
    {
      wasMarked |= markDereferenceable(user, sizeInBytes);
    }

    return wasMarked;
  }

  /**
   * @return The size in bytes, if \p input is marked as dereferenceable, otherwise std::nullopt.
   */
  [[nodiscard]] std::optional<size_t>
  isDereferenceable(const rvsdg::Input & input) const
  {
    const auto it = dereferenceableInputs_.find(&input);
    if (it == dereferenceableInputs_.end())
      return std::nullopt;

    return it->second;
  }

  [[nodiscard]] size_t
  numDereferenceableInputs() const
  {
    return dereferenceableInputs_.size();
  }

  static std::unique_ptr<Context>
  create()
  {
    return std::make_unique<Context>();
  }

private:
  std::unordered_map<const rvsdg::Input *, size_t> dereferenceableInputs_{};
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

  statistics->startNormalizationStatistics();
  normalizeIOBarriers(rvsdg.GetRootRegion());
  statistics->stopNormalizationStatistics();

  statistics->startMarkStatistics();
  markDereferenceable(rvsdg.GetRootRegion());
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

static void
divertUsersToIOBarrierOutput(rvsdg::Output & output)
{
  rvsdg::SimpleNode * ioBarrierNode = nullptr;
  for (auto & user : output.Users())
  {
    if (auto [node, ioBarrierOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(user);
        ioBarrierOp)
    {
      // Ensure that the IOBarrierOperation operands are both originating from the same owner
      if (IOBarrierOperation::getIOStateInput(*node).origin()->GetOwner() != output.GetOwner())
        continue;

      ioBarrierNode = node;
      break;
    }
  }

  if (!ioBarrierNode)
    return;

  output.divertUsersWhere(
      *ioBarrierNode->output(0),
      [&ioBarrierNode](const rvsdg::Input & user)
      {
        return &IOBarrierOperation::BarredInput(*ioBarrierNode) != &user;
      });
}

void
IOBarrierElimination::normalizeIOBarriers(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        // Handle innermost regions first
        normalizeIOBarriers(subregion);

        // Normalize subregion arguments
        for (auto & argument : subregion.Arguments())
        {
          if (is<PointerType>(argument->Type()))
          {
            divertUsersToIOBarrierOutput(*argument);
          }
        }
      }

      // Normalize node outputs
      for (auto & output : structuralNode->Outputs())
      {
        if (is<PointerType>(output.Type()))
        {
          divertUsersToIOBarrierOutput(output);
        }
      }
    }
  }
}

void
IOBarrierElimination::markDereferenceable(const rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        markDereferenceable(subregion);
      }
    }
    else
    {
      if (const auto loadOperation =
              dynamic_cast<const LoadNonVolatileOperation *>(&node.GetOperation()))
      {
        const auto & addressOperand = *LoadOperation::AddressInput(node).origin();
        const auto sizeInBytes = GetTypeStoreSize(*loadOperation->GetLoadedType());
        context_->markUsersDereferenceable(addressOperand, sizeInBytes);
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
              if (!is<PointerType>(input->Type()))
                continue;

              if (auto size = context_->isDereferenceable(*input))
              {
                for (const auto & argument : arguments)
                {
                  context_->markUsersDereferenceable(*argument, size.value());
                }
              }
            }

            for (auto & subregion : gammaNode.Subregions())
              propagate(subregion);

            for (auto & [results, output] : gammaNode.GetExitVars())
            {
              if (!is<PointerType>(output->Type()))
                continue;

              bool allResultsAreDereferenceable = true;
              size_t sizeInBytes = std::numeric_limits<std::size_t>::max();
              for (const auto & result : results)
              {
                auto sizeOpt = context_->isDereferenceable(*result);
                if (!sizeOpt)
                {
                  allResultsAreDereferenceable = false;
                  break;
                }

                sizeInBytes = std::min(sizeInBytes, sizeOpt.value());
              }
              if (allResultsAreDereferenceable)
                context_->markUsersDereferenceable(*output, sizeInBytes);
            }
          },
          [&](rvsdg::ThetaNode & thetaNode)
          {
            // FIXME: This could be improved
            for (const auto & loopVar : thetaNode.GetLoopVars())
            {
              if (!is<PointerType>(loopVar.input->Type()))
                continue;

              auto inputSizeOpt = context_->isDereferenceable(*loopVar.input);
              auto resultSizeOpt = context_->isDereferenceable(*loopVar.post);
              if (inputSizeOpt && resultSizeOpt)
              {
                auto sizeInBytes = std::min(inputSizeOpt.value(), resultSizeOpt.value());
                context_->markUsersDereferenceable(*loopVar.output, sizeInBytes);
              }
            }

            propagate(*thetaNode.subregion());
          },
          [&](rvsdg::DeltaNode &)
          {
            // Nothing needs to be done
          },
          [&](rvsdg::SimpleNode & simpleNode)
          {
            rvsdg::MatchType(
                simpleNode.GetOperation(),
                [this, &simpleNode](const IOBarrierOperation &)
                {
                  const auto & barredInput = IOBarrierOperation::BarredInput(simpleNode);
                  if (!is<PointerType>(barredInput.Type()))
                    return;

                  if (const auto size = context_->isDereferenceable(barredInput))
                    context_->markUsersDereferenceable(*simpleNode.output(0), size.value());
                });
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
  size_t numDereferenceableInputs = 0;
  do
  {
    numDereferenceableInputs = context_->numDereferenceableInputs();
    propagate(graph.GetRootRegion());
  } while (numDereferenceableInputs != context_->numDereferenceableInputs());
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

            auto & barredAddressInput = IOBarrierOperation::BarredInput(*ioBarrierNode);
            const auto sizeOpt = context_->isDereferenceable(barredAddressInput);
            const auto sizeInBytes = GetTypeStoreSize(*loadOperation->GetLoadedType());
            if (!sizeOpt.has_value() || sizeOpt.value() < sizeInBytes)
              return;

            loadAddress.divert_to(barredAddressInput.origin());
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
