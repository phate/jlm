/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
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
   * Mark all users of \p output as dereferenceable with size \p sizeInBytes.
   */
  void
  markUsersDereferenceable(const rvsdg::Output & output, const size_t sizeInBytes)
  {
    for (auto & user : output.Users())
    {
      markDereferenceable(user, sizeInBytes);
    }
  }

  /**
   * @return The size in bytes. If \p input was not marked as dereferenceable, then 0 is returned.
   */
  [[nodiscard]] size_t
  isDereferenceable(const rvsdg::Input & input) const
  {
    const auto it = dereferenceableInputs_.find(&input);
    if (it == dereferenceableInputs_.end())
      return 0;

    return it->second;
  }

  static std::unique_ptr<Context>
  create()
  {
    return std::make_unique<Context>();
  }

private:
  /**
   * Mark \p input as dereferenceable with size \p sizeInBytes.
   */
  void
  markDereferenceable(const rvsdg::Input & input, const size_t sizeInBytes)
  {
    if (const auto it = dereferenceableInputs_.find(&input); it == dereferenceableInputs_.end())
    {
      dereferenceableInputs_[&input] = sizeInBytes;
    }
    else
    {
      dereferenceableInputs_[&input] = std::max(it->second, sizeInBytes);
    }
  }

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

static std::vector<rvsdg::SimpleNode *>
collectMemoryHoistBarrierNodes(rvsdg::Output & output)
{
  std::vector<rvsdg::SimpleNode *> hoistBarrierNodes;
  for (auto & user : output.Users())
  {
    if (auto [node, hoistBarrierOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryHoistBarrierOperation>(user);
        hoistBarrierOp)
    {
      hoistBarrierNodes.push_back(node);
    }
  }

  return hoistBarrierNodes;
}

static std::optional<rvsdg::SimpleNode *>
selectMemoryHoistBarrierNode(
    const std::vector<rvsdg::SimpleNode *> & hoistBarrierNodes,
    const std::variant<rvsdg::Node *, rvsdg::Region *> ioStateOwner)
{
  for (auto node : hoistBarrierNodes)
  {
    if (MemoryHoistBarrierOperation::getIOStateInput(*node).origin()->GetOwner() == ioStateOwner)
      return node;
  }

  return std::nullopt;
}

static void
divertUsersToMemoryHoistBarrierNode(
    rvsdg::Output & output,
    rvsdg::SimpleNode & memoryHoistBarrierNode)
{
  JLM_ASSERT(is<MemoryHoistBarrierOperation>(&memoryHoistBarrierNode));

  output.divertUsersWhere(
      *memoryHoistBarrierNode.output(0),
      [&memoryHoistBarrierNode](const rvsdg::Input & user)
      {
        return &MemoryHoistBarrierOperation::getAddressInput(memoryHoistBarrierNode) != &user;
      });
}

void
IOBarrierElimination::normalizeIOBarriers(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeWithDefault(
        node,
        [](rvsdg::StructuralNode & structuralNode)
        {
          for (auto & subregion : structuralNode.Subregions())
          {
            // Handle innermost regions first
            normalizeIOBarriers(subregion);

            // Normalize subregion arguments
            for (auto & argument : subregion.Arguments())
            {
              if (is<PointerType>(argument->Type()))
              {
                auto ioBarrierNodes = collectMemoryHoistBarrierNodes(*argument);
                if (auto ioBarrierNode = selectMemoryHoistBarrierNode(ioBarrierNodes, &subregion))
                  divertUsersToMemoryHoistBarrierNode(*argument, **ioBarrierNode);
              }
            }
          }

          // Normalize node outputs
          for (auto & output : structuralNode.Outputs())
          {
            if (is<PointerType>(output.Type()))
            {
              auto ioBarrierNodes = collectMemoryHoistBarrierNodes(output);
              if (auto ioBarrierNode =
                      selectMemoryHoistBarrierNode(ioBarrierNodes, &structuralNode))
                divertUsersToMemoryHoistBarrierNode(output, **ioBarrierNode);
            }
          }
        },
        [](rvsdg::SimpleNode & simpleNode)
        {
          rvsdg::MatchType(
              simpleNode.GetOperation(),
              [&simpleNode](const LoadNonVolatileOperation &)
              {
                auto & loadedValue = LoadOperation::LoadedValueOutput(simpleNode);
                if (is<PointerType>(loadedValue.Type()))
                {
                  auto ioBarrierNodes = collectMemoryHoistBarrierNodes(loadedValue);
                  if (auto ioBarrierNode =
                          selectMemoryHoistBarrierNode(ioBarrierNodes, simpleNode.region()))
                    divertUsersToMemoryHoistBarrierNode(loadedValue, **ioBarrierNode);
                }
              });
        },
        []()
        {
          throw std::logic_error("Unexpected node type");
        });
  }
}

size_t
IOBarrierElimination::areAllArgumentsMarked(const rvsdg::GammaNode::EntryVar & entryVar) const
{
  // We do not care about non-pointer entry variables
  if (!rvsdg::is<PointerType>(entryVar.input->Type()))
    return 0;

  size_t size = std::numeric_limits<std::size_t>::max();
  for (const auto * argument : entryVar.branchArgument)
  {
    if (const size_t numUsers = argument->nusers(); numUsers == 0)
    {
      // If we have no users, nothing can be marked
      return 0;
    }

    auto * user = &*argument->Users().begin();
    auto userSize = context_->isDereferenceable(*user);
    if (userSize == 0)
    {
      // The user is not marked. Let's continue
      auto [ioBarrierNode, ioBarrierOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(*user);
      if (!ioBarrierOp)
      {
        // We do not even have an IOBarrierOperation. We are done here.
        return 0;
      }

      JLM_ASSERT(ioBarrierNode->output(0)->nusers() != 0);
      // Any user of an IOBarrierOperation should do as we always mark all users
      user = &*ioBarrierNode->output(0)->Users().begin();
      userSize = context_->isDereferenceable(*user);
      if (userSize == 0)
      {
        // The user of the IOBarrierOperation is not marked either. We are done for good.
        return 0;
      }
    }

    // The user is marked. Let's take the minimum marked size between this argument and all others.
    size = std::min(size, userSize);
  }

  return size;
}

void
IOBarrierElimination::markDereferenceable(const rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeWithDefault(
        node,
        [this](const rvsdg::PhiNode & phiNode)
        {
          markDereferenceable(*phiNode.subregion());
        },
        [this](const rvsdg::LambdaNode & lambdaNode)
        {
          markDereferenceable(*lambdaNode.subregion());
        },
        [](const rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [this](const rvsdg::ThetaNode & thetaNode)
        {
          markDereferenceable(*thetaNode.subregion());
        },
        [this](const rvsdg::GammaNode & gammaNode)
        {
          // Handle innermost regions first
          for (auto & subregion : gammaNode.Subregions())
          {
            markDereferenceable(subregion);
          }

          for (auto & entryVar : gammaNode.GetEntryVars())
          {
            if (const auto size = areAllArgumentsMarked(entryVar); size > 0)
            {
              // All gamma node arguments of this entry variable are marked. This means that on
              // every path through this gamma node, the pointer variable is at least dereferenced
              // by the returned size. Consequently, we can mark the origin of the input of this
              // gamma node as well.
              context_->markUsersDereferenceable(*entryVar.input->origin(), size);
            }
          }
        },
        [this](const rvsdg::SimpleNode & simpleNode)
        {
          rvsdg::MatchType(
              simpleNode.GetOperation(),
              [this, &simpleNode](const LoadNonVolatileOperation & loadOperation)
              {
                const auto & addressOperand = *LoadOperation::AddressInput(simpleNode).origin();
                const auto sizeInBytes = GetTypeStoreSize(*loadOperation.GetLoadedType());
                context_->markUsersDereferenceable(addressOperand, sizeInBytes);
              },
              [this, &simpleNode](const StoreNonVolatileOperation & storeOperation)
              {
                const auto & addressOperand = *StoreOperation::AddressInput(simpleNode).origin();
                const auto sizeInBytes = GetTypeStoreSize(storeOperation.GetStoredType());
                context_->markUsersDereferenceable(addressOperand, sizeInBytes);
              });
        },
        []()
        {
          throw std::logic_error("Unhandled node type");
        });
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

              if (const auto size = context_->isDereferenceable(*input); size > 0)
              {
                for (const auto & argument : arguments)
                {
                  context_->markUsersDereferenceable(*argument, size);
                }
              }
            }

            for (auto & subregion : gammaNode.Subregions())
              propagate(subregion);

            for (auto & [results, output] : gammaNode.GetExitVars())
            {
              if (!is<PointerType>(output->Type()))
                continue;

              size_t sizeInBytes = std::numeric_limits<std::size_t>::max();
              for (const auto & result : results)
              {
                sizeInBytes = std::min(sizeInBytes, context_->isDereferenceable(*result));
                if (sizeInBytes == 0)
                {
                  break;
                }
              }
              if (sizeInBytes > 0)
                context_->markUsersDereferenceable(*output, sizeInBytes);
            }
          },
          [&](rvsdg::ThetaNode & thetaNode)
          {
            // FIXME: This fix-point algorithm could be improved in terms of performance.

            std::unordered_map<rvsdg::Output *, size_t> loopVarPreSizes;

            // Mark loop variables in subregion
            for (const auto & loopVar : thetaNode.GetLoopVars())
            {
              if (!is<PointerType>(loopVar.input->Type()))
                continue;

              if (const auto inputSize = context_->isDereferenceable(*loopVar.input); inputSize > 0)
              {
                loopVarPreSizes[loopVar.pre] = inputSize;
                context_->markUsersDereferenceable(*loopVar.pre, inputSize);
              }
            }

            // Propagate information through loop until fix-point is reached
            bool repeat = false;
            do
            {
              repeat = false;
              propagate(*thetaNode.subregion());

              for (const auto & loopVar : thetaNode.GetLoopVars())
              {
                if (!is<PointerType>(loopVar.input->Type()))
                  continue;

                const auto preSize = loopVarPreSizes[loopVar.pre];
                const auto postSize = context_->isDereferenceable(*loopVar.post);
                if (preSize != postSize)
                {
                  loopVarPreSizes[loopVar.pre] = postSize;
                  context_->markUsersDereferenceable(*loopVar.pre, std::min(preSize, postSize));
                  repeat = true;
                }
              }
            } while (repeat);

            // Mark loop outputs
            for (const auto & loopVar : thetaNode.GetLoopVars())
            {
              if (!is<PointerType>(loopVar.output->Type()))
                continue;

              if (const auto postSize = context_->isDereferenceable(*loopVar.post); postSize > 0)
              {
                context_->markUsersDereferenceable(*loopVar.output, postSize);
              }
            }
          },
          [&](rvsdg::DeltaNode &)
          {
            // Nothing needs to be done
          },
          [&](rvsdg::SimpleNode & simpleNode)
          {
            rvsdg::MatchType(
                simpleNode.GetOperation(),
                [this, &simpleNode](const MemoryHoistBarrierOperation &)
                {
                  const auto & barredInput =
                      MemoryHoistBarrierOperation::getAddressInput(simpleNode);
                  if (!is<PointerType>(barredInput.Type()))
                    return;

                  if (const auto size = context_->isDereferenceable(barredInput); size > 0)
                    context_->markUsersDereferenceable(*simpleNode.output(0), size);
                });
          },
          []()
          {
            throw std::logic_error(
                "Unhandled node type encountered during dereferenceable propagation.");
          });
    }
  };

  propagate(graph.GetRootRegion());
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
            auto [hoistBarrierNode, hoistBarrierOp] =
                rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryHoistBarrierOperation>(
                    *loadAddress.origin());
            if (!hoistBarrierOp)
              return;

            auto & barredAddressInput =
                MemoryHoistBarrierOperation::getAddressInput(*hoistBarrierNode);
            const auto barredAddressSize = context_->isDereferenceable(barredAddressInput);
            if (barredAddressSize == 0)
              return;

            if (const auto storeSize = GetTypeStoreSize(*loadOperation->GetLoadedType());
                barredAddressSize < storeSize)
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
