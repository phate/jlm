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
  const char * EncodeTimerLabel_ = "EncodeTime";
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
  startEncodeStatistics() noexcept
  {
    AddTimer(EncodeTimerLabel_).start();
  }

  void
  stopEncodeStatistics() noexcept
  {
    GetTimer(EncodeTimerLabel_).stop();
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
   */
  void
  markDereferenceable(const rvsdg::Output & output, const size_t sizeInBytes)
  {
    if (const auto it = dereferenceableInputs_.find(&output); it == dereferenceableInputs_.end())
    {
      dereferenceableInputs_[&output] = sizeInBytes;
    }
    else
    {
      dereferenceableInputs_[&output] = std::max(it->second, sizeInBytes);
    }
  }

  /**
   * @return The size in bytes. If \p output was not marked as dereferenceable, then 0 is returned.
   */
  [[nodiscard]] size_t
  getDereferenceableSize(const rvsdg::Output & output) const
  {
    const auto it = dereferenceableInputs_.find(&output);
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
  std::unordered_map<const rvsdg::Output *, size_t> dereferenceableInputs_{};
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
  normalizeMemoryHoistBarriers(rvsdg.GetRootRegion());
  statistics->stopNormalizationStatistics();

  statistics->startMarkStatistics();
  markOutputs(rvsdg.GetRootRegion());
  statistics->stopMarkStatistics();

  statistics->startPropagateStatistics();
  propagateSize(rvsdg);
  statistics->stopPropagateStatistics();

  statistics->startEncodeStatistics();
  encodeSize(rvsdg);
  statistics->stopEncodeStatistics();

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
IOBarrierElimination::normalizeMemoryHoistBarriers(rvsdg::Region & region)
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
            normalizeMemoryHoistBarriers(subregion);

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

void
IOBarrierElimination::markOutputs(const rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeWithDefault(
        node,
        [this](const rvsdg::PhiNode & phiNode)
        {
          markOutputs(*phiNode.subregion());
        },
        [this](const rvsdg::LambdaNode & lambdaNode)
        {
          // Mark lambda arguments
          for (const auto argument : lambdaNode.GetFunctionArguments())
          {
            if (rvsdg::is<PointerType>(argument->Type()))
            {
              context_->markDereferenceable(*argument, 0);
            }
          }

          // Mark lambda context variables
          for (const auto [_, inner] : lambdaNode.GetContextVars())
          {
            if (rvsdg::is<PointerType>(inner->Type()))
            {
              // FIXME: We can do better here
              context_->markDereferenceable(*inner, 0);
            }
          }

          markOutputs(*lambdaNode.subregion());
        },
        [](const rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [this](const rvsdg::ThetaNode & thetaNode)
        {
          markOutputs(*thetaNode.subregion());
        },
        [this](const rvsdg::GammaNode & gammaNode)
        {
          for (auto & subregion : gammaNode.Subregions())
          {
            markOutputs(subregion);
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
                context_->markDereferenceable(addressOperand, sizeInBytes);
              },
              [this, &simpleNode](const StoreNonVolatileOperation & storeOperation)
              {
                const auto & addressOperand = *StoreOperation::AddressInput(simpleNode).origin();
                const auto sizeInBytes = GetTypeStoreSize(storeOperation.GetStoredType());
                context_->markDereferenceable(addressOperand, sizeInBytes);
              });
        },
        []()
        {
          throw std::logic_error("Unhandled node type");
        });
  }
}

void
IOBarrierElimination::propagateSize(rvsdg::Graph & graph)
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

              if (const auto size = context_->getDereferenceableSize(*input->origin()); size > 0)
              {
                for (const auto & argument : arguments)
                {
                  context_->markDereferenceable(*argument, size);
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
                sizeInBytes =
                    std::min(sizeInBytes, context_->getDereferenceableSize(*result->origin()));
                if (sizeInBytes == 0)
                {
                  break;
                }
              }
              if (sizeInBytes > 0)
                context_->markDereferenceable(*output, sizeInBytes);
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

              if (const auto inputSize = context_->getDereferenceableSize(*loopVar.input->origin());
                  inputSize > 0)
              {
                loopVarPreSizes[loopVar.pre] = inputSize;
                context_->markDereferenceable(*loopVar.pre, inputSize);
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
                const auto postSize = context_->getDereferenceableSize(*loopVar.post->origin());
                if (preSize != postSize)
                {
                  loopVarPreSizes[loopVar.pre] = postSize;
                  context_->markDereferenceable(*loopVar.pre, std::min(preSize, postSize));
                  repeat = true;
                }
              }
            } while (repeat);

            // Mark loop outputs
            for (const auto & loopVar : thetaNode.GetLoopVars())
            {
              if (!is<PointerType>(loopVar.output->Type()))
                continue;

              if (const auto postSize = context_->getDereferenceableSize(*loopVar.post->origin());
                  postSize > 0)
              {
                context_->markDereferenceable(*loopVar.output, postSize);
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

                  if (const auto size = context_->getDereferenceableSize(*barredInput.origin());
                      size > 0)
                    context_->markDereferenceable(*simpleNode.output(0), size);
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
IOBarrierElimination::encodeSize(rvsdg::Graph & graph)
{
  std::function<void(rvsdg::Region &)> encode = [&](rvsdg::Region & region)
  {
    for (auto & node : region.Nodes())
    {
      rvsdg::MatchTypeWithDefault(
          node,
          [&](rvsdg::PhiNode & phiNode)
          {
            encode(*phiNode.subregion());
          },
          [&](rvsdg::LambdaNode & lambdaNode)
          {
            encode(*lambdaNode.subregion());
          },
          [&](rvsdg::GammaNode & gammaNode)
          {
            for (auto & subregion : gammaNode.Subregions())
              encode(subregion);
          },
          [&](rvsdg::ThetaNode & thetaNode)
          {
            encode(*thetaNode.subregion());
          },
          [&](rvsdg::DeltaNode &)
          {
            // Nothing needs to be done
          },
          [&](rvsdg::SimpleNode & simpleNode)
          {
            rvsdg::MatchType(
                simpleNode.GetOperation(),
                [this, &simpleNode](const MemoryHoistBarrierOperation & memoryHoistBarrier)
                {
                  auto & addressOperand =
                      *MemoryHoistBarrierOperation::getAddressInput(simpleNode).origin();
                  const auto mhbSize = memoryHoistBarrier.getDereferenceableSize();

                  if (const auto size = context_->getDereferenceableSize(addressOperand);
                      size != mhbSize)
                  {
                    auto & ioStateOperand =
                        *MemoryHoistBarrierOperation::getIOStateInput(simpleNode).origin();
                    auto & mhbNode = MemoryHoistBarrierOperation::createNode(
                        addressOperand,
                        ioStateOperand,
                        size);
                    MemoryHoistBarrierOperation::getAddressOutput(simpleNode)
                        .divert_users(&MemoryHoistBarrierOperation::getAddressOutput(mhbNode));
                  }
                });
          },
          []()
          {
            throw std::logic_error("Unhandled node type encountered during encoding.");
          });
    }

    region.prune(false);
  };

  encode(graph.GetRootRegion());
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
            const auto barredAddressSize =
                context_->getDereferenceableSize(*barredAddressInput.origin());
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
