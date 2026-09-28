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
  const char * HoistingTimerLabel_ = "HoistingTime";

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
  startHoistingStatistics() noexcept
  {
    AddTimer(HoistingTimerLabel_).start();
  }

  void
  stopHoistingStatistics() noexcept
  {
    GetTimer(HoistingTimerLabel_).stop();
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

  statistics->startHoistingStatistics();
  hoistMemoryBarriers(rvsdg.GetRootRegion());
  statistics->stopHoistingStatistics();

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

class IOBarrierElimination::HoistContext final
{
public:
  void
  addTargetRegion(const rvsdg::Node & node, rvsdg::Region & region) noexcept
  {
    JLM_ASSERT(targetRegionMap_.find(&node) == targetRegionMap_.end());
    targetRegionMap_[&node] = &region;
  }

  rvsdg::Region *
  getTargetRegion(const rvsdg::Node & node) const noexcept
  {
    if (targetRegionMap_.find(&node) == targetRegionMap_.end())
      return nullptr;

    return targetRegionMap_.at(&node);
  }

  static std::unique_ptr<HoistContext>
  create()
  {
    return std::make_unique<HoistContext>();
  }

private:
  std::unordered_map<const rvsdg::Node *, rvsdg::Region *> targetRegionMap_{};
};

void
IOBarrierElimination::hoistMemoryBarriers(rvsdg::Region & region)
{
  hoistContext_ = HoistContext::create();
  computeTargetRegions(region);
  hoistNodes(region);
}

void
IOBarrierElimination::computeTargetRegions(const rvsdg::Region & region)
{
  for (const auto node : rvsdg::TopDownConstTraverser(&region))
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          for (auto & subregion : structuralNode.Subregions())
            computeTargetRegions(subregion);
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          rvsdg::MatchType(
              simpleNode.GetOperation(),
              [this, &simpleNode](const LoadNonVolatileOperation & loadOperation)
              {
                auto & loadAddress = LoadOperation::AddressInput(simpleNode);
                auto [mhbNode, mhbOp] =
                    rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryHoistBarrierOperation>(
                        *loadAddress.origin());
                if (!mhbOp)
                  return;

                auto & mhbAddressInput = MemoryHoistBarrierOperation::getAddressInput(*mhbNode);
                const auto mhbSize = context_->getDereferenceableSize(*mhbAddressInput.origin());
                if (mhbSize == 0)
                  return;

                if (const auto storeSize = GetTypeStoreSize(*loadOperation.GetLoadedType());
                    mhbSize >= storeSize)
                {
                  rvsdg::Region & targetRegion = computeTargetRegion(*mhbNode);
                  hoistContext_->addTargetRegion(*mhbNode, targetRegion);
                }
              });
        },
        []()
        {
          throw std::logic_error("Unhandled node type!");
        });
  }
}

rvsdg::Region &
IOBarrierElimination::computeTargetRegion(const rvsdg::Node & node) const
{
  // Compute target regions for all the inputs of the node
  rvsdg::Region * greatestCommonTargetRegion = nullptr;

  for (auto & input : node.Inputs())
  {
    auto & targetRegion = computeTargetRegion(*input.origin());
    if (&targetRegion == node.region())
    {
      // One of the node's predecessors cannot be hoisted, which means we can also not hoist this
      // node
      return *node.region();
    }

    // If we already have a common target region that is lower, keep it
    if (greatestCommonTargetRegion
        && greatestCommonTargetRegion->getDepth() >= targetRegion.getDepth())
      continue;
    greatestCommonTargetRegion = &targetRegion;
  }

  // Return the lowest-most common target region in the region tree among all inputs
  JLM_ASSERT(greatestCommonTargetRegion);
  return *greatestCommonTargetRegion;
}

rvsdg::Region &
IOBarrierElimination::computeTargetRegion(const rvsdg::Output & output) const
{
  // Handle lambda region arguments
  if (rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    return *output.region();
  }

  // Handle gamma region arguments
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    const auto roleVar = gammaNode->MapBranchArgument(output);
    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
    {
      return computeTargetRegion(*entryVar->input->origin());
    }

    return *output.region();
  }

  // Handle theta region arguments
  if (const auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    const auto loopVar = thetaNode->MapPreLoopVar(output);
    if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
    {
      return computeTargetRegion(*loopVar.input->origin());
    }

    if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
    {
      return computeTargetRegion(*loopVar.input->origin());
    }

    return *output.region();
  }

  // Handle gamma outputs
  if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    return *gammaNode->region();
  }

  // Handle theta outputs
  if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    return *thetaNode->region();
  }

  // Handle simple node outputs
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    auto targetRegion = hoistContext_->getTargetRegion(*node);
    return targetRegion != nullptr ? *targetRegion : *node->region();
  }

  throw std::logic_error("Unhandled output type!");
}

void
IOBarrierElimination::hoistNodes(rvsdg::Region & region) const
{
  for (auto & node : rvsdg::TopDownTraverser(&region))
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          hoistNodes(*lambdaNode.subregion());
        },
        [&](rvsdg::PhiNode & phiNode)
        {
          hoistNodes(*phiNode.subregion());
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [&](rvsdg::ThetaNode & thetaNode)
        {
          hoistNodes(*thetaNode.subregion());
        },
        [&](rvsdg::GammaNode & gammaNode)
        {
          for (auto & subregion : gammaNode.Subregions())
            hoistNodes(subregion);
        },
        [this](rvsdg::SimpleNode & simpleNode)
        {
          rvsdg::MatchType(
              simpleNode.GetOperation(),
              [this, &simpleNode](const MemoryHoistBarrierOperation &)
              {
                hoistNode(simpleNode);
              });
        },
        [&]()
        {
          throw std::logic_error(util::strfmt("Unhandled node type: ", node->DebugString()));
        });
  }

  region.prune(false);
}

rvsdg::Input &
IOBarrierElimination::getUserFromTargetRegion(rvsdg::Input & input, rvsdg::Region & targetRegion)
{
  if (input.region() == &targetRegion)
    return input;

  const auto & operand = *input.origin();

  // Handle gamma subregion arguments
  if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(operand))
  {
    const auto roleVar = gammaNode->MapBranchArgument(operand);
    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
    {
      return getUserFromTargetRegion(*entryVar->input, targetRegion);
    }
  }

  // Handle theta subregion arguments
  if (const auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(operand))
  {
    const auto loopVar = thetaNode->MapPreLoopVar(operand);
    JLM_ASSERT(rvsdg::ThetaLoopVarIsInvariant(loopVar));
    return getUserFromTargetRegion(*loopVar.input, targetRegion);
  }

  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(operand))
  {
    if (is<LoadNonVolatileOperation>(simpleNode->GetOperation()))
    {
      auto & memStateInput = LoadNonVolatileOperation::MapMemoryStateOutputToInput(operand);
      return getUserFromTargetRegion(memStateInput, targetRegion);
    }
  }

  throw std::logic_error("Unhandled output type!");
}

std::vector<rvsdg::Input *>
IOBarrierElimination::getUsersFromTargetRegion(rvsdg::Node & node, rvsdg::Region & targetRegion)
{
  std::vector<rvsdg::Input *> users;
  for (auto & input : node.Inputs())
  {
    auto & user = getUserFromTargetRegion(input, targetRegion);
    users.push_back(&user);
  }

  return users;
}

void
IOBarrierElimination::hoistNode(rvsdg::SimpleNode & mhbNode) const
{
  JLM_ASSERT(is<MemoryHoistBarrierOperation>(mhbNode.GetOperation()));

  auto targetRegion = hoistContext_->getTargetRegion(mhbNode);
  if (!targetRegion)
    return;

  const auto users = getUsersFromTargetRegion(mhbNode, *targetRegion);
  JLM_ASSERT(users.size() == 2);
  const auto addressUser = users[0];
  const auto ioStateUser = users[1];

  const auto hoistedMhbNode =
      mhbNode.copy(targetRegion, { addressUser->origin(), ioStateUser->origin() });

  addressUser->divert_to(&MemoryHoistBarrierOperation::getAddressOutput(*hoistedMhbNode));
  MemoryHoistBarrierOperation::getAddressOutput(mhbNode).divert_users(
      MemoryHoistBarrierOperation::getAddressInput(mhbNode).origin());
}

}
