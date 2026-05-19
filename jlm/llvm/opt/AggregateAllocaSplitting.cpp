/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/AggregateAllocaSplitting.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>

#include <deque>
#include <map>

namespace jlm::llvm
{

class AggregateAllocaSplitting::Statistics final : public util::Statistics
{
  const char * numAggregateAllocaNodesLabel_ = "#AggregateAllocaNodes";
  const char * numAggregateStructAllocaNodesLabel_ = "#AggregateStructAllocaNodes";
  const char * numSplitableTypeAggregateAllocaNodesLabel_ = "#SplitableTypeAggregateAllocaNodes";
  const char * numSplitAggregateAllocaNodesLabel_ = "#SplitAggregateAllocaNodes";
  const char * aggregateAllocaSplittingTimerLabel_ = "AggregateAllocaSplittingTime";

public:
  ~Statistics() noexcept override = default;

  explicit Statistics(util::FilePath filePath)
      : util::Statistics(Id::AggregateAllocaSplitting, std::move(filePath))
  {}

  void
  start()
  {
    AddTimer(aggregateAllocaSplittingTimerLabel_).start();
  }

  void
  stop(
      const size_t numAggregateAllocaNodes,
      const size_t numAggregateStructAllocaNodes,
      const size_t numSplitableTypeAggregateAllocaNodes,
      const size_t numSplitAggregateAllocaNodes)
  {
    GetTimer(aggregateAllocaSplittingTimerLabel_).stop();
    AddMeasurement(numAggregateAllocaNodesLabel_, numAggregateAllocaNodes);
    AddMeasurement(numAggregateStructAllocaNodesLabel_, numAggregateStructAllocaNodes);
    AddMeasurement(
        numSplitableTypeAggregateAllocaNodesLabel_,
        numSplitableTypeAggregateAllocaNodes);
    AddMeasurement(numSplitAggregateAllocaNodesLabel_, numSplitAggregateAllocaNodes);
  }

  static std::unique_ptr<Statistics>
  create(util::FilePath filePath)
  {
    return std::make_unique<Statistics>(std::move(filePath));
  }
};

struct AggregateAllocaSplitting::Context
{
  size_t numAggregateAllocaNodes = 0;
  size_t numAggregateStructAllocaNodes = 0;
  size_t numSplitableTypeAggregateAllocaNodes = 0;
  size_t numSplitAggregateAllocaNodes = 0;
};

struct AggregateAllocaSplitting::AllocaTraceInfo
{
  explicit AllocaTraceInfo(rvsdg::SimpleNode & allocaNode)
      : allocaNode(&allocaNode)
  {}

  rvsdg::SimpleNode * allocaNode;
  std::vector<rvsdg::SimpleNode *> allocaConsumers{};
};

AggregateAllocaSplitting::~AggregateAllocaSplitting() noexcept = default;

AggregateAllocaSplitting::AggregateAllocaSplitting()
    : Transformation("AggregateAllocaSplitting")
{}

bool
AggregateAllocaSplitting::isSplitableType(const rvsdg::Type & type)
{
  // FIXME: We currently only look at alloca nodes with a struct type. We might be able
  // to do something for alloca nodes with array types as well.
  const auto structType = dynamic_cast<const StructType *>(&type);
  if (!structType)
    return false;

  for (const auto & elementType : structType->elementTypes())
  {
    if (IsAggregateType(*elementType))
    {
      return isSplitableType(*elementType);
    }
  }

  return true;
}

std::optional<AggregateAllocaSplitting::AllocaTraceInfo>
AggregateAllocaSplitting::isSplitable(rvsdg::SimpleNode & allocaNode)
{
  [[maybe_unused]] auto allocaOperation =
      dynamic_cast<const AllocaOperation *>(&allocaNode.GetOperation());
  JLM_ASSERT(allocaOperation && isSplitableType(*allocaOperation->allocatedType()));

  auto & address = AllocaOperation::getPointerOutput(allocaNode);

  bool isSplitable = true;
  AllocaTraceInfo allocaTraceInfo(allocaNode);

  util::HashSet<rvsdg::Output *> seen;
  std::deque<rvsdg::Output *> toVisit{ &address };
  auto addToVisitSet = [&](rvsdg::Output & output)
  {
    if (!seen.Contains(&output))
    {
      toVisit.push_back(&output);
    }
    seen.insert(&output);
  };
  auto removeFromVisitSet = [&]()
  {
    const auto output = toVisit.front();
    toVisit.pop_front();
    return output;
  };

  while (!toVisit.empty() && isSplitable)
  {
    const auto currentOutput = removeFromVisitSet();

    for (auto & user : currentOutput->Users())
    {
      if (!isSplitable)
      {
        // Stop handling users if the previous user was already not splitable
        break;
      }

      if (auto userRegion = rvsdg::TryGetOwnerRegion(user))
      {
        // We should never have an alloca connected to a graph export
        JLM_ASSERT(userRegion->node());

        isSplitable = rvsdg::MatchTypeWithDefault(
            *userRegion->node(),
            [&](const rvsdg::GammaNode & gammaNode)
            {
              auto & gammaOutput = gammaNode.mapBranchResultToOutput(user);
              addToVisitSet(gammaOutput);
              return true;
            },
            [&](const rvsdg::ThetaNode & thetaNode)
            {
              const auto loopVar = thetaNode.MapPostLoopVar(user);
              addToVisitSet(*loopVar.pre);
              addToVisitSet(*loopVar.output);
              return true;
            },
            [&](const rvsdg::LambdaNode &)
            {
              return false;
            },
            [&]()
            {
              throw std::logic_error(util::strfmt(
                  "Unhandled owner region node type: ",
                  userRegion->node()->DebugString()));
              // Silence compiler
              return false;
            });
      }
      else if (auto userNode = rvsdg::TryGetOwnerNode<rvsdg::Node>(user))
      {
        isSplitable = rvsdg::MatchTypeWithDefault(
            *userNode,
            [&](rvsdg::GammaNode & gammaNode)
            {
              auto roleVar = gammaNode.MapInput(user);
              if (auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
              {
                for (auto argument : entryVar->branchArgument)
                {
                  addToVisitSet(*argument);
                }
              }
              else
              {
                throw std::logic_error(util::strfmt("Unhandled role variable."));
              }

              return true;
            },
            [&](rvsdg::ThetaNode & thetaNode)
            {
              const auto loopVar = thetaNode.MapInputLoopVar(user);
              addToVisitSet(*loopVar.pre);
              return true;
            },
            [&](rvsdg::SimpleNode & simpleNode)
            {
              auto & operation = simpleNode.GetOperation();
              return rvsdg::MatchTypeWithDefault(
                  operation,
                  [&](const GetElementPtrOperation &)
                  {
                    JLM_ASSERT(userNode->input(0) == &user);
                    JLM_ASSERT(
                        GetElementPtrOperation::tryGetConstantIndices(simpleNode).has_value());
                    allocaTraceInfo.allocaConsumers.push_back(&simpleNode);
                    return true;
                  },
                  [&]()
                  {
                    return false;
                  });
            },
            [&]()
            {
              throw std::logic_error(
                  util::strfmt("Unhandled node type: ", userNode->DebugString()));
              // Silence compiler
              return false;
            });
      }
      else
      {
        throw std::logic_error("Unhandled owner type");
      }
    }
  }

  if (!isSplitable)
    return std::nullopt;

  for (const auto allocaConsumer : allocaTraceInfo.allocaConsumers)
  {
    if (!checkGetElementPtrUsers(*allocaConsumer))
      return std::nullopt;
  }

  return std::make_optional(allocaTraceInfo);
}

bool
AggregateAllocaSplitting::checkGetElementPtrUsers(const rvsdg::SimpleNode & gepNode)
{
  [[maybe_unused]] auto gepOperation =
      dynamic_cast<const GetElementPtrOperation *>(&gepNode.GetOperation());
  auto & address = *gepNode.output(0);

  bool hasOnlyLoadsAndStores = true;

  util::HashSet<rvsdg::Output *> seen;
  std::deque<rvsdg::Output *> toVisit{ &address };
  auto addToVisitSet = [&](rvsdg::Output & output)
  {
    if (!seen.Contains(&output))
    {
      toVisit.push_back(&output);
    }
    seen.insert(&output);
  };
  auto removeFromVisitSet = [&]()
  {
    const auto output = toVisit.front();
    toVisit.pop_front();
    return output;
  };

  while (!toVisit.empty() && hasOnlyLoadsAndStores)
  {
    const auto currentOutput = removeFromVisitSet();
    for (auto & user : currentOutput->Users())
    {
      if (!hasOnlyLoadsAndStores)
      {
        // Stop handling users if the previous user was already not a load or store
        break;
      }

      if (auto userRegion = rvsdg::TryGetOwnerRegion(user))
      {
        // We should never have a gep node connected to a graph export
        JLM_ASSERT(userRegion->node());

        hasOnlyLoadsAndStores = rvsdg::MatchTypeWithDefault(
            *userRegion->node(),
            [&](const rvsdg::GammaNode & gammaNode)
            {
              auto & gammaOutput = gammaNode.mapBranchResultToOutput(user);
              addToVisitSet(gammaOutput);
              return true;
            },
            [&](const rvsdg::ThetaNode & thetaNode)
            {
              const auto loopVar = thetaNode.MapPostLoopVar(user);
              addToVisitSet(*loopVar.pre);
              addToVisitSet(*loopVar.output);
              return true;
            },
            [&](const rvsdg::LambdaNode &)
            {
              return false;
            },
            [&]()
            {
              throw std::logic_error(util::strfmt(
                  "Unhandled owner region node type: ",
                  userRegion->node()->DebugString()));
              // Silence compiler
              return false;
            });
      }
      else if (auto userNode = rvsdg::TryGetOwnerNode<rvsdg::Node>(user))
      {
        hasOnlyLoadsAndStores = rvsdg::MatchTypeWithDefault(
            *userNode,
            [&](const rvsdg::GammaNode & gammaNode)
            {
              auto roleVar = gammaNode.MapInput(user);
              if (auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
              {
                for (auto argument : entryVar->branchArgument)
                {
                  addToVisitSet(*argument);
                }
              }
              else
              {
                throw std::logic_error(util::strfmt("Unhandled role variable."));
              }

              return true;
            },
            [&](const rvsdg::ThetaNode & thetaNode)
            {
              const auto loopVar = thetaNode.MapInputLoopVar(user);
              addToVisitSet(*loopVar.pre);
              return true;
            },
            [&](const rvsdg::SimpleNode & simpleNode)
            {
              auto & operation = simpleNode.GetOperation();
              return rvsdg::MatchTypeWithDefault(
                  operation,
                  [&](const LoadOperation &)
                  {
                    return true;
                  },
                  [&](const StoreOperation &)
                  {
                    if (&user != &StoreOperation::AddressInput(simpleNode))
                      return false;

                    return true;
                  },
                  [&](const IOBarrierOperation &)
                  {
                    addToVisitSet(*simpleNode.output(0));
                    return true;
                  },
                  [&]()
                  {
                    return false;
                  });
            },
            [&]()
            {
              throw std::logic_error(
                  util::strfmt("Unhandled node type: ", userNode->DebugString()));
              // Silence compiler
              return false;
            });
      }
      else
      {
        throw std::logic_error("Unhandled owner type");
      }
    }
  }

  return hasOnlyLoadsAndStores;
}

std::vector<AggregateAllocaSplitting::AllocaTraceInfo>
AggregateAllocaSplitting::findSplitableAllocaNodes(rvsdg::Region & region) const
{
  std::function<void(rvsdg::Region &, std::vector<AllocaTraceInfo> &)> findAllocaNodes =
      [&](rvsdg::Region & region, std::vector<AllocaTraceInfo> & traceInfo)
  {
    for (auto & node : region.Nodes())
    {
      MatchTypeWithDefault(
          node,
          [&](rvsdg::GammaNode & gammaNode)
          {
            for (auto & subregion : gammaNode.Subregions())
              findAllocaNodes(subregion, traceInfo);
          },
          [&](rvsdg::ThetaNode & thetaNode)
          {
            findAllocaNodes(*thetaNode.subregion(), traceInfo);
          },
          [&](rvsdg::LambdaNode & lambdaNode)
          {
            findAllocaNodes(*lambdaNode.subregion(), traceInfo);
          },
          [&](rvsdg::PhiNode & phiNode)
          {
            findAllocaNodes(*phiNode.subregion(), traceInfo);
          },
          [&](rvsdg::DeltaNode &)
          {
            // Nothing needs to be done
          },
          [&](rvsdg::SimpleNode & simpleNode)
          {
            const auto allocaOperation =
                dynamic_cast<const AllocaOperation *>(&simpleNode.GetOperation());
            if (!allocaOperation)
              return;

            auto & allocaType = *allocaOperation->allocatedType();
            if (is<StructType>(allocaType))
            {
              context_->numAggregateStructAllocaNodes++;
              context_->numAggregateAllocaNodes++;
            }
            else if (IsAggregateType(allocaType))
            {
              context_->numAggregateAllocaNodes++;
            }

            if (isSplitableType(*allocaOperation->allocatedType()))
            {
              context_->numSplitableTypeAggregateAllocaNodes++;
              if (auto allocaTraceInfo = isSplitable(simpleNode))
              {
                traceInfo.emplace_back(*allocaTraceInfo);
              }
            }
          },
          [&]()
          {
            throw std::logic_error("Unhandled node type.");
          });
    }
  };

  std::vector<AllocaTraceInfo> traceInfo;
  findAllocaNodes(region, traceInfo);
  return traceInfo;
}

// FIXME: Do not use std::map, and rather use std::unordered_map
static std::map<std::vector<uint64_t>, rvsdg::Node *>
createElementAllocaNodes(rvsdg::SimpleNode & allocaNode)
{
  const auto allocaOperation =
      util::assertedCast<const AllocaOperation>(&allocaNode.GetOperation());
  auto & allocaType = *util::assertedCast<const StructType>(allocaOperation->allocatedType().get());
  const auto & countInput = AllocaOperation::getCountInput(allocaNode);
  const auto alignment = allocaOperation->alignment();

  std::function<void(
      const StructType &,
      std::map<std::vector<uint64_t>, rvsdg::Node *> &,
      std::vector<uint64_t> &)>
      createAllocaNodes = [&](const StructType & structType,
                              std::map<std::vector<uint64_t>, rvsdg::Node *> & allocaNodes,
                              std::vector<uint64_t> & indices)
  {
    size_t index = 0;
    for (const auto & elementType : structType.elementTypes())
    {
      indices.push_back(index++);
      if (auto structType = std::dynamic_pointer_cast<const StructType>(elementType))
      {
        createAllocaNodes(*structType, allocaNodes, indices);
      }
      else
      {
        auto & elementAlloca =
            AllocaOperation::createNode(elementType, *countInput.origin(), alignment);

        allocaNodes[indices] = &elementAlloca;
      }

      indices.pop_back();
    }
  };

  std::vector<uint64_t> indices(1, 0);
  std::map<std::vector<uint64_t>, rvsdg::Node *> allocaNodes;
  createAllocaNodes(allocaType, allocaNodes, indices);
  return allocaNodes;
}

void
AggregateAllocaSplitting::splitAllocaNode(const AllocaTraceInfo & allocaTraceInfo)
{
  auto & allocaNode = *allocaTraceInfo.allocaNode;
  const auto allocaOperation = dynamic_cast<const AllocaOperation *>(&allocaNode.GetOperation());
  JLM_ASSERT(allocaOperation && isSplitableType(*allocaOperation->allocatedType()));

  std::vector<rvsdg::Output *> allocaMemoryStates;
  auto elementAllocaMap = createElementAllocaNodes(allocaNode);
  for (auto [_, elementAllocaNode] : elementAllocaMap)
  {
    allocaMemoryStates.push_back(&AllocaOperation::getMemoryStateOutput(*elementAllocaNode));
  }

  // Replace alloca node's memory state output
  const auto memoryState = MemoryStateMergeOperation::Create(allocaMemoryStates);
  AllocaOperation::getMemoryStateOutput(allocaNode).divert_users(memoryState);

  // Replace alloca node consumers
  for (auto allocaConsumer : allocaTraceInfo.allocaConsumers)
  {
    rvsdg::MatchTypeWithDefault(
        allocaConsumer->GetOperation(),
        [&](const GetElementPtrOperation &)
        {
          JLM_ASSERT(GetElementPtrOperation::numIndices(*allocaConsumer) >= 2);
          auto & consumerRegion = *allocaConsumer->region();
          const auto indices =
              GetElementPtrOperation::tryGetConstantIndices(*allocaConsumer).value();
          JLM_ASSERT(indices[0] == 0);

          auto elementAlloca = elementAllocaMap.at(indices);
          // FIXME: Introduce caching of routed values to avoid duplicated routing.
          auto & routedAddress = rvsdg::RouteToRegion(
              AllocaOperation::getPointerOutput(*elementAlloca),
              consumerRegion);
          allocaConsumer->output(0)->divert_users(&routedAddress);
        },
        [&]()
        {
          throw std::logic_error(
              util::strfmt("Unhandled node type: ", allocaConsumer->DebugString()));
        });
  }
}

void
AggregateAllocaSplitting::splitAllocaNodes(rvsdg::RvsdgModule & rvsdgModule)
{
  const auto traceInfo = findSplitableAllocaNodes(rvsdgModule.Rvsdg().GetRootRegion());
  for (const auto & allocaTraceInfo : traceInfo)
  {
    splitAllocaNode(allocaTraceInfo);
    context_->numSplitAggregateAllocaNodes++;
  }

  // Remove all nodes that became dead throughout the transformation
  rvsdgModule.Rvsdg().PruneNodes();
}

void
AggregateAllocaSplitting::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  context_ = std::make_unique<Context>();
  auto statistics = Statistics::create(module.SourceFilePath().value());

  statistics->start();
  splitAllocaNodes(module);
  statistics->stop(
      context_->numAggregateAllocaNodes,
      context_->numAggregateStructAllocaNodes,
      context_->numSplitableTypeAggregateAllocaNodes,
      context_->numSplitAggregateAllocaNodes);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  context_.reset();
}

}
