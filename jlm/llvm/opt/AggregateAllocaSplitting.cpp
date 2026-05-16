/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
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
      // FIXME: We currently only look at alloca nodes that do not contain nested aggregate types.
      return false;
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

  util::HashSet<rvsdg::Output *> visited;
  util::HashSet<rvsdg::Output *> toVisit{ &address };
  auto addToVisitSet = [&](rvsdg::Output & output)
  {
    if (!visited.Contains(&output) && !toVisit.Contains(&output))
    {
      toVisit.insert(&output);
    }
  };
  auto removeFromVisitSet = [&]()
  {
    const auto output = *toVisit.Items().begin();
    toVisit.Remove(output);
    [[maybe_unused]] auto inserted = visited.insert(output);
    JLM_ASSERT(inserted);
    return output;
  };

  while (!toVisit.IsEmpty() && isSplitable)
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
                    for (size_t n = 1; n < userNode->ninputs(); n++)
                    {
                      if (!tryGetConstantSignedInteger(*userNode->input(n)->origin()).has_value())
                      {
                        return false;
                      }
                    }

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

  util::HashSet<rvsdg::Output *> visited;
  util::HashSet<rvsdg::Output *> toVisit{ &address };
  auto addToVisitSet = [&](rvsdg::Output & output)
  {
    if (!visited.Contains(&output) && !toVisit.Contains(&output))
    {
      toVisit.insert(&output);
    }
  };
  auto removeFromVisitSet = [&]()
  {
    const auto output = *toVisit.Items().begin();
    toVisit.Remove(output);
    [[maybe_unused]] auto inserted = visited.insert(output);
    JLM_ASSERT(inserted);
    return output;
  };

  while (!toVisit.IsEmpty() && hasOnlyLoadsAndStores)
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
                context_->numSplitAggregateAllocaNodes++;
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

void
AggregateAllocaSplitting::splitAllocaNode(const AllocaTraceInfo & allocaTraceInfo)
{
  auto & allocaNode = *allocaTraceInfo.allocaNode;
  const auto allocaOperation = dynamic_cast<const AllocaOperation *>(&allocaNode.GetOperation());
  JLM_ASSERT(allocaOperation && isSplitableType(*allocaOperation->allocatedType()));
  auto & allocaType = *std::static_pointer_cast<const StructType>(allocaOperation->allocatedType());
  const auto & countInput = AllocaOperation::getCountInput(allocaNode);
  const auto alignment = allocaOperation->alignment();

  // Create alloca nodes for each element in the aggregate type
  std::vector<rvsdg::Node *> elementAllocaNodes;
  std::vector<rvsdg::Output *> allocaMemoryStates;
  for (const auto & elementType : allocaType.elementTypes())
  {
    auto & elementAlloca =
        AllocaOperation::createNode(elementType, *countInput.origin(), alignment);
    elementAllocaNodes.push_back(&elementAlloca);
    allocaMemoryStates.push_back(&AllocaOperation::getMemoryStateOutput(elementAlloca));
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
          JLM_ASSERT(allocaConsumer->ninputs() == 3);
          auto & consumerRegion = *allocaConsumer->region();
          // FIXME: Introduce convenient functions
          [[maybe_unused]] auto index0 =
              tryGetConstantSignedInteger(*allocaConsumer->input(1)->origin()).value();
          const auto index1 =
              tryGetConstantSignedInteger(*allocaConsumer->input(2)->origin()).value();
          JLM_ASSERT(index0 == 0);

          auto elementAlloca = elementAllocaNodes[index1];
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
