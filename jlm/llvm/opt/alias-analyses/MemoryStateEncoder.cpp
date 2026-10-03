/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Statistics.hpp>

#include <unordered_map>

namespace jlm::llvm::aa
{

namespace
{
/**
 * \brief Helper struct for counting up MemoryNodes, among some set of entities that use them
 */
struct MemoryStateTypeCounter final
{
  // The number of entities that have been counted
  uint64_t NumEntities = 0;

  // Count of total memory states, separated by Ref/Mod/ModRef
  uint64_t NumRefOnly = 0;
  uint64_t NumModOnly = 0;
  uint64_t NumModRef = 0;

  // Count of total memory states, separated by MemoryNode type
  uint64_t NumAllocas = 0;
  uint64_t NumMallocs = 0;
  uint64_t NumDeltas = 0;
  uint64_t NumImports = 0;
  uint64_t NumLambdas = 0;
  uint64_t NumExternalNode = 0;

  // Count of the total memory states, how many are not externally available
  uint64_t NumNonEscaped = 0;

  // Remember the single entity with the highest number of memory states
  uint64_t MaxMemoryStateEntity = 0;
  // Do the same, but only include non-escaped MemoryNodes
  uint64_t MaxNonEscapedMemoryStateEntity = 0;

  void
  CountEntity(
      uint64_t numRefOnly,
      uint64_t numModOnly,
      uint64_t numModRef,
      uint64_t numAllocas,
      uint64_t numMallocs,
      uint64_t numDeltas,
      uint64_t numImports,
      uint64_t numLambdas,
      uint64_t numExternalNode,
      uint64_t numNonEscaped)
  {
    NumEntities++;

    NumRefOnly += numRefOnly;
    NumModOnly += numModOnly;
    NumModRef += numModRef;

    NumAllocas += numAllocas;
    NumMallocs += numMallocs;
    NumDeltas += numDeltas;
    NumImports += numImports;
    NumLambdas += numLambdas;
    NumExternalNode += numExternalNode;

    const uint64_t totalMemoryStates = numRefOnly + numModOnly + numModRef;
    if (totalMemoryStates > MaxMemoryStateEntity)
      MaxMemoryStateEntity = totalMemoryStates;

    NumNonEscaped += numNonEscaped;
    if (numNonEscaped > MaxNonEscapedMemoryStateEntity)
      MaxNonEscapedMemoryStateEntity = numNonEscaped;
  }

  void
  CountEntity(const PointsToGraph & pointsToGraph, const ModRefSet & memoryNodes)
  {
    uint64_t numRefOnly = 0;
    uint64_t numModOnly = 0;
    uint64_t numModRef = 0;

    uint64_t numAllocas = 0;
    uint64_t numMallocs = 0;
    uint64_t numDeltas = 0;
    uint64_t numImports = 0;
    uint64_t numLambdas = 0;
    uint64_t numExternalNode = 0;

    uint64_t numNonEscaped = 0;

    for (const auto [memoryNode, modRefEffect] : memoryNodes.getModRefNodes())
    {
      switch (modRefEffect)
      {
      case jlm::llvm::aa::ModRefEffect::RefOnly:
        numRefOnly++;
        break;
      case jlm::llvm::aa::ModRefEffect::ModOnly:
        numModOnly++;
        break;
      case jlm::llvm::aa::ModRefEffect::ModRef:
        numModRef++;
        break;
      default:
        JLM_UNREACHABLE("Unknown ModRefEffect");
      }

      if (!pointsToGraph.isExternallyAvailable(memoryNode))
        numNonEscaped++;

      const auto kind = pointsToGraph.getNodeKind(memoryNode);
      switch (kind)
      {
      case PointsToGraph::NodeKind::AllocaNode:
        numAllocas++;
        break;
      case PointsToGraph::NodeKind::DeltaNode:
        numDeltas++;
        break;
      case PointsToGraph::NodeKind::LambdaNode:
        numLambdas++;
        break;
      case PointsToGraph::NodeKind::ImportNode:
        numImports++;
        break;
      case PointsToGraph::NodeKind::MallocNode:
        numMallocs++;
        break;
      case PointsToGraph::NodeKind::ExternalNode:
        numExternalNode++;
        break;
      default:
        throw std::logic_error("Unknown MemoryNode kind");
      }
    }

    CountEntity(
        numRefOnly,
        numModOnly,
        numModRef,
        numAllocas,
        numMallocs,
        numDeltas,
        numImports,
        numLambdas,
        numExternalNode,
        numNonEscaped);
  }
};

/** \brief Statistics class for memory state encoder encoding
 *
 */
class EncodingStatistics final : public util::Statistics
{
  // Prefixes for statistics that count ModRef vs RefOnly
  static constexpr auto NumTotalRefOnlyStates_ = "#TotalRefOnlyState";
  static constexpr auto NumTotalModOnlyStates_ = "#TotalModOnlyState";
  static constexpr auto NumTotalModRefStates_ = "#TotalModRefState";
  // These are prefixes for statistics that count MemoryNode types
  static constexpr auto NumTotalAllocaState_ = "#TotalAllocaState";
  static constexpr auto NumTotalMallocState_ = "#TotalMallocState";
  static constexpr auto NumTotalDeltaState_ = "#TotalDeltaState";
  static constexpr auto NumTotalImportState_ = "#TotalImportState";
  static constexpr auto NumTotalLambdaState_ = "#TotalLambdaState";
  static constexpr auto NumTotalExternalNodeState_ = "#TotalExternalNodeState";
  // Among all the MemoryNodes counted above, how many of them are not externally available
  static constexpr auto NumTotalNonEscapedState_ = "#TotalNonEscapedState";
  // Maximums in a single counted entity
  static constexpr auto NumMaxMemoryState_ = "#MaxMemoryState";
  static constexpr auto NumMaxNonEscapedMemoryState_ = "#MaxNonEscapedMemoryState";

  // The number of regions that are inside lambda nodes (including the lambda subregion itself)
  static constexpr auto NumIntraProceduralRegions_ = "#IntraProceduralRegions";
  // Suffix used when counting region state arguments (or LambdaEntrySplit for lambda subregions)
  static constexpr auto RegionArgumentStateSuffix_ = "Arguments";

  // Counting both volatile and non-volatile loads
  static constexpr auto NumLoadOperations_ = "#LoadOperations";
  // Suffix used when counting memory states routed through loads
  static constexpr auto LoadStateSuffix_ = "sThroughLoad";

  // Counting both volatile and non-volatile stores
  static constexpr auto NumStoreOperations_ = "#StoreOperations";
  // Suffix used when counting memory states routed through stores
  static constexpr auto StoreStateSuffix_ = "sThroughStore";

  // Counting call entry merges
  static constexpr auto NumCallEntryMergeOperations_ = "#CallEntryMergeOperations";
  // Suffix used when counting memory states routed into call entry merges
  static constexpr auto CallEntryMergeStateSuffix_ = "sIntoCallEntryMerge";

public:
  ~EncodingStatistics() override = default;

  explicit EncodingStatistics(const util::FilePath & sourceFile)
      : Statistics(Statistics::Id::MemoryStateEncoder, sourceFile)
  {}

  void
  Start(const rvsdg::Graph & graph)
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  Stop()
  {
    GetTimer(Label::Timer).stop();
  }

  void
  AddIntraProceduralRegionMemoryStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumIntraProceduralRegions_, counter.NumEntities);
    AddMemoryStateTypeCounter(RegionArgumentStateSuffix_, counter);
  }

  void
  AddLoadMemoryStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumLoadOperations_, counter.NumEntities);
    AddMemoryStateTypeCounter(LoadStateSuffix_, counter);
  }

  void
  AddStoreMemoryStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumStoreOperations_, counter.NumEntities);
    AddMemoryStateTypeCounter(StoreStateSuffix_, counter);
  }

  void
  AddCallEntryMergeStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumCallEntryMergeOperations_, counter.NumEntities);
    AddMemoryStateTypeCounter(CallEntryMergeStateSuffix_, counter);
  }

  static std::unique_ptr<EncodingStatistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<EncodingStatistics>(sourceFile);
  }

private:
  void
  AddMemoryStateTypeCounter(const std::string & suffix, const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumTotalRefOnlyStates_ + suffix, counter.NumRefOnly);
    AddMeasurement(NumTotalModOnlyStates_ + suffix, counter.NumModOnly);
    AddMeasurement(NumTotalModRefStates_ + suffix, counter.NumModRef);

    AddMeasurement(NumTotalAllocaState_ + suffix, counter.NumAllocas);
    AddMeasurement(NumTotalMallocState_ + suffix, counter.NumMallocs);
    AddMeasurement(NumTotalDeltaState_ + suffix, counter.NumDeltas);
    AddMeasurement(NumTotalImportState_ + suffix, counter.NumImports);
    AddMeasurement(NumTotalLambdaState_ + suffix, counter.NumLambdas);
    AddMeasurement(NumTotalExternalNodeState_ + suffix, counter.NumExternalNode);
    AddMeasurement(NumTotalNonEscapedState_ + suffix, counter.NumNonEscaped);

    AddMeasurement(NumMaxMemoryState_ + suffix, counter.MaxMemoryStateEntity);
    AddMeasurement(NumMaxNonEscapedMemoryState_ + suffix, counter.MaxNonEscapedMemoryStateEntity);
  }
};

}

struct MemoryStateEncoder::MemoryStateTypeCounters final
{
  MemoryStateTypeCounter interProceduralRegionCounter;
  MemoryStateTypeCounter loadCounter;
  MemoryStateTypeCounter storeCounter;
  MemoryStateTypeCounter callEntryMergeCounter;
};

/** \brief Hash map for mapping points-to graph memory nodes to RVSDG memory states.
 */
class MemoryStateEncoder::StateMap final
{
public:
  /**
   * Represents the pairing of a points-to graph's memory node and a memory state.
   */
  class MemoryNodeStatePair final
  {
    friend StateMap;

    MemoryNodeStatePair(PointsToGraph::NodeIndex memoryNode, rvsdg::Output & state)
        : MemoryNode_(memoryNode),
          State_(&state)
    {
      JLM_ASSERT(is<MemoryStateType>(state.Type()));
    }

  public:
    [[nodiscard]] PointsToGraph::NodeIndex
    MemoryNode() const noexcept
    {
      return MemoryNode_;
    }

    [[nodiscard]] rvsdg::Output &
    State() const noexcept
    {
      return *State_;
    }

    void
    ReplaceState(rvsdg::Output & state) noexcept
    {
      JLM_ASSERT(State_->region() == state.region());
      JLM_ASSERT(is<MemoryStateType>(state.Type()));

      State_ = &state;
    }

    static void
    ReplaceStates(
        const std::vector<MemoryNodeStatePair *> & memoryNodeStatePairs,
        const std::vector<rvsdg::Output *> & states)
    {
      JLM_ASSERT(memoryNodeStatePairs.size() == states.size());
      for (size_t n = 0; n < memoryNodeStatePairs.size(); n++)
        memoryNodeStatePairs[n]->ReplaceState(*states[n]);
    }

    static void
    ReplaceStates(
        const std::vector<MemoryNodeStatePair *> & memoryNodeStatePairs,
        const rvsdg::Node::OutputIteratorRange & states)
    {
      auto it = states.begin();
      for (auto memoryNodeStatePair : memoryNodeStatePairs)
      {
        memoryNodeStatePair->ReplaceState(*it);
        it++;
      }
      JLM_ASSERT(it.GetOutput() == nullptr);
    }

    static std::vector<rvsdg::Output *>
    States(const std::vector<MemoryNodeStatePair *> & memoryNodeStatePairs)
    {
      std::vector<rvsdg::Output *> states;
      for (auto & memoryNodeStatePair : memoryNodeStatePairs)
        states.push_back(memoryNodeStatePair->State_);

      return states;
    }

  private:
    PointsToGraph::NodeIndex MemoryNode_;
    rvsdg::Output * State_;
  };

  StateMap() = default;

  StateMap(const StateMap &) = delete;

  StateMap(StateMap &&) = delete;

  StateMap &
  operator=(const StateMap &) = delete;

  StateMap &
  operator=(StateMap &&) = delete;

  MemoryNodeStatePair *
  TryGetState(PointsToGraph::NodeIndex memoryNode) noexcept
  {
    if (const auto it = states_.find(memoryNode); it != states_.end())
      return &it->second;

    return nullptr;
  }

  MemoryNodeStatePair *
  GetState(PointsToGraph::NodeIndex memoryNode)
  {
    if (const auto statePair = TryGetState(memoryNode))
      return statePair;
    throw std::logic_error("Memory node does not have a state.");
  }

  std::vector<MemoryNodeStatePair *>
  GetStates(const ModRefSet & modRefSet)
  {
    std::vector<MemoryNodeStatePair *> memoryNodeStatePairs;
    for (const auto [memoryNode, modRefEffect] : modRefSet.getModRefNodes())
    {
      JLM_ASSERT(modRefEffect != ModRefEffect::NoEffect);
      memoryNodeStatePairs.push_back(GetState(memoryNode));
    }

    return memoryNodeStatePairs;
  }

  /**
   * Gets MemoryNodeStatePairs for each of the given memory nodes,
   * unless there is no memory state in the region representing the memory node.
   * @param modRefSet the set of memory nodes to retrieve states for.
   * @return The MemoryNodeStatePairs for each given memory nodes, if one exists.
   * @see RegionalizedStateMap::GetExistingStates()
   */
  std::vector<MemoryNodeStatePair *>
  GetExistingStates(const ModRefSet & modRefSet)
  {
    std::vector<MemoryNodeStatePair *> memoryNodeStatePairs;
    for (auto & [memoryNode, _] : modRefSet.getModRefNodes())
    {
      if (const auto statePair = TryGetState(memoryNode))
        memoryNodeStatePairs.push_back(statePair);
    }

    return memoryNodeStatePairs;
  }

  /**
   * Creates a new memory node / memory state pair in the region.
   * The memory node must not have an already associated state.
   * @param memoryNode the memory node
   * @param state the output that produces the memory state associated with the memory node
   * @return pointer to the new pair
   */
  MemoryNodeStatePair *
  InsertState(PointsToGraph::NodeIndex memoryNode, rvsdg::Output & state)
  {
    auto [it, added] = states_.insert({ memoryNode, { memoryNode, state } });
    if (!added)
      throw std::logic_error("Memory node already has a state.");
    return &it->second;
  }

private:
  // std::unordered_map guarantees pointers to keys and values remain valid even when
  // new pairs are added to the container.
  std::unordered_map<PointsToGraph::NodeIndex, MemoryNodeStatePair> states_;
};

static std::vector<MemoryNodeId>
GetMemoryNodeIds(const ModRefSet & modRefSet)
{
  std::vector<MemoryNodeId> memoryNodeIds;
  for (const auto [memoryNode, _] : modRefSet.getModRefNodes())
  {
    memoryNodeIds.push_back(memoryNode);
  }

  return memoryNodeIds;
}

MemoryStateEncoder::~MemoryStateEncoder() noexcept = default;

MemoryStateEncoder::MemoryStateEncoder() = default;

void
MemoryStateEncoder::Encode(
    rvsdg::RvsdgModule & rvsdgModule,
    const ModRefSummary & modRefSummary,
    util::StatisticsCollector & statisticsCollector)
{
  modRefSummary_ = &modRefSummary;
  auto statistics = EncodingStatistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->Start(rvsdgModule.Rvsdg());
  // FIXME: separate handling of inter- and intra-procedural nodes to avoid stateMap parameter for
  // inter-procedural subregions
  StateMap stateMap;
  EncodeRegion(rvsdgModule.Rvsdg().GetRootRegion(), stateMap);
  statistics->Stop();

  if (statisticsCollector.IsDemanded(util::Statistics::Id::MemoryStateEncoder))
  {
    const auto counters = gatherStatistics(rvsdgModule.Rvsdg().GetRootRegion());

    statistics->AddIntraProceduralRegionMemoryStateCounts(counters->interProceduralRegionCounter);
    statistics->AddLoadMemoryStateCounts(counters->loadCounter);
    statistics->AddStoreMemoryStateCounts(counters->storeCounter);
    statistics->AddCallEntryMergeStateCounts(counters->callEntryMergeCounter);
  }

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Remove all nodes that became dead throughout the encoding.
  DeadNodeElimination deadNodeElimination;
  deadNodeElimination.Run(rvsdgModule, statisticsCollector);
}

void
MemoryStateEncoder::EncodeRegion(rvsdg::Region & region, StateMap & stateMap) const
{
  for (const auto node : rvsdg::TopDownTraverser(&region))
  {
    MatchTypeOrFail(
        *node,
        [this, &stateMap](rvsdg::PhiNode & phiNode)
        {
          EncodeRegion(*phiNode.subregion(), stateMap);
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [this](rvsdg::LambdaNode & lambdaNode)
        {
          EncodeLambda(lambdaNode);
        },
        [this, &stateMap](rvsdg::ThetaNode & thetaNode)
        {
          EncodeTheta(thetaNode, stateMap);
        },
        [this, &stateMap](rvsdg::GammaNode & gammaNode)
        {
          EncodeGamma(gammaNode, stateMap);
        },
        [this, &stateMap](const rvsdg::SimpleNode & simpleNode)
        {
          MatchTypeWithDefault(
              simpleNode.GetOperation(),
              [this, &simpleNode, &stateMap](const AllocaOperation &)
              {
                EncodeAlloca(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MallocOperation &)
              {
                EncodeMalloc(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const LoadOperation &)
              {
                EncodeLoad(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const StoreOperation &)
              {
                EncodeStore(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const CallOperation &)
              {
                EncodeCall(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const FreeOperation &)
              {
                EncodeFree(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MemCpyOperation &)
              {
                EncodeMemcpy(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MemSetOperation &)
              {
                EncodeMemset(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MemMoveOperation &)
              {
                EncodeMemmove(simpleNode, stateMap);
              },
              [](const MemoryStateOperation &)
              {
                // Nothing needs to be done
              },
              [&simpleNode]()
              {
                // Ensure we took care of all memory state consuming/producing nodes
                JLM_ASSERT(!hasMemoryState(simpleNode));
              });
        });
  }
}

std::unique_ptr<MemoryStateEncoder::MemoryStateTypeCounters>
MemoryStateEncoder::gatherStatistics(const rvsdg::Region & region) const
{
  std::function<void(
      const rvsdg::Region & region,
      const ModRefSummary & modRefSummary,
      MemoryStateTypeCounters & counters)>
      gather = [&](const rvsdg::Region & region,
                   const ModRefSummary & modRefSummary,
                   MemoryStateTypeCounters & counters)
  {
    for (const auto & node : region.Nodes())
    {
      MatchTypeOrFail(
          node,
          [&](const rvsdg::PhiNode & phiNode)
          {
            gather(*phiNode.subregion(), modRefSummary, counters);
          },
          [](const rvsdg::DeltaNode &)
          {
            // Nothing needs to be done
          },
          [&](const rvsdg::LambdaNode & lambdaNode)
          {
            const auto & modRefSet = modRefSummary.GetLambdaEntryModRef(lambdaNode);
            counters.interProceduralRegionCounter.CountEntity(
                modRefSummary.GetPointsToGraph(),
                modRefSet);
            gather(*lambdaNode.subregion(), modRefSummary, counters);
          },
          [&](const rvsdg::ThetaNode & thetaNode)
          {
            const auto & modRefSet = modRefSummary.GetThetaModRef(thetaNode);
            counters.interProceduralRegionCounter.CountEntity(
                modRefSummary.GetPointsToGraph(),
                modRefSet);
            gather(*thetaNode.subregion(), modRefSummary, counters);
          },
          [&](const rvsdg::GammaNode & gammaNode)
          {
            auto & modRefSet = modRefSummary.GetGammaEntryModRef(gammaNode);
            for (auto & subregion : gammaNode.Subregions())
            {
              counters.interProceduralRegionCounter.CountEntity(
                  modRefSummary.GetPointsToGraph(),
                  modRefSet);
              gather(subregion, modRefSummary, counters);
            }
          },
          [&modRefSummary, &counters](const rvsdg::SimpleNode & simpleNode)
          {
            MatchTypeWithDefault(
                simpleNode.GetOperation(),
                [](const AllocaOperation &)
                {
                  // Nothing needs to be done
                },
                [](const MallocOperation &)
                {
                  // Nothing needs to be done
                },
                [&modRefSummary, &counters, &simpleNode](const LoadOperation &)
                {
                  const auto & modRefSet = modRefSummary.GetSimpleNodeModRef(simpleNode);
                  counters.loadCounter.CountEntity(modRefSummary.GetPointsToGraph(), modRefSet);
                },
                [&modRefSummary, &counters, &simpleNode](const StoreOperation &)
                {
                  const auto & modRefSet = modRefSummary.GetSimpleNodeModRef(simpleNode);
                  counters.storeCounter.CountEntity(modRefSummary.GetPointsToGraph(), modRefSet);
                },
                [&modRefSummary, &counters, &simpleNode](const CallOperation &)
                {
                  const auto & memoryNodes = modRefSummary.GetSimpleNodeModRef(simpleNode);
                  counters.callEntryMergeCounter.CountEntity(
                      modRefSummary.GetPointsToGraph(),
                      memoryNodes);
                },
                [](const FreeOperation &)
                {
                  // Nothing needs to be done
                },
                [](const MemCpyOperation &)
                {
                  // Nothing needs to be done
                },
                [](const MemSetOperation &)
                {
                  // Nothing needs to be done
                },
                [](const MemMoveOperation &)
                {
                  // Nothing needs to be done
                },
                [](const MemoryStateOperation &)
                {
                  // Nothing needs to be done
                },
                [&simpleNode]()
                {
                  // Ensure we took care of all memory state consuming/producing nodes
                  JLM_ASSERT(!hasMemoryState(simpleNode));
                });
          });
    }
  };

  auto counters = std::make_unique<MemoryStateTypeCounters>();
  gather(region, *this->modRefSummary_, *counters);
  return counters;
}

void
MemoryStateEncoder::EncodeAlloca(const rvsdg::SimpleNode & allocaNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<AllocaOperation>(allocaNode.GetOperation()));

  auto & allocaMemoryNodes = modRefSummary_->GetSimpleNodeModRef(allocaNode).getModRefNodes();
  // It is possible for read-only allocas to not have any associated memory nodes
  if (allocaMemoryNodes.size() == 0)
    return;
  // An alloca can have at most one associated memory node
  JLM_ASSERT(allocaMemoryNodes.size() == 1);
  auto allocaMemoryNode = allocaMemoryNodes.begin()->first;
  auto & allocaNodeStateOutput = *allocaNode.output(1);

  // If a state representing the alloca already exists in the region,
  // merge it with the state created by the alloca using a MemoryStateJoin node.
  if (const auto statePair = stateMap.TryGetState(allocaMemoryNode))
  {
    auto & joinNode =
        MemoryStateJoinOperation::CreateNode({ &allocaNodeStateOutput, &statePair->State() });
    auto & joinOutput = *joinNode.output(0);
    statePair->ReplaceState(joinOutput);
  }
  else
  {
    stateMap.InsertState(allocaMemoryNode, allocaNodeStateOutput);
  }
}

void
MemoryStateEncoder::EncodeMalloc(const rvsdg::SimpleNode & mallocNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<MallocOperation>(mallocNode.GetOperation()));

  auto & mallocMemoryNodes = modRefSummary_->GetSimpleNodeModRef(mallocNode).getModRefNodes();
  // It is possible for read-only mallocs to not have any associated memory nodes
  if (mallocMemoryNodes.size() == 0)
    return;
  // A malloc can have at most one associated memory node
  JLM_ASSERT(mallocMemoryNodes.size() == 1);
  auto mallocMemoryNode = mallocMemoryNodes.begin()->first;

  auto & mallocNodeStateOutput = MallocOperation::memoryStateOutput(mallocNode);

  // We use a static heap model. This means that multiple invocations of an malloc
  // at runtime can refer to the same abstract memory location. We therefore need to
  // merge the previous and the current state to ensure that the previous state
  // is not just simply replaced and therefore "lost".
  if (const auto statePair = stateMap.TryGetState(mallocMemoryNode))
  {
    auto & joinNode =
        MemoryStateJoinOperation::CreateNode({ &mallocNodeStateOutput, &statePair->State() });
    auto & joinOutput = *joinNode.output(0);
    statePair->ReplaceState(joinOutput);
  }
  else
  {
    stateMap.InsertState(mallocMemoryNode, mallocNodeStateOutput);
  }
}

void
MemoryStateEncoder::EncodeLoad(const rvsdg::SimpleNode & node, StateMap & stateMap) const
{
  JLM_ASSERT(is<LoadOperation>(node.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(node);
  const auto memoryNodeStatePairs = stateMap.GetExistingStates(modRefSet);
  const auto memoryStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  const auto & newLoadNode = ReplaceLoadNode(node, memoryStates);

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      LoadOperation::MemoryStateOutputs(newLoadNode));
}

void
MemoryStateEncoder::EncodeStore(const rvsdg::SimpleNode & node, StateMap & stateMap) const
{
  JLM_ASSERT(is<StoreOperation>(node.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(node);
  const auto memoryNodeStatePairs = stateMap.GetExistingStates(modRefSet);
  const auto memoryStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  const auto & newStoreNode = ReplaceStoreNode(node, memoryStates);

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      StoreOperation::MemoryStateOutputs(newStoreNode));
}

void
MemoryStateEncoder::EncodeFree(const rvsdg::SimpleNode & freeNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<FreeOperation>(freeNode.GetOperation()));

  auto address = freeNode.input(0)->origin();
  auto ioState = freeNode.input(freeNode.ninputs() - 1)->origin();
  auto memoryNodeStatePairs =
      stateMap.GetExistingStates(modRefSummary_->GetSimpleNodeModRef(freeNode));
  auto inStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto outputs = FreeOperation::Create(address, inStates, ioState);

  // Redirect IO state edge
  freeNode.output(freeNode.noutputs() - 1)->divert_users(outputs.back());

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      { outputs.begin(), std::prev(outputs.end()) });
}

void
MemoryStateEncoder::EncodeCall(const rvsdg::SimpleNode & callNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<CallOperation>(callNode.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(callNode);
  const auto statePairs = stateMap.GetExistingStates(modRefSet);

  std::vector<rvsdg::Output *> inputStates;
  std::vector<MemoryNodeId> memoryNodeIds;
  for (auto statePair : statePairs)
  {
    inputStates.emplace_back(&statePair->State());
    memoryNodeIds.push_back(statePair->MemoryNode());
  }

  auto & entryMergeNode = CallEntryMemoryStateMergeOperation::CreateNode(
      *callNode.region(),
      inputStates,
      memoryNodeIds);
  CallOperation::GetMemoryStateInput(callNode).divert_to(entryMergeNode.output(0));

  auto & exitSplitNode = CallExitMemoryStateSplitOperation::CreateNode(
      CallOperation::GetMemoryStateOutput(callNode),
      memoryNodeIds);

  StateMap::MemoryNodeStatePair::ReplaceStates(statePairs, rvsdg::outputs(&exitSplitNode));
}

void
MemoryStateEncoder::EncodeMemcpy(const rvsdg::SimpleNode & memcpyNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<MemCpyOperation>(memcpyNode.GetOperation()));

  auto memoryNodeStatePairs =
      stateMap.GetExistingStates(modRefSummary_->GetSimpleNodeModRef(memcpyNode));
  auto memoryStateOperands = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto memoryStateResults = ReplaceMemcpyNode(memcpyNode, memoryStateOperands);

  StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, memoryStateResults);
}

void
MemoryStateEncoder::EncodeMemset(const rvsdg::SimpleNode & memsetNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<MemSetOperation>(memsetNode.GetOperation()));

  auto memoryNodeStatePairs =
      stateMap.GetExistingStates(modRefSummary_->GetSimpleNodeModRef(memsetNode));
  auto memoryStateOperands = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto memoryStateResults = ReplaceMemsetNode(memsetNode, memoryStateOperands);

  StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, memoryStateResults);
}

void
MemoryStateEncoder::EncodeMemmove(const rvsdg::SimpleNode & memmoveNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<MemMoveOperation>(memmoveNode.GetOperation()));

  auto memoryNodeStatePairs =
      stateMap.GetExistingStates(modRefSummary_->GetSimpleNodeModRef(memmoveNode));
  auto memoryStateOperands = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto memoryStateResults = ReplaceMemmoveNode(memmoveNode, memoryStateOperands);

  StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, memoryStateResults);
}

void
MemoryStateEncoder::EncodeLambda(const rvsdg::LambdaNode & lambdaNode) const
{
  StateMap subregionStateMap;

  // Handle lambda entry
  {
    auto & memoryStateArgument = GetMemoryStateRegionArgument(lambdaNode);

    const auto & modRefSet = modRefSummary_->GetLambdaEntryModRef(lambdaNode);
    const auto memoryNodeIds = GetMemoryNodeIds(modRefSet);
    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(memoryStateArgument, memoryNodeIds);
    const auto states = rvsdg::outputs(&lambdaEntrySplitNode);

    size_t n = 0;
    for (const auto [memoryNode, _] : modRefSet.getModRefNodes())
      subregionStateMap.InsertState(memoryNode, *states[n++]);

    if (!states.empty())
    {
      // This additional MemoryStateMergeOperation node makes all other nodes in the function that
      // consume the memory state dependent on this node and therefore transitively on the
      // LambdaEntryMemoryStateSplitOperation. This ensures that the
      // LambdaEntryMemoryStateSplitOperation is always visited before all other memory state
      // consuming nodes:
      //
      // ... := LAMBDA[f]
      //   [..., a1, ...]
      //     o1, ..., ox := LambdaEntryMemoryStateSplit a1
      //     oy = MemoryStateMerge o1, ..., ox
      //     ....
      //
      // No other memory state consuming node aside from the LambdaEntryMemoryStateSplitOperation
      // should now consume a1.
      auto state = MemoryStateMergeOperation::Create(states);
      memoryStateArgument.divertUsersWhere(
          *state,
          [&lambdaEntrySplitNode](const rvsdg::Input & user)
          {
            return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user) != &lambdaEntrySplitNode;
          });
    }
  }

  EncodeRegion(*lambdaNode.subregion(), subregionStateMap);

  // Handle lambda exit
  {
    const auto & modRefSet = modRefSummary_->GetLambdaExitModRef(lambdaNode);
    auto & memoryStateResult = GetMemoryStateRegionResult(lambdaNode);

    std::vector<rvsdg::Output *> states;
    std::vector<MemoryNodeId> memoryNodeIds;
    auto & subregion = *lambdaNode.subregion();
    const auto memoryNodeStatePairs = subregionStateMap.GetStates(modRefSet);
    for (const auto memoryNodeStatePair : memoryNodeStatePairs)
    {
      states.push_back(&memoryNodeStatePair->State());
      memoryNodeIds.push_back(memoryNodeStatePair->MemoryNode());
    }

    const auto mergedState =
        LambdaExitMemoryStateMergeOperation::CreateNode(subregion, states, memoryNodeIds).output(0);
    memoryStateResult.divert_to(mergedState);
  }
}

void
MemoryStateEncoder::EncodeGamma(rvsdg::GammaNode & gammaNode, StateMap & stateMap) const
{
  std::vector<StateMap> subregionStateMap(gammaNode.nsubregions());

  // Handle gamma entry
  {
    auto & modRefSet = modRefSummary_->GetGammaEntryModRef(gammaNode);
    auto memoryNodeStatePairs = stateMap.GetExistingStates(modRefSet);
    for (auto & memoryNodeStatePair : memoryNodeStatePairs)
    {
      auto gammaInput = gammaNode.AddEntryVar(&memoryNodeStatePair->State());
      for (auto & argument : gammaInput.branchArgument)
        subregionStateMap[argument->region()->index()].InsertState(
            memoryNodeStatePair->MemoryNode(),
            *argument);
    }
  }

  for (auto & subregion : gammaNode.Subregions())
    EncodeRegion(subregion, subregionStateMap[subregion.index()]);

  // Handle gamma exit
  {
    auto & modRefSet = modRefSummary_->GetGammaExitModRef(gammaNode);
    auto memoryNodeStatePairs = stateMap.GetExistingStates(modRefSet);

    for (auto & memoryNodeStatePair : memoryNodeStatePairs)
    {
      std::vector<rvsdg::Output *> states;

      for (auto & subregion : gammaNode.Subregions())
      {
        auto & state = subregionStateMap[subregion.index()]
                           .GetState(memoryNodeStatePair->MemoryNode())
                           ->State();
        states.push_back(&state);
      }

      auto state = gammaNode.AddExitVar(states).output;
      memoryNodeStatePair->ReplaceState(*state);
    }
  }
}

void
MemoryStateEncoder::EncodeTheta(rvsdg::ThetaNode & thetaNode, StateMap & stateMap) const
{
  StateMap subregionStateMap;

  // Handle theta entry
  std::vector<rvsdg::Output *> thetaStateOutputs;
  {
    const auto & modRefSet = modRefSummary_->GetThetaModRef(thetaNode);
    auto memoryNodeStatePairs = stateMap.GetExistingStates(modRefSet);
    for (auto & memoryNodeStatePair : memoryNodeStatePairs)
    {
      auto loopvar = thetaNode.AddLoopVar(&memoryNodeStatePair->State());
      subregionStateMap.InsertState(memoryNodeStatePair->MemoryNode(), *loopvar.pre);
      thetaStateOutputs.push_back(loopvar.output);
    }
  }

  EncodeRegion(*thetaNode.subregion(), subregionStateMap);

  // Handle theta exit
  {
    const auto & memoryNodes = modRefSummary_->GetThetaModRef(thetaNode);
    auto memoryNodeStatePairs = stateMap.GetExistingStates(memoryNodes);

    JLM_ASSERT(memoryNodeStatePairs.size() == thetaStateOutputs.size());
    for (size_t n = 0; n < thetaStateOutputs.size(); n++)
    {
      auto thetaStateOutput = thetaStateOutputs[n];
      auto & memoryNodeStatePair = memoryNodeStatePairs[n];
      auto memoryNode = memoryNodeStatePair->MemoryNode();
      auto loopvar = thetaNode.MapOutputLoopVar(*thetaStateOutput);
      JLM_ASSERT(loopvar.input->origin() == &memoryNodeStatePair->State());

      auto & subregionState = subregionStateMap.GetState(memoryNode)->State();
      loopvar.post->divert_to(&subregionState);
      memoryNodeStatePair->ReplaceState(*thetaStateOutput);
    }
  }
}

rvsdg::SimpleNode &
MemoryStateEncoder::ReplaceLoadNode(
    const rvsdg::SimpleNode & node,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<LoadOperation>(node.GetOperation()));

  if (const auto loadVolatileOperation =
          dynamic_cast<const LoadVolatileOperation *>(&node.GetOperation()))
  {
    auto & newLoadNode = LoadVolatileOperation::CreateNode(
        *LoadOperation::AddressInput(node).origin(),
        *LoadVolatileOperation::IOStateInput(node).origin(),
        memoryStates,
        loadVolatileOperation->GetLoadedType(),
        loadVolatileOperation->GetAlignment());
    auto & oldLoadedValueOutput = LoadOperation::LoadedValueOutput(node);
    auto & newLoadedValueOutput = LoadOperation::LoadedValueOutput(newLoadNode);
    auto & oldIOStateOutput = LoadVolatileOperation::IOStateOutput(node);
    auto & newIOStateOutput = LoadVolatileOperation::IOStateOutput(newLoadNode);
    oldLoadedValueOutput.divert_users(&newLoadedValueOutput);
    oldIOStateOutput.divert_users(&newIOStateOutput);
    return newLoadNode;
  }

  if (const auto loadNonVolatileOperation =
          dynamic_cast<const LoadNonVolatileOperation *>(&node.GetOperation()))
  {
    auto & newLoadNode = LoadNonVolatileOperation::CreateNode(
        *LoadOperation::AddressInput(node).origin(),
        memoryStates,
        loadNonVolatileOperation->GetLoadedType(),
        loadNonVolatileOperation->GetAlignment());
    auto & oldLoadedValueOutput = LoadOperation::LoadedValueOutput(node);
    auto & newLoadedValueOutput = LoadNonVolatileOperation::LoadedValueOutput(newLoadNode);
    oldLoadedValueOutput.divert_users(&newLoadedValueOutput);
    return newLoadNode;
  }

  JLM_UNREACHABLE("Unhandled load node type.");
}

rvsdg::SimpleNode &
MemoryStateEncoder::ReplaceStoreNode(
    const rvsdg::SimpleNode & node,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<StoreOperation>(node.GetOperation()));

  if (const auto oldStoreVolatileOperation =
          dynamic_cast<const StoreVolatileOperation *>(&node.GetOperation()))
  {
    auto & newStoreNode = StoreVolatileOperation::CreateNode(
        *StoreOperation::AddressInput(node).origin(),
        *StoreOperation::StoredValueInput(node).origin(),
        *StoreVolatileOperation::IOStateInput(node).origin(),
        memoryStates,
        oldStoreVolatileOperation->GetAlignment());
    auto & oldIOStateOutput = StoreVolatileOperation::IOStateOutput(node);
    auto & newIOStateOutput = StoreVolatileOperation::IOStateOutput(newStoreNode);
    oldIOStateOutput.divert_users(&newIOStateOutput);
    return newStoreNode;
  }

  if (const auto oldStoreNonVolatileOperation =
          dynamic_cast<const StoreNonVolatileOperation *>(&node.GetOperation()))
  {
    return StoreNonVolatileOperation::CreateNode(
        *StoreOperation::AddressInput(node).origin(),
        *StoreOperation::StoredValueInput(node).origin(),
        memoryStates,
        oldStoreNonVolatileOperation->GetAlignment());
  }

  JLM_UNREACHABLE("Unhandled store node type.");
}

std::vector<rvsdg::Output *>
MemoryStateEncoder::ReplaceMemcpyNode(
    const rvsdg::SimpleNode & memcpyNode,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<MemCpyOperation>(memcpyNode.GetOperation()));

  auto destination = memcpyNode.input(0)->origin();
  auto source = memcpyNode.input(1)->origin();
  auto length = memcpyNode.input(2)->origin();

  if (is<MemCpyVolatileOperation>(memcpyNode.GetOperation()))
  {
    auto & ioState = *memcpyNode.input(3)->origin();
    auto & newMemcpyNode =
        MemCpyVolatileOperation::CreateNode(*destination, *source, *length, ioState, memoryStates);
    auto results = rvsdg::outputs(&newMemcpyNode);

    // Redirect I/O state
    memcpyNode.output(0)->divert_users(results[0]);

    // Skip I/O state and only return memory states
    return { std::next(results.begin()), results.end() };
  }
  if (is<MemCpyNonVolatileOperation>(memcpyNode.GetOperation()))
  {
    return MemCpyNonVolatileOperation::create(destination, source, length, memoryStates);
  }

  throw std::logic_error("Unhandled memcpy operation type.");
}

std::vector<rvsdg::Output *>
MemoryStateEncoder::ReplaceMemsetNode(
    const rvsdg::SimpleNode & memsetNode,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<MemSetOperation>(memsetNode.GetOperation()));

  auto destination = MemSetOperation::destinationInput(memsetNode).origin();
  auto value = MemSetOperation::valueInput(memsetNode).origin();
  auto length = MemSetOperation::lengthInput(memsetNode).origin();

  if (is<MemSetNonVolatileOperation>(memsetNode.GetOperation()))
  {
    return outputs(
        &MemSetNonVolatileOperation::createNode(*destination, *value, *length, memoryStates));
  }

  throw std::logic_error("Unhandled memset operation type.");
}

std::vector<rvsdg::Output *>
MemoryStateEncoder::ReplaceMemmoveNode(
    const rvsdg::SimpleNode & memmoveNode,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<MemMoveOperation>(memmoveNode.GetOperation()));

  auto & destOperand = *memmoveNode.input(0)->origin();
  auto & srcOperand = *memmoveNode.input(1)->origin();
  auto & lengthOperand = *memmoveNode.input(2)->origin();

  if (is<MemMoveNonVolatileOperation>(memmoveNode.GetOperation()))
  {
    return outputs(&MemMoveNonVolatileOperation::createNode(
        destOperand,
        srcOperand,
        lengthOperand,
        memoryStates));
  }

  throw std::logic_error("Unhandled memmove operation type.");
}

}
