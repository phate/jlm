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

}

struct MemoryStateEncoder::MemoryStateTypeCounters final
{
  MemoryStateTypeCounter interProceduralRegionCounter;
  MemoryStateTypeCounter loadCounter;
  MemoryStateTypeCounter storeCounter;
  MemoryStateTypeCounter callEntryMergeCounter;

  size_t numAllocaNodes = 0;
  size_t numMallocNodes = 0;
  size_t numLoadNodes = 0;
  size_t numStoreNodes = 0;
  size_t numCallNodes = 0;
  size_t numFreeNodes = 0;
  size_t numMemCpyNodes = 0;
  size_t numMemSetNodes = 0;
  size_t numMemMoveNodes = 0;
  size_t numGammaNodes = 0;
  size_t numThetaNodes = 0;
  size_t numLambdaNodes = 0;
};

/** \brief Statistics class for memory state encoder encoding
 *
 */
class MemoryStateEncoder::Statistics final : public util::Statistics
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

  static constexpr auto EncodingTimerLabel_ = "EncodingTime";

  static constexpr auto NumReplacedLoadsLabel_ = "#ReplacedLoads";
  static constexpr auto NumRedirectedLoadsLabel_ = "#RedirectedLoads";

  static constexpr auto NumReplacedStoresLabel_ = "#ReplacedStores";
  static constexpr auto NumRedirectedStoresLabel_ = "#RedirectedStores";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::MemoryStateEncoder, sourceFile)
  {}

  void
  StartEncoding(const rvsdg::Graph & graph)
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(EncodingTimerLabel_).start();
  }

  void
  StopEncoding(const EncodingCounter & counter)
  {
    AddMeasurement(NumRedirectedLoadsLabel_, counter.numRedirectedLoads);
    AddMeasurement(NumReplacedLoadsLabel_, counter.numReplacedLoads);
    AddMeasurement(NumRedirectedStoresLabel_, counter.numRedirectedStores);
    AddMeasurement(NumReplacedStoresLabel_, counter.numReplacedStores);
    GetTimer(EncodingTimerLabel_).stop();
  }

  void
  AddCounters(const MemoryStateTypeCounters & counters)
  {
    AddMeasurement(NumIntraProceduralRegions_, counters.interProceduralRegionCounter.NumEntities);
    AddMemoryStateTypeCounter(RegionArgumentStateSuffix_, counters.interProceduralRegionCounter);

    AddMemoryStateTypeCounter(LoadStateSuffix_, counters.loadCounter);

    AddMemoryStateTypeCounter(StoreStateSuffix_, counters.storeCounter);

    AddMeasurement(NumCallEntryMergeOperations_, counters.callEntryMergeCounter.NumEntities);
    AddMemoryStateTypeCounter(CallEntryMergeStateSuffix_, counters.callEntryMergeCounter);

    AddMeasurement("#AllocaNodes", counters.numAllocaNodes);
    AddMeasurement("#MallocNodes", counters.numMallocNodes);
    AddMeasurement("#LoadNodes", counters.numLoadNodes);
    AddMeasurement("#StoreNodes", counters.numStoreNodes);
    AddMeasurement("#CallNodes", counters.numCallNodes);
    AddMeasurement("#FreeNodes", counters.numFreeNodes);
    AddMeasurement("#MemCpyNodes", counters.numMemCpyNodes);
    AddMeasurement("#MemSetNodes", counters.numMemSetNodes);
    AddMeasurement("#MemMoveNodes", counters.numMemMoveNodes);
    AddMeasurement("#GammaNodes", counters.numGammaNodes);
    AddMeasurement("#ThetaNodes", counters.numThetaNodes);
    AddMeasurement("#LambdaNodes", counters.numLambdaNodes);
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
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

/** \brief Hash map for mapping points-to graph memory nodes to RVSDG memory states.
 */
class MemoryStateEncoder::StateMap final
{
public:
  StateMap() = default;

  StateMap(const StateMap &) = delete;

  StateMap(StateMap &&) = delete;

  StateMap &
  operator=(const StateMap &) = delete;

  StateMap &
  operator=(StateMap &&) = delete;

  rvsdg::Output *
  tryGetState(const PointsToGraph::NodeIndex modRefNode) noexcept
  {
    if (const auto it = states_.find(modRefNode); it != states_.end())
      return it->second;

    return nullptr;
  }

  rvsdg::Output &
  getState(const PointsToGraph::NodeIndex modRefNode)
  {
    if (const auto state = tryGetState(modRefNode))
      return *state;
    throw std::logic_error("Memory node does not have a state.");
  }

  /**
   * Gets the memory state for each of the given memory nodes.
   * If no memory state output exists for a given memory node, an UndefValue node is created.
   * This only happens when a \ref ModRefSet contains alloca whose memory state is not routed
   * in as a function argument, and the alloca operation has not been encoded yet.
   * This can happen when alloca operations are inside subregions, or when the allocation count is a
   * runtime value that depends on a load.
   *
   * @param modRefNodes the set of memory nodes to retrieve states for.
   * @param region the region in which the states are needed
   *
   * @return The memory states for each given memory nodes.
   */
  std::vector<rvsdg::Output *>
  getOrCreateStates(const std::vector<MemoryNodeId> & modRefNodes, rvsdg::Region & region)
  {
    std::vector<rvsdg::Output *> memoryStates;
    for (auto & modRefNode : modRefNodes)
    {
      if (const auto state = tryGetState(modRefNode))
      {
        memoryStates.push_back(state);
      }
      else
      {
        // If no memory state output exists for the memory node, create an UndefValue for it

        // Using undef for memory states that do not exist yet should only be done for allocas.
        // TODO: After refactoring, add an assert here like so:
        // JLM_ASSERT(modRefSummary_->getPointsToGraph().getKind(memoryNode) == NodeKind::Alloca);

        auto & undefOutput = *UndefValueOperation::Create(region, MemoryStateType::Create());
        insertState(modRefNode, undefOutput);
        memoryStates.push_back(&undefOutput);
      }
    }

    return memoryStates;
  }

  void
  updateState(const MemoryNodeId modRefNode, rvsdg::Output & memoryState)
  {
    if (!tryGetState(modRefNode))
      throw std::logic_error("Unknown modRefNode in StateMap");

    states_[modRefNode] = &memoryState;
  }

  void
  updateStates(
      const std::vector<MemoryNodeId> & modRefNodes,
      const rvsdg::Node::OutputIteratorRange & memoryStates)
  {
    JLM_ASSERT(
        modRefNodes.size()
        == static_cast<size_t>(std::distance(memoryStates.begin(), memoryStates.end())));

    size_t i = 0;
    for (auto & memoryState : memoryStates)
    {
      auto & modRefNode = modRefNodes[i++];
      updateState(modRefNode, memoryState);
    }
  }

  /**
   * Inserts memory state \p state for memory node \p modRefNode in the state map.
   * The memory node must not have an already associated state.
   *
   * @param modRefNode the memory node
   * @param state the output that produces the memory state associated with the memory node
   */
  void
  insertState(PointsToGraph::NodeIndex modRefNode, rvsdg::Output & state)
  {
    if (auto [_, added] = states_.insert({ modRefNode, &state }); !added)
      throw std::logic_error("Memory node already has a state.");
  }

private:
  std::unordered_map<PointsToGraph::NodeIndex, rvsdg::Output *> states_;
};

static std::vector<MemoryNodeId>
getModRefSetNodes(const ModRefSet & modRefSet)
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
  statistics_ = Statistics::Create(rvsdgModule.SourceFilePath().value());
  auto & rvsdg = rvsdgModule.Rvsdg();

  // The statistics gathering needs to happen before the encoding as the encoding replaces nodes in
  // the RVSDG and these new nodes would not have any ModRefSets associated with them.
  if (statisticsCollector.IsDemanded(util::Statistics::Id::MemoryStateEncoder))
  {
    auto counters = gatherStatistics(rvsdg.GetRootRegion());
    statistics_->AddCounters(*counters);
  }

  statistics_->StartEncoding(rvsdg);
  encodeInterProcedural(rvsdg.GetRootRegion());
  statistics_->StopEncoding(encodingCounter_);
  encodingCounter_ = EncodingCounter();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics_));
}

void
MemoryStateEncoder::encodeInterProcedural(rvsdg::Region & region)
{
  for (const auto node : rvsdg::TopDownTraverser(&region))
  {
    MatchTypeOrFail(
        *node,
        [this](rvsdg::PhiNode & phiNode)
        {
          encodeInterProcedural(*phiNode.subregion());
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [this](rvsdg::LambdaNode & lambdaNode)
        {
          encodeLambda(lambdaNode);
        },
        [](const rvsdg::SimpleNode &)
        {
          // Nothing needs to be done
        });
  }
}

void
MemoryStateEncoder::encodeIntraProcedural(rvsdg::Region & region, StateMap & stateMap)
{
  for (const auto node : rvsdg::TopDownTraverser(&region))
  {
    MatchTypeOrFail(
        *node,
        [this, &stateMap](rvsdg::ThetaNode & thetaNode)
        {
          encodeTheta(thetaNode, stateMap);
        },
        [this, &stateMap](rvsdg::GammaNode & gammaNode)
        {
          encodeGamma(gammaNode, stateMap);
        },
        [this, &stateMap](const rvsdg::SimpleNode & simpleNode)
        {
          MatchTypeWithDefault(
              simpleNode.GetOperation(),
              [this, &simpleNode, &stateMap](const AllocaOperation &)
              {
                encodeAlloca(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MallocOperation &)
              {
                encodeMalloc(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const LoadOperation &)
              {
                encodeLoad(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const StoreOperation &)
              {
                encodeStore(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const CallOperation &)
              {
                encodeCall(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const FreeOperation &)
              {
                encodeFree(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MemCpyOperation &)
              {
                encodeMemcpy(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MemSetOperation &)
              {
                encodeMemset(simpleNode, stateMap);
              },
              [this, &simpleNode, &stateMap](const MemMoveOperation &)
              {
                encodeMemmove(simpleNode, stateMap);
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
            counters.numLambdaNodes++;
            gather(*lambdaNode.subregion(), modRefSummary, counters);
          },
          [&](const rvsdg::ThetaNode & thetaNode)
          {
            const auto & modRefSet = modRefSummary.GetThetaModRef(thetaNode);
            counters.interProceduralRegionCounter.CountEntity(
                modRefSummary.GetPointsToGraph(),
                modRefSet);
            counters.numThetaNodes++;
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
              counters.numGammaNodes++;
              gather(subregion, modRefSummary, counters);
            }
          },
          [&modRefSummary, &counters](const rvsdg::SimpleNode & simpleNode)
          {
            MatchTypeWithDefault(
                simpleNode.GetOperation(),
                [&counters](const AllocaOperation &)
                {
                  counters.numAllocaNodes++;
                },
                [&counters](const MallocOperation &)
                {
                  counters.numMallocNodes++;
                },
                [&modRefSummary, &counters, &simpleNode](const LoadOperation &)
                {
                  const auto & modRefSet = modRefSummary.GetSimpleNodeModRef(simpleNode);
                  counters.loadCounter.CountEntity(modRefSummary.GetPointsToGraph(), modRefSet);
                  counters.numLoadNodes++;
                },
                [&modRefSummary, &counters, &simpleNode](const StoreOperation &)
                {
                  const auto & modRefSet = modRefSummary.GetSimpleNodeModRef(simpleNode);
                  counters.storeCounter.CountEntity(modRefSummary.GetPointsToGraph(), modRefSet);
                  counters.numStoreNodes++;
                },
                [&modRefSummary, &counters, &simpleNode](const CallOperation &)
                {
                  const auto & memoryNodes = modRefSummary.GetSimpleNodeModRef(simpleNode);
                  counters.callEntryMergeCounter.CountEntity(
                      modRefSummary.GetPointsToGraph(),
                      memoryNodes);
                  counters.numCallNodes++;
                },
                [&counters](const FreeOperation &)
                {
                  counters.numFreeNodes++;
                },
                [&counters](const MemCpyOperation &)
                {
                  counters.numMemCpyNodes++;
                },
                [&counters](const MemSetOperation &)
                {
                  counters.numMemSetNodes++;
                },
                [&counters](const MemMoveOperation &)
                {
                  counters.numMemMoveNodes++;
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
MemoryStateEncoder::encodeAlloca(const rvsdg::SimpleNode & allocaNode, StateMap & stateMap) const
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
  if (const auto memoryState = stateMap.tryGetState(allocaMemoryNode))
  {
    auto & joinNode = MemoryStateJoinOperation::CreateNode({ &allocaNodeStateOutput, memoryState });
    auto & joinOutput = *joinNode.output(0);
    stateMap.updateState(allocaMemoryNode, joinOutput);
  }
  else
  {
    stateMap.insertState(allocaMemoryNode, allocaNodeStateOutput);
  }
}

void
MemoryStateEncoder::encodeMalloc(const rvsdg::SimpleNode & mallocNode, StateMap & stateMap) const
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
  if (const auto memoryState = stateMap.tryGetState(mallocMemoryNode))
  {
    auto & joinNode = MemoryStateJoinOperation::CreateNode({ &mallocNodeStateOutput, memoryState });
    auto & joinOutput = *joinNode.output(0);
    stateMap.updateState(mallocMemoryNode, joinOutput);
  }
  else
  {
    stateMap.insertState(mallocMemoryNode, mallocNodeStateOutput);
  }
}

void
MemoryStateEncoder::encodeLoad(const rvsdg::SimpleNode & node, StateMap & stateMap)
{
  JLM_ASSERT(is<LoadOperation>(node.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(node);
  const auto modRefNodes = getModRefSetNodes(modRefSet);
  const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *node.region());

  if (memStateOperands.size() == LoadOperation::numMemoryStates(node))
  {
    encodingCounter_.numRedirectedLoads += 1;
    for (auto & memoryStateOutput : LoadOperation::MemoryStateOutputs(node))
    {
      const auto memoryStateOperand =
          LoadOperation::MapMemoryStateOutputToInput(memoryStateOutput).origin();
      memoryStateOutput.divert_users(memoryStateOperand);
    }

    size_t n = 0;
    for (auto & memoryStateInput : LoadOperation::MemoryStateInputs(node))
      memoryStateInput.divert_to(memStateOperands[n++]);

    stateMap.updateStates(modRefNodes, LoadOperation::MemoryStateOutputs(node));
  }
  else
  {
    encodingCounter_.numReplacedLoads += 1;
    const auto & newLoadNode = replaceLoadNode(node, memStateOperands);
    stateMap.updateStates(modRefNodes, LoadOperation::MemoryStateOutputs(newLoadNode));
  }
}

void
MemoryStateEncoder::encodeStore(const rvsdg::SimpleNode & node, StateMap & stateMap)
{
  JLM_ASSERT(is<StoreOperation>(node.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(node);
  const auto modRefNodes = getModRefSetNodes(modRefSet);
  const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *node.region());

  if (memStateOperands.size() == StoreOperation::numMemoryStates(node))
  {
    encodingCounter_.numRedirectedStores++;
    for (auto & memoryStateOutput : StoreOperation::MemoryStateOutputs(node))
    {
      const auto memoryStateOperand =
          StoreOperation::MapMemoryStateOutputToInput(memoryStateOutput).origin();
      memoryStateOutput.divert_users(memoryStateOperand);
    }

    size_t n = 0;
    for (auto & memoryStateInput : StoreOperation::getMemoryStateInputs(node))
      memoryStateInput.divert_to(memStateOperands[n++]);

    stateMap.updateStates(modRefNodes, StoreOperation::MemoryStateOutputs(node));
  }
  else
  {
    encodingCounter_.numReplacedStores++;
    const auto & newStoreNode = replaceStoreNode(node, memStateOperands);
    stateMap.updateStates(modRefNodes, StoreOperation::MemoryStateOutputs(newStoreNode));
  }
}

void
MemoryStateEncoder::encodeFree(const rvsdg::SimpleNode & freeNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<FreeOperation>(freeNode.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(freeNode);
  const auto modRefNodes = getModRefSetNodes(modRefSet);

  const auto addressOperand = FreeOperation::getAddressInput(freeNode).origin();
  const auto ioStateOperand = FreeOperation::getIOStateInput(freeNode).origin();
  const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *freeNode.region());

  auto & newFreeNode =
      FreeOperation::createNode(*addressOperand, *ioStateOperand, memStateOperands);

  FreeOperation::getIOStateOutput(freeNode).divert_users(
      &FreeOperation::getIOStateOutput(newFreeNode));

  for (auto & oldMemStateOutput : FreeOperation::memoryStateOutputs(freeNode))
  {
    auto oldMemStateOperand =
        FreeOperation::mapMemoryStateOutputToInput(oldMemStateOutput).origin();
    oldMemStateOutput.divert_users(oldMemStateOperand);
  }
  JLM_ASSERT(freeNode.IsDead());

  stateMap.updateStates(modRefNodes, FreeOperation::memoryStateOutputs(newFreeNode));
}

void
MemoryStateEncoder::encodeCall(const rvsdg::SimpleNode & callNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<CallOperation>(callNode.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(callNode);
  const auto modRefNodes = getModRefSetNodes(modRefSet);
  const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *callNode.region());

  auto & entryMergeNode = CallEntryMemoryStateMergeOperation::CreateNode(
      *callNode.region(),
      memStateOperands,
      modRefNodes);
  CallOperation::GetMemoryStateInput(callNode).divert_to(entryMergeNode.output(0));

  auto & exitSplitNode = CallExitMemoryStateSplitOperation::CreateNode(
      CallOperation::GetMemoryStateOutput(callNode),
      modRefNodes);

  stateMap.updateStates(modRefNodes, exitSplitNode.Outputs());
}

void
MemoryStateEncoder::encodeMemcpy(const rvsdg::SimpleNode & memcpyNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<MemCpyOperation>(memcpyNode.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(memcpyNode);
  const auto modRefNodes = getModRefSetNodes(modRefSet);
  const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *memcpyNode.region());

  const auto & newMemCpyNode = replaceMemcpyNode(memcpyNode, memStateOperands);
  stateMap.updateStates(modRefNodes, MemCpyOperation::memoryStateOutputs(newMemCpyNode));
}

void
MemoryStateEncoder::encodeMemset(const rvsdg::SimpleNode & memsetNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<MemSetOperation>(memsetNode.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(memsetNode);
  const auto modRefNodes = getModRefSetNodes(modRefSet);
  const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *memsetNode.region());

  auto & newMemSetNode = replaceMemsetNode(memsetNode, memStateOperands);
  stateMap.updateStates(modRefNodes, MemSetOperation::memoryStateOutputs(newMemSetNode));
}

void
MemoryStateEncoder::encodeMemmove(const rvsdg::SimpleNode & memmoveNode, StateMap & stateMap) const
{
  JLM_ASSERT(is<MemMoveOperation>(memmoveNode.GetOperation()));

  const auto & modRefSet = modRefSummary_->GetSimpleNodeModRef(memmoveNode);
  const auto modRefNodes = getModRefSetNodes(modRefSet);
  const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *memmoveNode.region());

  auto & newMemMoveNode = replaceMemmoveNode(memmoveNode, memStateOperands);
  stateMap.updateStates(modRefNodes, MemMoveOperation::memoryStateOutputs(newMemMoveNode));
}

void
MemoryStateEncoder::encodeLambda(const rvsdg::LambdaNode & lambdaNode)
{
  StateMap subregionStateMap;

  // Handle lambda entry
  {
    const auto & modRefSet = modRefSummary_->GetLambdaEntryModRef(lambdaNode);
    const auto modRefNodes = getModRefSetNodes(modRefSet);
    auto & memoryStateArgument = GetMemoryStateRegionArgument(lambdaNode);

    auto & lambdaEntrySplitNode =
        LambdaEntryMemoryStateSplitOperation::CreateNode(memoryStateArgument, modRefNodes);
    const auto memStates = rvsdg::outputs(&lambdaEntrySplitNode);

    size_t n = 0;
    for (const auto modRefNode : modRefNodes)
      subregionStateMap.insertState(modRefNode, *memStates[n++]);

    if (!memStates.empty())
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
      auto state = MemoryStateMergeOperation::Create(memStates);
      memoryStateArgument.divertUsersWhere(
          *state,
          [&lambdaEntrySplitNode](const rvsdg::Input & user)
          {
            return rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user) != &lambdaEntrySplitNode;
          });
    }
  }

  auto & lambdaSubregion = *lambdaNode.subregion();
  encodeIntraProcedural(lambdaSubregion, subregionStateMap);

  // Handle lambda exit
  {
    const auto & modRefSet = modRefSummary_->GetLambdaExitModRef(lambdaNode);
    const auto modRefNodes = getModRefSetNodes(modRefSet);
    const auto memStateOperands = subregionStateMap.getOrCreateStates(modRefNodes, lambdaSubregion);
    auto & memoryStateResult = GetMemoryStateRegionResult(lambdaNode);

    auto & lambdaExitMergNode = LambdaExitMemoryStateMergeOperation::CreateNode(
        lambdaSubregion,
        memStateOperands,
        modRefNodes);
    memoryStateResult.divert_to(lambdaExitMergNode.output(0));
  }

  lambdaSubregion.prune(false);
}

void
MemoryStateEncoder::encodeGamma(rvsdg::GammaNode & gammaNode, StateMap & stateMap)
{
  std::vector<StateMap> subregionStateMap(gammaNode.nsubregions());

  // Handle gamma entry
  {
    auto & modRefSet = modRefSummary_->GetGammaEntryModRef(gammaNode);
    const auto modRefNodes = getModRefSetNodes(modRefSet);
    const auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *gammaNode.region());

    size_t n = 0;
    for (auto & modRefNode : modRefNodes)
    {
      auto gammaInput = gammaNode.AddEntryVar(memStateOperands[n++]);
      for (auto argument : gammaInput.branchArgument)
        subregionStateMap[argument->region()->index()].insertState(modRefNode, *argument);
    }
  }

  for (auto & subregion : gammaNode.Subregions())
    encodeIntraProcedural(subregion, subregionStateMap[subregion.index()]);

  // Handle gamma exit
  {
    auto & modRefSet = modRefSummary_->GetGammaExitModRef(gammaNode);
    const auto modRefNodes = getModRefSetNodes(modRefSet);

    for (auto modRefNode : modRefNodes)
    {
      std::vector<rvsdg::Output *> memStateOperands;
      for (auto & subregion : gammaNode.Subregions())
      {
        auto & state = subregionStateMap[subregion.index()].getState(modRefNode);
        memStateOperands.push_back(&state);
      }

      auto state = gammaNode.AddExitVar(memStateOperands).output;
      stateMap.updateState(modRefNode, *state);
    }
  }

  for (auto & subregion : gammaNode.Subregions())
    subregion.prune(false);
}

void
MemoryStateEncoder::encodeTheta(rvsdg::ThetaNode & thetaNode, StateMap & stateMap)
{
  StateMap subregionStateMap;
  const auto & modRefSet = modRefSummary_->GetThetaModRef(thetaNode);
  const auto modRefNodes = getModRefSetNodes(modRefSet);
  auto memStateOperands = stateMap.getOrCreateStates(modRefNodes, *thetaNode.region());

  // Handle theta entry
  std::vector<rvsdg::ThetaNode::LoopVar> loopVars;
  {
    size_t n = 0;
    for (auto & modRefNode : modRefNodes)
    {
      auto loopVar = thetaNode.AddLoopVar(memStateOperands[n++]);
      subregionStateMap.insertState(modRefNode, *loopVar.pre);
      loopVars.push_back(loopVar);
    }
  }

  encodeIntraProcedural(*thetaNode.subregion(), subregionStateMap);

  // Handle theta exit
  {
    JLM_ASSERT(modRefNodes.size() == loopVars.size());
    JLM_ASSERT(memStateOperands.size() == loopVars.size());
    for (size_t n = 0; n < loopVars.size(); n++)
    {
      const auto loopVar = loopVars[n];
      const auto modRefNode = modRefNodes[n];

      auto & subregionState = subregionStateMap.getState(modRefNode);
      loopVar.post->divert_to(&subregionState);
      stateMap.updateState(modRefNode, *loopVar.output);
    }
  }

  thetaNode.subregion()->prune(false);
}

rvsdg::SimpleNode &
MemoryStateEncoder::replaceLoadNode(
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
    for (auto & oldMemStateOutput : LoadOperation::MemoryStateOutputs(node))
    {
      const auto oldMemStateOperand =
          LoadOperation::MapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }

    JLM_ASSERT(node.IsDead());
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
    for (auto & oldMemStateOutput : LoadOperation::MemoryStateOutputs(node))
    {
      const auto oldMemStateOperand =
          LoadOperation::MapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }

    JLM_ASSERT(node.IsDead());
    return newLoadNode;
  }

  JLM_UNREACHABLE("Unhandled load node type.");
}

rvsdg::SimpleNode &
MemoryStateEncoder::replaceStoreNode(
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
    for (auto & oldMemStateOutput : StoreOperation::MemoryStateOutputs(node))
    {
      const auto oldMemStateOperand =
          StoreOperation::MapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }

    JLM_ASSERT(node.IsDead());
    return newStoreNode;
  }

  if (const auto oldStoreNonVolatileOperation =
          dynamic_cast<const StoreNonVolatileOperation *>(&node.GetOperation()))
  {
    for (auto & oldMemStateOutput : StoreOperation::MemoryStateOutputs(node))
    {
      const auto oldMemStateOperand =
          StoreOperation::MapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }

    JLM_ASSERT(node.IsDead());
    return StoreNonVolatileOperation::CreateNode(
        *StoreOperation::AddressInput(node).origin(),
        *StoreOperation::StoredValueInput(node).origin(),
        memoryStates,
        oldStoreNonVolatileOperation->GetAlignment());
  }

  JLM_UNREACHABLE("Unhandled store node type.");
}

rvsdg::SimpleNode &
MemoryStateEncoder::replaceMemcpyNode(
    const rvsdg::SimpleNode & memCpyNode,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<MemCpyOperation>(memCpyNode.GetOperation()));

  auto & destination = *MemCpyOperation::destinationInput(memCpyNode).origin();
  auto & source = *MemCpyOperation::sourceInput(memCpyNode).origin();
  auto & length = *MemCpyOperation::countInput(memCpyNode).origin();

  if (is<MemCpyVolatileOperation>(memCpyNode.GetOperation()))
  {
    auto & ioStateOperand = *MemCpyVolatileOperation::getIOStateInput(memCpyNode).origin();
    auto & newMemCpyNode = MemCpyVolatileOperation::CreateNode(
        destination,
        source,
        length,
        ioStateOperand,
        memoryStates);

    MemCpyVolatileOperation::getIOStateOutput(memCpyNode)
        .divert_users(&MemCpyVolatileOperation::getIOStateOutput(newMemCpyNode));
    for (auto & oldMemStateOutput : MemCpyOperation::memoryStateOutputs(memCpyNode))
    {
      auto oldMemStateOperand =
          MemCpyOperation::mapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }
    JLM_ASSERT(memCpyNode.IsDead());

    return newMemCpyNode;
  }

  if (is<MemCpyNonVolatileOperation>(memCpyNode.GetOperation()))
  {
    for (auto & oldMemStateOutput : MemCpyOperation::memoryStateOutputs(memCpyNode))
    {
      auto oldMemStateOperand =
          MemCpyOperation::mapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }
    JLM_ASSERT(memCpyNode.IsDead());

    return MemCpyNonVolatileOperation::createNode(destination, source, length, memoryStates);
  }

  throw std::logic_error("Unhandled memcpy operation type.");
}

rvsdg::SimpleNode &
MemoryStateEncoder::replaceMemsetNode(
    const rvsdg::SimpleNode & memsetNode,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<MemSetOperation>(memsetNode.GetOperation()));

  auto destination = MemSetOperation::destinationInput(memsetNode).origin();
  auto value = MemSetOperation::valueInput(memsetNode).origin();
  auto length = MemSetOperation::lengthInput(memsetNode).origin();

  if (is<MemSetNonVolatileOperation>(memsetNode.GetOperation()))
  {
    for (auto & oldMemStateOutput : MemSetOperation::memoryStateOutputs(memsetNode))
    {
      auto oldMemStateOperand =
          MemSetOperation::mapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }
    JLM_ASSERT(memsetNode.IsDead());

    return MemSetNonVolatileOperation::createNode(*destination, *value, *length, memoryStates);
  }

  throw std::logic_error("Unhandled memset operation type.");
}

rvsdg::SimpleNode &
MemoryStateEncoder::replaceMemmoveNode(
    const rvsdg::SimpleNode & memmoveNode,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<MemMoveOperation>(memmoveNode.GetOperation()));

  auto & destOperand = *MemMoveOperation::destinationInput(memmoveNode).origin();
  auto & srcOperand = *MemMoveOperation::sourceInput(memmoveNode).origin();
  auto & lengthOperand = *MemMoveOperation::lengthInput(memmoveNode).origin();

  if (is<MemMoveNonVolatileOperation>(memmoveNode.GetOperation()))
  {
    for (auto & oldMemStateOutput : MemMoveOperation::memoryStateOutputs(memmoveNode))
    {
      auto oldMemStateOperand =
          MemMoveOperation::mapMemoryStateOutputToInput(oldMemStateOutput).origin();
      oldMemStateOutput.divert_users(oldMemStateOperand);
    }
    JLM_ASSERT(memmoveNode.IsDead());

    return MemMoveNonVolatileOperation::createNode(
        destOperand,
        srcOperand,
        lengthOperand,
        memoryStates);
  }

  throw std::logic_error("Unhandled memmove operation type.");
}

}
