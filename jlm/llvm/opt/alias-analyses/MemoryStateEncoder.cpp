/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm::aa
{

/**
 * \brief Helper struct for counting up MemoryNodes, among some set of entities that use them
 */
struct MemoryStateTypeCounter final
{
  // The number of entities that have been counted
  uint64_t NumEntities = 0;

  // Count of total memory states, separated by MemoryNode type
  uint64_t NumAllocas = 0;
  uint64_t NumMallocs = 0;
  uint64_t NumDeltas = 0;
  uint64_t NumImports = 0;
  uint64_t NumLambdas = 0;
  uint64_t NumExternal = 0;

  // Among the MemoryNodes counted above, how many were not escaping
  uint64_t NumNonEscaped = 0;

  // Remember the single entity with the highest number of memory states
  uint64_t MaxMemoryStateEntity = 0;
  // Do the same, but only include non-escaped MemoryNodes
  uint64_t MaxNonEscapedMemoryStateEntity = 0;

  void
  CountEntity(
      uint64_t numAllocas,
      uint64_t numMallocs,
      uint64_t numDeltas,
      uint64_t numImports,
      uint64_t numLambdas,
      uint64_t numExternal,
      uint64_t numNonEscaped)
  {
    NumEntities++;

    NumAllocas += numAllocas;
    NumMallocs += numMallocs;
    NumDeltas += numDeltas;
    NumImports += numImports;
    NumLambdas += numLambdas;
    NumExternal += numExternal;

    const uint64_t totalMemoryStates =
        numAllocas + numMallocs + numDeltas + numImports + numLambdas + numExternal;
    if (totalMemoryStates > MaxMemoryStateEntity)
      MaxMemoryStateEntity = totalMemoryStates;

    NumNonEscaped += numNonEscaped;
    if (numNonEscaped > MaxNonEscapedMemoryStateEntity)
      MaxNonEscapedMemoryStateEntity = numNonEscaped;
  }

  void
  CountEntity(util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes)
  {
    uint64_t numAllocas = 0;
    uint64_t numMallocs = 0;
    uint64_t numDeltas = 0;
    uint64_t numImports = 0;
    uint64_t numLambdas = 0;
    uint64_t numExternal = 0;

    uint64_t numNonEscaped = 0;

    for (const auto memoryNode : memoryNodes.Items())
    {
      if (!memoryNode->IsModuleEscaping())
        numNonEscaped++;

      if (PointsToGraph::Node::Is<PointsToGraph::AllocaNode>(*memoryNode))
        numAllocas++;
      else if (PointsToGraph::Node::Is<PointsToGraph::MallocNode>(*memoryNode))
        numMallocs++;
      else if (PointsToGraph::Node::Is<PointsToGraph::DeltaNode>(*memoryNode))
        numDeltas++;
      else if (PointsToGraph::Node::Is<PointsToGraph::ImportNode>(*memoryNode))
        numImports++;
      else if (PointsToGraph::Node::Is<PointsToGraph::LambdaNode>(*memoryNode))
        numLambdas++;
      else if (PointsToGraph::Node::Is<PointsToGraph::ExternalMemoryNode>(*memoryNode))
        numExternal++;
      else
        JLM_UNREACHABLE("Unknown MemoryNode type");
    }

    CountEntity(
        numAllocas,
        numMallocs,
        numDeltas,
        numImports,
        numLambdas,
        numExternal,
        numNonEscaped);
  }
};

/** \brief Statistics class for memory state encoder encoding
 *
 */
class EncodingStatistics final : public util::Statistics
{
  // These are prefixes for statistics that count MemoryNode types
  static constexpr auto NumTotalAllocaState_ = "#TotalAllocaState";
  static constexpr auto NumTotalMallocState_ = "#TotalMallocState";
  static constexpr auto NumTotalDeltaState_ = "#TotalDeltaState";
  static constexpr auto NumTotalImportState_ = "#TotalImportState";
  static constexpr auto NumTotalLambdaState_ = "#TotalLambdaState";
  static constexpr auto NumTotalExternalState_ = "#TotalExternalState";
  // Among all the MemoryNodes counted above, how many of them are not marked as escaped
  static constexpr auto NumTotalNonEscapedState_ = "#TotalNonEscapedState";
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
    AddMeasurement(NumTotalAllocaState_ + suffix, counter.NumAllocas);
    AddMeasurement(NumTotalMallocState_ + suffix, counter.NumMallocs);
    AddMeasurement(NumTotalDeltaState_ + suffix, counter.NumDeltas);
    AddMeasurement(NumTotalImportState_ + suffix, counter.NumImports);
    AddMeasurement(NumTotalLambdaState_ + suffix, counter.NumLambdas);
    AddMeasurement(NumTotalExternalState_ + suffix, counter.NumExternal);
    AddMeasurement(NumTotalNonEscapedState_ + suffix, counter.NumNonEscaped);

    AddMeasurement(NumMaxMemoryState_ + suffix, counter.MaxMemoryStateEntity);
    AddMeasurement(NumMaxNonEscapedMemoryState_ + suffix, counter.MaxNonEscapedMemoryStateEntity);
  }
};

/** \brief Hash map for mapping points-to graph memory nodes to RVSDG memory states.
 */
class StateMap final
{
public:
  /**
   * Represents the pairing of a points-to graph's memory node and a memory state.
   */
  class MemoryNodeStatePair final
  {
    friend StateMap;

    MemoryNodeStatePair(const PointsToGraph::MemoryNode & memoryNode, rvsdg::Output & state)
        : MemoryNode_(&memoryNode),
          State_(&state)
    {
      JLM_ASSERT(is<MemoryStateType>(state.Type()));
    }

  public:
    [[nodiscard]] const PointsToGraph::MemoryNode &
    MemoryNode() const noexcept
    {
      return *MemoryNode_;
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
    const PointsToGraph::MemoryNode * MemoryNode_;
    rvsdg::Output * State_;
  };

  StateMap() = default;

  StateMap(const StateMap &) = delete;

  StateMap(StateMap &&) = delete;

  StateMap &
  operator=(const StateMap &) = delete;

  StateMap &
  operator=(StateMap &&) = delete;

  bool
  HasState(const PointsToGraph::MemoryNode & memoryNode) const noexcept
  {
    return states_.find(&memoryNode) != states_.end();
  }

  MemoryNodeStatePair *
  GetState(const PointsToGraph::MemoryNode & memoryNode) noexcept
  {
    JLM_ASSERT(HasState(memoryNode));
    return &states_.at(&memoryNode);
  }

  std::vector<MemoryNodeStatePair *>
  GetStates(const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    std::vector<MemoryNodeStatePair *> memoryNodeStatePairs;
    for (auto & memoryNode : memoryNodes.Items())
      memoryNodeStatePairs.push_back(GetState(*memoryNode));

    return memoryNodeStatePairs;
  }

  MemoryNodeStatePair *
  InsertState(const PointsToGraph::MemoryNode & memoryNode, rvsdg::Output & state)
  {
    JLM_ASSERT(!HasState(memoryNode));

    auto pair = std::make_pair<const PointsToGraph::MemoryNode *, MemoryNodeStatePair>(
        &memoryNode,
        { memoryNode, state });
    states_.insert(pair);
    return GetState(memoryNode);
  }

  static std::unique_ptr<StateMap>
  Create()
  {
    return std::make_unique<StateMap>();
  }

private:
  std::unordered_map<const PointsToGraph::MemoryNode *, MemoryNodeStatePair> states_;
};

/** \brief Hash map for mapping Rvsdg regions to StateMap class instances.
 */
class RegionalizedStateMap final
{
public:
  ~RegionalizedStateMap()
  {
    // Ensure that a PopRegion() was invoked for each invocation of a PushRegion().
    JLM_ASSERT(StateMaps_.empty());
  }

  explicit RegionalizedStateMap(const ModRefSummary & modRefSummary)
      : ModRefSummary_(modRefSummary)
  {}

  RegionalizedStateMap(const RegionalizedStateMap &) = delete;

  RegionalizedStateMap(RegionalizedStateMap &&) = delete;

  RegionalizedStateMap &
  operator=(const RegionalizedStateMap &) = delete;

  RegionalizedStateMap &
  operator=(RegionalizedStateMap &&) = delete;

  StateMap::MemoryNodeStatePair *
  InsertState(const PointsToGraph::MemoryNode & memoryNode, rvsdg::Output & state)
  {
    return GetStateMap(*state.region()).InsertState(memoryNode, state);
  }

  void
  ReplaceAddress(const rvsdg::Output & oldAddress, const rvsdg::Output & newAddress)
  {
    // TODO: This method is temporary until the new ModRefSummary interface is added
    ReplacementMap_[&newAddress] = &oldAddress;
  }

  const rvsdg::Output &
  LookupAddress(const rvsdg::Output & output)
  {
    // TODO: This method is temporary until the new ModRefSummary interface is added
    const rvsdg::Output * result = &output;
    while (true)
    {
      auto it = ReplacementMap_.find(result);
      if (it == ReplacementMap_.end())
        return *result;
      result = it->second;
    }
  }

  bool
  HasState(const rvsdg::Region & region, const PointsToGraph::MemoryNode & memoryNode)
  {
    return GetStateMap(region).HasState(memoryNode);
  }

  StateMap::MemoryNodeStatePair *
  GetState(const rvsdg::Region & region, const PointsToGraph::MemoryNode & memoryNode)
  {
    return GetStateMap(region).GetState(memoryNode);
  }

  std::vector<StateMap::MemoryNodeStatePair *>
  GetStates(
      const rvsdg::Region & region,
      const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    return GetStateMap(region).GetStates(memoryNodes);
  }

  std::vector<StateMap::MemoryNodeStatePair *>
  GetStates(const rvsdg::SimpleNode & node) noexcept
  {
    return GetStates(*node.region(), GetSimpleNodeModRef(node));
  }

  util::HashSet<const PointsToGraph::MemoryNode *>
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node)
  {
    // TODO: This implementation is temporary, until the new ModRefSummary interface is added
    util::HashSet<const PointsToGraph::MemoryNode *> result;
    const auto & addOutputToModRefSet = [&](const rvsdg::Output & output)
    {
      auto & trueOutput = LookupAddress(output);
      auto memoryNodes = ModRefSummary_.GetOutputNodes(trueOutput);
      result.UnionWithAndClear(memoryNodes);
    };

    if (jlm::rvsdg::is<StoreOperation>(&node))
    {
      addOutputToModRefSet(*StoreOperation::AddressInput(node).origin());
    }
    else if (jlm::rvsdg::is<LoadOperation>(&node))
    {
      addOutputToModRefSet(*LoadOperation::AddressInput(node).origin());
    }
    else if (jlm::rvsdg::is<FreeOperation>(&node))
    {
      addOutputToModRefSet(*node.input(0)->origin());
    }
    else if (jlm::rvsdg::is<MemCpyOperation>(&node))
    {
      addOutputToModRefSet(*node.input(0)->origin());
      addOutputToModRefSet(*node.input(1)->origin());
    }
    else if (jlm::rvsdg::is<AllocaOperation>(&node))
    {
      result.Insert(&ModRefSummary_.GetPointsToGraph().GetAllocaNode(node));
    }
    else if (jlm::rvsdg::is<MallocOperation>(&node))
    {
      result.Insert(&ModRefSummary_.GetPointsToGraph().GetMallocNode(node));
    }
    else if (jlm::rvsdg::is<CallOperation>(&node))
    {
      return ModRefSummary_.GetCallEntryNodes(node);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type.");
    }
    return result;
  }

  void
  PushRegion(const rvsdg::Region & region)
  {
    JLM_ASSERT(StateMaps_.find(&region) == StateMaps_.end());
    StateMaps_[&region] = StateMap::Create();
  }

  void
  PopRegion(const rvsdg::Region & region)
  {
    JLM_ASSERT(StateMaps_.find(&region) != StateMaps_.end());
    StateMaps_.erase(&region);
  }

private:

  StateMap &
  GetStateMap(const rvsdg::Region & region) const noexcept
  {
    JLM_ASSERT(StateMaps_.find(&region) != StateMaps_.end());
    return *StateMaps_.at(&region);
  }

  const ModRefSummary & ModRefSummary_;

  // TODO: Temporary until new ModRefSummary interface is added
  std::unordered_map<const rvsdg::Output *, const rvsdg::Output *> ReplacementMap_;

  std::unordered_map<const rvsdg::Region *, std::unique_ptr<StateMap>> StateMaps_;
};

/** \brief Context for the memory state encoder
 */
class MemoryStateEncoder::Context final
{
public:
  explicit Context(const ModRefSummary & modRefSummary)
      : RegionalizedStateMap_(modRefSummary),
        ModRefSummary_(modRefSummary)
  {}

  Context(const Context &) = delete;

  Context(Context &&) = delete;

  Context &
  operator=(const Context &) = delete;

  Context &
  operator=(Context &&) = delete;

  RegionalizedStateMap &
  GetRegionalizedStateMap() noexcept
  {
    return RegionalizedStateMap_;
  }

  const ModRefSummary &
  GetModRefSummary() const noexcept
  {
    return ModRefSummary_;
  }

  MemoryStateTypeCounter &
  GetInterProceduralRegionCounter()
  {
    return InterProceduralRegionCounter_;
  }

  MemoryStateTypeCounter &
  GetLoadCounter()
  {
    return LoadCounter_;
  }

  MemoryStateTypeCounter &
  GetStoreCounter()
  {
    return StoreCounter_;
  }

  MemoryStateTypeCounter &
  GetCallEntryMergeCounter()
  {
    return CallEntryMergeCounter_;
  }

  static std::unique_ptr<MemoryStateEncoder::Context>
  Create(const ModRefSummary & modRefSummary)
  {
    return std::make_unique<Context>(modRefSummary);
  }

private:
  RegionalizedStateMap RegionalizedStateMap_;
  const ModRefSummary & ModRefSummary_;

  // Counters used for producing statistics about memory states
  MemoryStateTypeCounter InterProceduralRegionCounter_;
  MemoryStateTypeCounter LoadCounter_;
  MemoryStateTypeCounter StoreCounter_;
  MemoryStateTypeCounter CallEntryMergeCounter_;
};

static std::vector<MemoryNodeId>
GetMemoryNodeIds(const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
{
  std::vector<MemoryNodeId> memoryNodeIds;
  for (const auto memoryNode : memoryNodes.Items())
  {
    memoryNodeIds.push_back(memoryNode->GetId());
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
  Context_ = Context::Create(modRefSummary);
  auto statistics = EncodingStatistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->Start(rvsdgModule.Rvsdg());
  EncodeRegion(rvsdgModule.Rvsdg().GetRootRegion());
  statistics->Stop();

  statistics->AddIntraProceduralRegionMemoryStateCounts(
      Context_->GetInterProceduralRegionCounter());
  statistics->AddLoadMemoryStateCounts(Context_->GetLoadCounter());
  statistics->AddStoreMemoryStateCounts(Context_->GetStoreCounter());
  statistics->AddCallEntryMergeStateCounts(Context_->GetCallEntryMergeCounter());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done with the encoding
  Context_.reset();

  // Remove all nodes that became dead throughout the encoding.
  DeadNodeElimination deadNodeElimination;
  deadNodeElimination.Run(rvsdgModule, statisticsCollector);
}

void
MemoryStateEncoder::EncodeRegion(rvsdg::Region & region)
{
  using namespace jlm::rvsdg;

  TopDownTraverser traverser(&region);
  for (auto & node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const SimpleNode *>(node))
    {
      EncodeSimpleNode(*simpleNode);
    }
    else if (auto structuralNode = dynamic_cast<StructuralNode *>(node))
    {
      EncodeStructuralNode(*structuralNode);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type.");
    }
  }
}

void
MemoryStateEncoder::EncodeStructuralNode(rvsdg::StructuralNode & structuralNode)
{
  if (auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(&structuralNode))
  {
    EncodeLambda(*lambdaNode);
  }
  else if (auto deltaNode = dynamic_cast<const rvsdg::DeltaNode *>(&structuralNode))
  {
    EncodeDelta(*deltaNode);
  }
  else if (auto phiNode = dynamic_cast<const rvsdg::PhiNode *>(&structuralNode))
  {
    EncodePhi(*phiNode);
  }
  else if (auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(&structuralNode))
  {
    EncodeGamma(*gammaNode);
  }
  else if (auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(&structuralNode))
  {
    EncodeTheta(*thetaNode);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled node type.");
  }
}

void
MemoryStateEncoder::EncodeSimpleNode(const rvsdg::SimpleNode & simpleNode)
{
  if (is<AllocaOperation>(&simpleNode))
  {
    EncodeAlloca(simpleNode);
  }
  else if (is<MallocOperation>(&simpleNode))
  {
    EncodeMalloc(simpleNode);
  }
  else if (is<LoadOperation>(&simpleNode))
  {
    EncodeLoad(simpleNode);
  }
  else if (is<StoreOperation>(&simpleNode))
  {
    EncodeStore(simpleNode);
  }
  else if (is<CallOperation>(&simpleNode))
  {
    EncodeCall(simpleNode);
  }
  else if (is<FreeOperation>(&simpleNode))
  {
    EncodeFree(simpleNode);
  }
  else if (is<MemCpyOperation>(&simpleNode))
  {
    EncodeMemcpy(simpleNode);
  }
  else if (is<MemoryStateOperation>(&simpleNode))
  {
    // Nothing needs to be done
  }
  else
  {
    // Ensure we took care of all memory state consuming nodes
    JLM_ASSERT(!ShouldHandle(simpleNode));
  }
}

void
MemoryStateEncoder::EncodeAlloca(const rvsdg::SimpleNode & allocaNode)
{
  JLM_ASSERT(is<AllocaOperation>(&allocaNode));

  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto allocaMemoryNodes = stateMap.GetSimpleNodeModRef(allocaNode);
  JLM_ASSERT(allocaMemoryNodes.Size() == 1);
  auto & allocaMemoryNode = **allocaMemoryNodes.Items().begin();
  auto & allocaNodeStateOutput = *allocaNode.output(1);

  // If a state representing the alloca already exists in the region,
  // merge it with the state created by the alloca using a MemoryStateJoin node.
  if (stateMap.HasState(*allocaNode.region(), allocaMemoryNode))
  {
    auto memoryNodeStatePair = stateMap.GetState(*allocaNode.region(), allocaMemoryNode);
    auto & joinNode = MemoryStateJoinOperation::CreateNode(
        { &allocaNodeStateOutput, &memoryNodeStatePair->State() });
    auto & joinOutput = *joinNode.output(0);
    memoryNodeStatePair->ReplaceState(joinOutput);
  }
  else
  {
    stateMap.InsertState(allocaMemoryNode, allocaNodeStateOutput);
  }
}

void
MemoryStateEncoder::EncodeMalloc(const rvsdg::SimpleNode & mallocNode)
{
  JLM_ASSERT(is<MallocOperation>(&mallocNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto mallocMemoryNodes = stateMap.GetSimpleNodeModRef(mallocNode);
  JLM_ASSERT(mallocMemoryNodes.Size() == 1);
  auto & mallocMemoryNode = **mallocMemoryNodes.Items().begin();

  auto & mallocNodeStateOutput = *mallocNode.output(1);

  // We use a static heap model. This means that multiple invocations of an malloc
  // at runtime can refer to the same abstract memory location. We therefore need to
  // merge the previous and the current state to ensure that the previous state
  // is not just simply replaced and therefore "lost".
  if (stateMap.HasState(*mallocNode.region(), mallocMemoryNode))
  {
    auto memoryNodeStatePair = stateMap.GetState(*mallocNode.region(), mallocMemoryNode);
    auto & joinNode = MemoryStateJoinOperation::CreateNode(
        { &mallocNodeStateOutput, &memoryNodeStatePair->State() });
    auto & joinOutput = *joinNode.output(0);
    memoryNodeStatePair->ReplaceState(joinOutput);
  }
  else
  {
    stateMap.InsertState(mallocMemoryNode, mallocNodeStateOutput);
  }
}

void
MemoryStateEncoder::EncodeLoad(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<LoadOperation>(&node));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  const auto memoryNodes = stateMap.GetSimpleNodeModRef(node);
  Context_->GetLoadCounter().CountEntity(memoryNodes);

  const auto memoryNodeStatePairs = stateMap.GetStates(*node.region(), memoryNodes);
  const auto memoryStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  const auto & newLoadNode = ReplaceLoadNode(node, memoryStates);

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      LoadOperation::MemoryStateOutputs(newLoadNode));

  if (is<PointerType>(LoadOperation::LoadedValueOutput(node).Type()))
  {
    stateMap.ReplaceAddress(
        LoadOperation::LoadedValueOutput(node),
        LoadOperation::LoadedValueOutput(newLoadNode));
  }
}

void
MemoryStateEncoder::EncodeStore(const rvsdg::SimpleNode & node)
{
  auto & stateMap = Context_->GetRegionalizedStateMap();

  const auto memoryNodes = stateMap.GetSimpleNodeModRef(node);
  Context_->GetStoreCounter().CountEntity(memoryNodes);

  const auto memoryNodeStatePairs = stateMap.GetStates(*node.region(), memoryNodes);
  const auto memoryStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  const auto & newStoreNode = ReplaceStoreNode(node, memoryStates);

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      StoreOperation::MemoryStateOutputs(newStoreNode));
}

void
MemoryStateEncoder::EncodeFree(const rvsdg::SimpleNode & freeNode)
{
  JLM_ASSERT(is<FreeOperation>(&freeNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = freeNode.input(0)->origin();
  auto ioState = freeNode.input(freeNode.ninputs() - 1)->origin();
  auto memoryNodeStatePairs = stateMap.GetStates(freeNode);
  auto inStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto outputs = FreeOperation::Create(address, inStates, ioState);

  // Redirect IO state edge
  freeNode.output(freeNode.noutputs() - 1)->divert_users(outputs.back());

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      { outputs.begin(), std::prev(outputs.end()) });
}

void
MemoryStateEncoder::EncodeCall(const rvsdg::SimpleNode & callNode)
{
  EncodeCallEntry(callNode);
  EncodeCallExit(callNode);
}

void
MemoryStateEncoder::EncodeCallEntry(const rvsdg::SimpleNode & callNode)
{
  const auto region = callNode.region();
  auto & regionalizedStateMap = Context_->GetRegionalizedStateMap();
  const auto & memoryNodes = regionalizedStateMap.GetSimpleNodeModRef(callNode);
  Context_->GetCallEntryMergeCounter().CountEntity(memoryNodes);

  std::vector<rvsdg::Output *> states;
  std::vector<MemoryNodeId> memoryNodeIds;
  for (const auto memoryNode : memoryNodes.Items())
  {
    states.emplace_back(&regionalizedStateMap.GetState(*region, *memoryNode)->State());
    memoryNodeIds.push_back(memoryNode->GetId());
  }

  const auto state =
      CallEntryMemoryStateMergeOperation::CreateNode(*region, states, memoryNodeIds).output(0);
  CallOperation::GetMemoryStateInput(callNode).divert_to(state);
}

void
MemoryStateEncoder::EncodeCallExit(const rvsdg::SimpleNode & callNode)
{
  auto & stateMap = Context_->GetRegionalizedStateMap();
  const auto & memoryNodes = stateMap.GetSimpleNodeModRef(callNode);

  const auto states = rvsdg::outputs(&CallExitMemoryStateSplitOperation::CreateNode(
      CallOperation::GetMemoryStateOutput(callNode),
      GetMemoryNodeIds(memoryNodes)));

  const auto memoryNodeStatePairs = stateMap.GetStates(*callNode.region(), memoryNodes);
  StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, states);
}

void
MemoryStateEncoder::EncodeMemcpy(const rvsdg::SimpleNode & memcpyNode)
{
  JLM_ASSERT(is<MemCpyOperation>(&memcpyNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto memoryNodeStatePairs = stateMap.GetStates(memcpyNode);
  auto memoryStateOperands = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto memoryStateResults = ReplaceMemcpyNode(memcpyNode, memoryStateOperands);

  StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, memoryStateResults);
}

void
MemoryStateEncoder::EncodeLambda(const rvsdg::LambdaNode & lambdaNode)
{
  EncodeLambdaEntry(lambdaNode);
  EncodeRegion(*lambdaNode.subregion());
  EncodeLambdaExit(lambdaNode);
}

void
MemoryStateEncoder::EncodeLambdaEntry(const rvsdg::LambdaNode & lambdaNode)
{
  auto & memoryStateArgument = GetMemoryStateRegionArgument(lambdaNode);
  JLM_ASSERT(memoryStateArgument.nusers() == 1);
  auto & memoryStateArgumentUser = memoryStateArgument.SingleUser();

  const auto & memoryNodes = Context_->GetModRefSummary().GetLambdaEntryNodes(lambdaNode);
  Context_->GetInterProceduralRegionCounter().CountEntity(memoryNodes);

  const auto memoryNodeIds = GetMemoryNodeIds(memoryNodes);
  auto & stateMap = Context_->GetRegionalizedStateMap();

  stateMap.PushRegion(*lambdaNode.subregion());
  const auto states = rvsdg::outputs(&LambdaEntryMemoryStateSplitOperation::CreateNode(
      memoryStateArgument,
      memoryNodes.Size(),
      memoryNodeIds));

  size_t n = 0;
  for (auto & memoryNode : memoryNodes.Items())
    stateMap.InsertState(*memoryNode, *states[n++]);

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
    memoryStateArgumentUser.divert_to(state);
  }
}

void
MemoryStateEncoder::EncodeLambdaExit(const rvsdg::LambdaNode & lambdaNode)
{
  auto & memoryNodes = Context_->GetModRefSummary().GetLambdaExitNodes(lambdaNode);
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto & memoryStateResult = GetMemoryStateRegionResult(lambdaNode);

  std::vector<rvsdg::Output *> states;
  std::vector<MemoryNodeId> memoryNodeIds;
  auto & subregion = *lambdaNode.subregion();
  const auto memoryNodeStatePairs = stateMap.GetStates(subregion, memoryNodes);
  for (const auto memoryNodeStatePair : memoryNodeStatePairs)
  {
    states.push_back(&memoryNodeStatePair->State());
    memoryNodeIds.push_back(memoryNodeStatePair->MemoryNode().GetId());
  }

  const auto mergedState =
      LambdaExitMemoryStateMergeOperation::CreateNode(subregion, states, memoryNodeIds).output(0);
  memoryStateResult.divert_to(mergedState);

  stateMap.PopRegion(*lambdaNode.subregion());
}

void
MemoryStateEncoder::EncodePhi(const rvsdg::PhiNode & phiNode)
{
  EncodeRegion(*phiNode.subregion());
}

void
MemoryStateEncoder::EncodeDelta(const rvsdg::DeltaNode &)
{
  // Nothing needs to be done
}

void
MemoryStateEncoder::EncodeGamma(rvsdg::GammaNode & gammaNode)
{
  for (auto & subregion : gammaNode.Subregions())
    Context_->GetRegionalizedStateMap().PushRegion(subregion);

  EncodeGammaEntry(gammaNode);

  for (auto & subregion : gammaNode.Subregions())
    EncodeRegion(subregion);

  EncodeGammaExit(gammaNode);

  for (auto & subregion : gammaNode.Subregions())
    Context_->GetRegionalizedStateMap().PopRegion(subregion);
}

void
MemoryStateEncoder::EncodeGammaEntry(rvsdg::GammaNode & gammaNode)
{
  auto region = gammaNode.region();
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto memoryNodes = Context_->GetModRefSummary().GetGammaEntryNodes(gammaNode);

  // Count the memory state arguments once per subregion
  for ([[maybe_unused]] auto & subregion : gammaNode.Subregions())
    Context_->GetInterProceduralRegionCounter().CountEntity(memoryNodes);

  auto memoryNodeStatePairs = stateMap.GetStates(*region, memoryNodes);
  for (auto & memoryNodeStatePair : memoryNodeStatePairs)
  {
    auto gammaInput = gammaNode.AddEntryVar(&memoryNodeStatePair->State());
    for (auto & argument : gammaInput.branchArgument)
      stateMap.InsertState(memoryNodeStatePair->MemoryNode(), *argument);
  }
}

void
MemoryStateEncoder::EncodeGammaExit(rvsdg::GammaNode & gammaNode)
{
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto memoryNodes = Context_->GetModRefSummary().GetGammaExitNodes(gammaNode);
  auto memoryNodeStatePairs = stateMap.GetStates(*gammaNode.region(), memoryNodes);

  for (auto & memoryNodeStatePair : memoryNodeStatePairs)
  {
    std::vector<rvsdg::Output *> states;

    for (auto & subregion : gammaNode.Subregions())
    {
      auto & state = stateMap.GetState(subregion, memoryNodeStatePair->MemoryNode())->State();
      states.push_back(&state);
    }

    auto state = gammaNode.AddExitVar(states).output;
    memoryNodeStatePair->ReplaceState(*state);
  }
}

void
MemoryStateEncoder::EncodeTheta(rvsdg::ThetaNode & thetaNode)
{
  Context_->GetRegionalizedStateMap().PushRegion(*thetaNode.subregion());

  auto thetaStateOutputs = EncodeThetaEntry(thetaNode);
  EncodeRegion(*thetaNode.subregion());
  EncodeThetaExit(thetaNode, thetaStateOutputs);

  Context_->GetRegionalizedStateMap().PopRegion(*thetaNode.subregion());
}

std::vector<rvsdg::Output *>
MemoryStateEncoder::EncodeThetaEntry(rvsdg::ThetaNode & thetaNode)
{
  auto region = thetaNode.region();
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto & memoryNodes = Context_->GetModRefSummary().GetThetaEntryExitNodes(thetaNode);
  Context_->GetInterProceduralRegionCounter().CountEntity(memoryNodes);

  std::vector<rvsdg::Output *> thetaStateOutputs;
  auto memoryNodeStatePairs = stateMap.GetStates(*region, memoryNodes);
  for (auto & memoryNodeStatePair : memoryNodeStatePairs)
  {
    auto loopvar = thetaNode.AddLoopVar(&memoryNodeStatePair->State());
    stateMap.InsertState(memoryNodeStatePair->MemoryNode(), *loopvar.pre);
    thetaStateOutputs.push_back(loopvar.output);
  }

  return thetaStateOutputs;
}

void
MemoryStateEncoder::EncodeThetaExit(
    rvsdg::ThetaNode & thetaNode,
    const std::vector<rvsdg::Output *> & thetaStateOutputs)
{
  auto subregion = thetaNode.subregion();
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto & memoryNodes = Context_->GetModRefSummary().GetThetaEntryExitNodes(thetaNode);
  auto memoryNodeStatePairs = stateMap.GetStates(*thetaNode.region(), memoryNodes);

  JLM_ASSERT(memoryNodeStatePairs.size() == thetaStateOutputs.size());
  for (size_t n = 0; n < thetaStateOutputs.size(); n++)
  {
    auto thetaStateOutput = thetaStateOutputs[n];
    auto & memoryNodeStatePair = memoryNodeStatePairs[n];
    auto & memoryNode = memoryNodeStatePair->MemoryNode();
    auto loopvar = thetaNode.MapOutputLoopVar(*thetaStateOutput);
    JLM_ASSERT(loopvar.input->origin() == &memoryNodeStatePair->State());

    auto & subregionState = stateMap.GetState(*subregion, memoryNode)->State();
    loopvar.post->divert_to(&subregionState);
    memoryNodeStatePair->ReplaceState(*thetaStateOutput);
  }
}

rvsdg::SimpleNode &
MemoryStateEncoder::ReplaceLoadNode(
    const rvsdg::SimpleNode & node,
    const std::vector<rvsdg::Output *> & memoryStates)
{
  JLM_ASSERT(is<LoadOperation>(&node));

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
  JLM_ASSERT(is<MemCpyOperation>(&memcpyNode));

  auto destination = memcpyNode.input(0)->origin();
  auto source = memcpyNode.input(1)->origin();
  auto length = memcpyNode.input(2)->origin();

  if (is<MemCpyVolatileOperation>(&memcpyNode))
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
  else if (is<MemCpyNonVolatileOperation>(&memcpyNode))
  {
    return MemCpyNonVolatileOperation::create(destination, source, length, memoryStates);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled memcpy operation type.");
  }
}

bool
MemoryStateEncoder::ShouldHandle(const rvsdg::SimpleNode & simpleNode) noexcept
{
  for (size_t n = 0; n < simpleNode.ninputs(); n++)
  {
    auto input = simpleNode.input(n);
    if (is<MemoryStateType>(input->Type()))
    {
      return true;
    }
  }

  for (size_t n = 0; n < simpleNode.noutputs(); n++)
  {
    auto output = simpleNode.output(n);
    if (is<MemoryStateType>(output->Type()))
    {
      return true;
    }
  }

  return false;
}

}
