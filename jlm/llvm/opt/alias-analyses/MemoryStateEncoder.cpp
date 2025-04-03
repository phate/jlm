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
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm::aa
{

/** \brief Statistics class for memory state encoder encoding
 *
 */
class EncodingStatistics final : public util::Statistics
{
public:
  ~EncodingStatistics() override = default;

  explicit EncodingStatistics(const util::filepath & sourceFile)
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

  static std::unique_ptr<EncodingStatistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<EncodingStatistics>(sourceFile);
  }
};

/** \brief A cache for points-to graph memory nodes of pointer outputs.
 *
 */
class MemoryNodeCache final
{
  explicit MemoryNodeCache(const ModRefSummary & modRefSummary)
      : ModRefSummary_(modRefSummary)
  {}

public:
  MemoryNodeCache(const MemoryNodeCache &) = delete;

  MemoryNodeCache(MemoryNodeCache &&) = delete;

  MemoryNodeCache &
  operator=(const MemoryNodeCache &) = delete;

  MemoryNodeCache &
  operator=(MemoryNodeCache &&) = delete;

  bool
  Contains(const rvsdg::output & output) const noexcept
  {
    return MemoryNodeMap_.find(&output) != MemoryNodeMap_.end();
  }

  util::HashSet<const PointsToGraph::MemoryNode *>
  GetMemoryNodes(const rvsdg::output & output)
  {
    JLM_ASSERT(is<PointerType>(output.type()));

    if (Contains(output))
      return MemoryNodeMap_[&output];

    auto memoryNodes = ModRefSummary_.GetOutputNodes(output);

    // There is no need to cache the memory nodes, if the address is only once used.
    if (output.nusers() <= 1)
      return memoryNodes;

    MemoryNodeMap_[&output] = std::move(memoryNodes);

    return MemoryNodeMap_[&output];
  }

  void
  ReplaceAddress(const rvsdg::output & oldAddress, const rvsdg::output & newAddress)
  {
    JLM_ASSERT(!Contains(oldAddress));
    JLM_ASSERT(!Contains(newAddress));

    MemoryNodeMap_[&newAddress] = ModRefSummary_.GetOutputNodes(oldAddress);
  }

  static std::unique_ptr<MemoryNodeCache>
  Create(const ModRefSummary & memoryNodeProvisioning)
  {
    return std::unique_ptr<MemoryNodeCache>(new MemoryNodeCache(memoryNodeProvisioning));
  }

private:
  const ModRefSummary & ModRefSummary_;
  std::unordered_map<const rvsdg::output *, util::HashSet<const PointsToGraph::MemoryNode *>>
      MemoryNodeMap_;
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

    MemoryNodeStatePair(const PointsToGraph::MemoryNode & memoryNode, rvsdg::output & state)
        : MemoryNode_(&memoryNode),
          State_(&state)
    {
      JLM_ASSERT(is<MemoryStateType>(state.type()));
    }

  public:
    [[nodiscard]] const PointsToGraph::MemoryNode &
    MemoryNode() const noexcept
    {
      return *MemoryNode_;
    }

    [[nodiscard]] rvsdg::output &
    State() const noexcept
    {
      return *State_;
    }

    void
    ReplaceState(rvsdg::output & state) noexcept
    {
      JLM_ASSERT(State_->region() == state.region());
      JLM_ASSERT(is<MemoryStateType>(state.type()));

      State_ = &state;
    }

    static void
    ReplaceStates(
        const std::vector<MemoryNodeStatePair *> & memoryNodeStatePairs,
        const std::vector<rvsdg::output *> & states)
    {
      JLM_ASSERT(memoryNodeStatePairs.size() == states.size());
      for (size_t n = 0; n < memoryNodeStatePairs.size(); n++)
        memoryNodeStatePairs[n]->ReplaceState(*states[n]);
    }

    static void
    ReplaceStates(
        const std::vector<MemoryNodeStatePair *> & memoryNodeStatePairs,
        const LoadNode::MemoryStateOutputRange & states)
    {
      auto it = states.begin();
      for (auto memoryNodeStatePair : memoryNodeStatePairs)
      {
        memoryNodeStatePair->ReplaceState(*it);
        it++;
      }
      JLM_ASSERT(it.value() == nullptr);
    }

    static void
    ReplaceStates(
        const std::vector<MemoryNodeStatePair *> & memoryNodeStatePairs,
        const StoreNode::MemoryStateOutputRange & states)
    {
      auto it = states.begin();
      for (auto memoryNodeStatePair : memoryNodeStatePairs)
      {
        memoryNodeStatePair->ReplaceState(*it);
        it++;
      }
      JLM_ASSERT(it.value() == nullptr);
    }

    static std::vector<rvsdg::output *>
    States(const std::vector<MemoryNodeStatePair *> & memoryNodeStatePairs)
    {
      std::vector<rvsdg::output *> states;
      for (auto & memoryNodeStatePair : memoryNodeStatePairs)
        states.push_back(memoryNodeStatePair->State_);

      return states;
    }

  private:
    const PointsToGraph::MemoryNode * MemoryNode_;
    rvsdg::output * State_;
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
  InsertState(const PointsToGraph::MemoryNode & memoryNode, rvsdg::output & state)
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
    JLM_ASSERT(MemoryNodeCacheMaps_.empty());
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
  InsertState(const PointsToGraph::MemoryNode & memoryNode, rvsdg::output & state)
  {
    return GetStateMap(*state.region()).InsertState(memoryNode, state);
  }

  StateMap::MemoryNodeStatePair *
  InsertUndefinedState(rvsdg::Region & region, const PointsToGraph::MemoryNode & memoryNode)
  {
    auto & undefinedState = GetOrInsertUndefinedMemoryState(region);
    return InsertState(memoryNode, undefinedState);
  }

  void
  ReplaceAddress(const rvsdg::output & oldAddress, const rvsdg::output & newAddress)
  {
    GetMemoryNodeCache(*oldAddress.region()).ReplaceAddress(oldAddress, newAddress);
  }

  std::vector<StateMap::MemoryNodeStatePair *>
  GetStates(const rvsdg::output & output) noexcept
  {
    auto memoryNodes = GetMemoryNodes(output);
    return memoryNodes.Size() == 0 ? std::vector<StateMap::MemoryNodeStatePair *>()
                                   : GetStates(*output.region(), memoryNodes);
  }

  std::vector<StateMap::MemoryNodeStatePair *>
  GetStates(
      const rvsdg::Region & region,
      const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    return GetStateMap(region).GetStates(memoryNodes);
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

  util::HashSet<const PointsToGraph::MemoryNode *>
  GetMemoryNodes(const rvsdg::output & output)
  {
    auto & memoryNodeCache = GetMemoryNodeCache(*output.region());
    return memoryNodeCache.GetMemoryNodes(output);
  }

  void
  PushRegion(const rvsdg::Region & region)
  {
    JLM_ASSERT(StateMaps_.find(&region) == StateMaps_.end());
    JLM_ASSERT(MemoryNodeCacheMaps_.find(&region) == MemoryNodeCacheMaps_.end());

    StateMaps_[&region] = StateMap::Create();
    MemoryNodeCacheMaps_[&region] = MemoryNodeCache::Create(ModRefSummary_);
  }

  void
  PopRegion(const rvsdg::Region & region)
  {
    JLM_ASSERT(StateMaps_.find(&region) != StateMaps_.end());
    JLM_ASSERT(MemoryNodeCacheMaps_.find(&region) != MemoryNodeCacheMaps_.end());

    StateMaps_.erase(&region);
    MemoryNodeCacheMaps_.erase(&region);
  }

private:
  rvsdg::output &
  GetOrInsertUndefinedMemoryState(rvsdg::Region & region)
  {
    return HasUndefinedMemoryState(region) ? GetUndefinedMemoryState(region)
                                           : InsertUndefinedMemoryState(region);
  }

  bool
  HasUndefinedMemoryState(const rvsdg::Region & region) const noexcept
  {
    return UndefinedMemoryStates_.find(&region) != UndefinedMemoryStates_.end();
  }

  rvsdg::output &
  GetUndefinedMemoryState(const rvsdg::Region & region) const noexcept
  {
    JLM_ASSERT(HasUndefinedMemoryState(region));
    return *UndefinedMemoryStates_.find(&region)->second;
  }

  rvsdg::output &
  InsertUndefinedMemoryState(rvsdg::Region & region) noexcept
  {
    auto undefinedMemoryState = UndefValueOperation::Create(region, MemoryStateType::Create());
    UndefinedMemoryStates_[&region] = undefinedMemoryState;
    return *undefinedMemoryState;
  }

  StateMap &
  GetStateMap(const rvsdg::Region & region) const noexcept
  {
    JLM_ASSERT(StateMaps_.find(&region) != StateMaps_.end());
    return *StateMaps_.at(&region);
  }

  MemoryNodeCache &
  GetMemoryNodeCache(const rvsdg::Region & region) const noexcept
  {
    JLM_ASSERT(MemoryNodeCacheMaps_.find(&region) != MemoryNodeCacheMaps_.end());
    return *MemoryNodeCacheMaps_.at(&region);
  }

  std::unordered_map<const rvsdg::Region *, std::unique_ptr<StateMap>> StateMaps_;
  std::unordered_map<const rvsdg::Region *, std::unique_ptr<MemoryNodeCache>> MemoryNodeCacheMaps_;
  std::unordered_map<const rvsdg::Region *, rvsdg::output *> UndefinedMemoryStates_;

  const ModRefSummary & ModRefSummary_;
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

  static std::unique_ptr<MemoryStateEncoder::Context>
  Create(const ModRefSummary & modRefSummary)
  {
    return std::make_unique<Context>(modRefSummary);
  }

private:
  RegionalizedStateMap RegionalizedStateMap_;
  const ModRefSummary & ModRefSummary_;
};

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
  else if (auto deltaNode = dynamic_cast<const delta::node *>(&structuralNode))
  {
    EncodeDelta(*deltaNode);
  }
  else if (auto phiNode = dynamic_cast<const phi::node *>(&structuralNode))
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
  if (is<alloca_op>(&simpleNode))
  {
    EncodeAlloca(simpleNode);
  }
  else if (is<malloc_op>(&simpleNode))
  {
    EncodeMalloc(simpleNode);
  }
  else if (auto loadNode = dynamic_cast<const LoadNode *>(&simpleNode))
  {
    EncodeLoad(*loadNode);
  }
  else if (auto storeNode = dynamic_cast<const StoreNode *>(&simpleNode))
  {
    EncodeStore(*storeNode);
  }
  else if (auto callNode = dynamic_cast<const CallNode *>(&simpleNode))
  {
    EncodeCall(*callNode);
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
  JLM_ASSERT(is<alloca_op>(&allocaNode));

  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto & allocaMemoryNode =
      Context_->GetModRefSummary().GetPointsToGraph().GetAllocaNode(allocaNode);
  auto & allocaNodeStateOutput = *allocaNode.output(1);

  if (stateMap.HasState(*allocaNode.region(), allocaMemoryNode))
  {
    // The state for the alloca memory node should already exist in case of lifetime agnostic
    // provisioning.
    auto memoryNodeStatePair = stateMap.GetState(*allocaNode.region(), allocaMemoryNode);
    memoryNodeStatePair->ReplaceState(allocaNodeStateOutput);
  }
  else
  {
    stateMap.InsertState(allocaMemoryNode, allocaNodeStateOutput);
  }
}

void
MemoryStateEncoder::EncodeMalloc(const rvsdg::SimpleNode & mallocNode)
{
  JLM_ASSERT(is<malloc_op>(&mallocNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto & mallocMemoryNode =
      Context_->GetModRefSummary().GetPointsToGraph().GetMallocNode(mallocNode);

  // We use a static heap model. This means that multiple invocations of an malloc
  // at runtime can refer to the same abstract memory location. We therefore need to
  // merge the previous and the current state to ensure that the previous state
  // is not just simply replaced and therefore "lost".
  auto memoryNodeStatePair = stateMap.GetState(*mallocNode.region(), mallocMemoryNode);
  auto mallocState = mallocNode.output(1);
  auto mergedState =
      MemoryStateMergeOperation::Create({ mallocState, &memoryNodeStatePair->State() });
  memoryNodeStatePair->ReplaceState(*mergedState);
}

void
MemoryStateEncoder::EncodeLoad(const LoadNode & loadNode)
{
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = loadNode.GetAddressInput().origin();
  auto memoryNodeStatePairs = stateMap.GetStates(*address);
  auto memoryStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto & newLoadNode = ReplaceLoadNode(loadNode, memoryStates);

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      newLoadNode.MemoryStateOutputs());

  if (is<PointerType>(loadNode.GetOperation().GetLoadedType()))
    stateMap.ReplaceAddress(loadNode.GetLoadedValueOutput(), newLoadNode.GetLoadedValueOutput());
}

void
MemoryStateEncoder::EncodeStore(const StoreNode & storeNode)
{
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = storeNode.GetAddressInput().origin();
  auto memoryNodeStatePairs = stateMap.GetStates(*address);
  auto memoryStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto & newStoreNode = ReplaceStoreNode(storeNode, memoryStates);

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      newStoreNode.MemoryStateOutputs());
}

void
MemoryStateEncoder::EncodeFree(const rvsdg::SimpleNode & freeNode)
{
  JLM_ASSERT(is<FreeOperation>(&freeNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = freeNode.input(0)->origin();
  auto ioState = freeNode.input(freeNode.ninputs() - 1)->origin();
  auto memoryNodeStatePairs = stateMap.GetStates(*address);
  auto inStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto outputs = FreeOperation::Create(address, inStates, ioState);

  // Redirect IO state edge
  freeNode.output(freeNode.noutputs() - 1)->divert_users(outputs.back());

  StateMap::MemoryNodeStatePair::ReplaceStates(
      memoryNodeStatePairs,
      { outputs.begin(), std::prev(outputs.end()) });
}

void
MemoryStateEncoder::EncodeCall(const CallNode & callNode)
{
  EncodeCallEntry(callNode);
  EncodeCallExit(callNode);
}

void
MemoryStateEncoder::EncodeCallEntry(const CallNode & callNode)
{
  auto region = callNode.region();
  auto & regionalizedStateMap = Context_->GetRegionalizedStateMap();
  auto & memoryNodes = Context_->GetModRefSummary().GetCallEntryNodes(callNode);

  std::vector<StateMap::MemoryNodeStatePair *> memoryNodeStatePairs;
  for (auto memoryNode : memoryNodes.Items())
  {
    if (regionalizedStateMap.HasState(*region, *memoryNode))
    {
      memoryNodeStatePairs.emplace_back(regionalizedStateMap.GetState(*region, *memoryNode));
    }
    else
    {
      // The state might not exist on the call side in case of lifetime aware provisioning
      memoryNodeStatePairs.emplace_back(
          regionalizedStateMap.InsertUndefinedState(*region, *memoryNode));
    }
  }

  auto states = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);
  auto & state = CallEntryMemoryStateMergeOperation::Create(*region, states);
  callNode.GetMemoryStateInput()->divert_to(&state);
}

void
MemoryStateEncoder::EncodeCallExit(const CallNode & callNode)
{
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto & memoryNodes = Context_->GetModRefSummary().GetCallExitNodes(callNode);

  auto states = CallExitMemoryStateSplitOperation::Create(
      *callNode.GetMemoryStateOutput(),
      memoryNodes.Size());
  auto memoryNodeStatePairs = stateMap.GetStates(*callNode.region(), memoryNodes);
  StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, states);
}

void
MemoryStateEncoder::EncodeMemcpy(const rvsdg::SimpleNode & memcpyNode)
{
  JLM_ASSERT(is<MemCpyOperation>(&memcpyNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto destination = memcpyNode.input(0)->origin();
  auto source = memcpyNode.input(1)->origin();

  auto destMemoryNodeStatePairs = stateMap.GetStates(*destination);
  auto srcMemoryNodeStatePairs = stateMap.GetStates(*source);

  auto memoryStateOperands = StateMap::MemoryNodeStatePair::States(destMemoryNodeStatePairs);
  auto srcStates = StateMap::MemoryNodeStatePair::States(srcMemoryNodeStatePairs);
  memoryStateOperands.insert(memoryStateOperands.end(), srcStates.begin(), srcStates.end());

  auto memoryStateResults = ReplaceMemcpyNode(memcpyNode, memoryStateOperands);

  auto end = std::next(memoryStateResults.begin(), (ssize_t)destMemoryNodeStatePairs.size());
  StateMap::MemoryNodeStatePair::ReplaceStates(
      destMemoryNodeStatePairs,
      { memoryStateResults.begin(),
        std::next(memoryStateResults.begin(), (ssize_t)destMemoryNodeStatePairs.size()) });

  JLM_ASSERT(
      (size_t)std::distance(end, memoryStateResults.end()) == srcMemoryNodeStatePairs.size());
  StateMap::MemoryNodeStatePair::ReplaceStates(
      srcMemoryNodeStatePairs,
      { end, memoryStateResults.end() });
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
  auto memoryStateArgumentUser = *memoryStateArgument.begin();

  auto & memoryNodes = Context_->GetModRefSummary().GetLambdaEntryNodes(lambdaNode);
  auto & stateMap = Context_->GetRegionalizedStateMap();

  stateMap.PushRegion(*lambdaNode.subregion());

  auto states =
      LambdaEntryMemoryStateSplitOperation::Create(memoryStateArgument, memoryNodes.Size());

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
    memoryStateArgumentUser->divert_to(state);
  }
}

void
MemoryStateEncoder::EncodeLambdaExit(const rvsdg::LambdaNode & lambdaNode)
{
  auto subregion = lambdaNode.subregion();
  auto & memoryNodes = Context_->GetModRefSummary().GetLambdaExitNodes(lambdaNode);
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto & memoryStateResult = GetMemoryStateRegionResult(lambdaNode);

  auto memoryNodeStatePairs = stateMap.GetStates(*subregion, memoryNodes);
  auto states = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);
  auto & mergedState = LambdaExitMemoryStateMergeOperation::Create(*subregion, states);
  memoryStateResult.divert_to(&mergedState);

  stateMap.PopRegion(*lambdaNode.subregion());
}

void
MemoryStateEncoder::EncodePhi(const phi::node & phiNode)
{
  EncodeRegion(*phiNode.subregion());
}

void
MemoryStateEncoder::EncodeDelta(const delta::node &)
{
  // Nothing needs to be done
}

void
MemoryStateEncoder::EncodeGamma(rvsdg::GammaNode & gammaNode)
{
  for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    Context_->GetRegionalizedStateMap().PushRegion(*gammaNode.subregion(n));

  EncodeGammaEntry(gammaNode);

  for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    EncodeRegion(*gammaNode.subregion(n));

  EncodeGammaExit(gammaNode);

  for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    Context_->GetRegionalizedStateMap().PopRegion(*gammaNode.subregion(n));
}

void
MemoryStateEncoder::EncodeGammaEntry(rvsdg::GammaNode & gammaNode)
{
  auto region = gammaNode.region();
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto memoryNodes = Context_->GetModRefSummary().GetGammaEntryNodes(gammaNode);

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
    std::vector<rvsdg::output *> states;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto subregion = gammaNode.subregion(n);

      auto & state = stateMap.GetState(*subregion, memoryNodeStatePair->MemoryNode())->State();
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

std::vector<rvsdg::output *>
MemoryStateEncoder::EncodeThetaEntry(rvsdg::ThetaNode & thetaNode)
{
  auto region = thetaNode.region();
  auto & stateMap = Context_->GetRegionalizedStateMap();
  auto & memoryNodes = Context_->GetModRefSummary().GetThetaEntryExitNodes(thetaNode);

  std::vector<rvsdg::output *> thetaStateOutputs;
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
    const std::vector<rvsdg::output *> & thetaStateOutputs)
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

LoadNode &
MemoryStateEncoder::ReplaceLoadNode(
    const LoadNode & loadNode,
    const std::vector<rvsdg::output *> & memoryStates)
{
  if (auto loadVolatileNode = dynamic_cast<const LoadVolatileNode *>(&loadNode))
  {
    auto & newLoadNode = loadVolatileNode->CopyWithNewMemoryStates(memoryStates);
    loadVolatileNode->GetLoadedValueOutput().divert_users(&newLoadNode.GetLoadedValueOutput());
    loadVolatileNode->GetIoStateOutput().divert_users(&newLoadNode.GetIoStateOutput());
    return newLoadNode;
  }
  else if (auto loadNonVolatileNode = dynamic_cast<const LoadNonVolatileNode *>(&loadNode))
  {
    auto & newLoadNode = loadNonVolatileNode->CopyWithNewMemoryStates(memoryStates);
    loadNode.GetLoadedValueOutput().divert_users(&newLoadNode.GetLoadedValueOutput());
    return newLoadNode;
  }
  else
  {
    JLM_UNREACHABLE("Unhandled load node type.");
  }
}

StoreNode &
MemoryStateEncoder::ReplaceStoreNode(
    const jlm::llvm::StoreNode & storeNode,
    const std::vector<rvsdg::output *> & memoryStates)
{
  if (auto storeVolatileNode = dynamic_cast<const StoreVolatileNode *>(&storeNode))
  {
    auto & newStoreNode = storeVolatileNode->CopyWithNewMemoryStates(memoryStates);
    storeVolatileNode->GetIoStateOutput().divert_users(&newStoreNode.GetIoStateOutput());
    return newStoreNode;
  }
  else if (auto storeNonVolatileNode = dynamic_cast<const StoreNonVolatileNode *>(&storeNode))
  {
    return storeNonVolatileNode->CopyWithNewMemoryStates(memoryStates);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled store node type.");
  }
}

std::vector<rvsdg::output *>
MemoryStateEncoder::ReplaceMemcpyNode(
    const rvsdg::SimpleNode & memcpyNode,
    const std::vector<rvsdg::output *> & memoryStates)
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
    if (is<MemoryStateType>(input->type()))
    {
      return true;
    }
  }

  for (size_t n = 0; n < simpleNode.noutputs(); n++)
  {
    auto output = simpleNode.output(n);
    if (is<MemoryStateType>(output->type()))
    {
      return true;
    }
  }

  return false;
}

}
