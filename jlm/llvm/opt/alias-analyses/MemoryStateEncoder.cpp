/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Operators.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

namespace jlm::aa {


/** \brief Statistics class for memory state encoder encoding
 *
 */
class EncodingStatistics final : public Statistics {
public:
  ~EncodingStatistics() override
  = default;

  explicit
  EncodingStatistics(jlm::filepath sourceFile)
    : Statistics(Statistics::Id::BasicEncoderEncoding)
    , NumNodesBefore_(0)
    , SourceFile_(std::move(sourceFile))
  {}

  void
  Start(const jive::graph & graph)
  {
    NumNodesBefore_ = jive::nnodes(graph.root());
    Timer_.start();
  }

  void
  Stop()
  {
    Timer_.stop();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return strfmt("BasicEncoderEncoding ",
                  SourceFile_.to_str(), " ",
                  "#RvsdgNodes:", NumNodesBefore_, " ",
                  "Time[ns]:", Timer_.ns());
  }

  static std::unique_ptr<EncodingStatistics>
  Create(const jlm::filepath & sourceFile)
  {
    return std::make_unique<EncodingStatistics>(sourceFile);
  }

private:
  jlm::timer Timer_;
  size_t NumNodesBefore_;
  jlm::filepath SourceFile_;
};

static jive::argument *
GetMemoryStateArgument(const lambda::node & lambda)
{
  auto subregion = lambda.subregion();

  /*
    FIXME: This function should be part of the lambda node.
  */
  for (size_t n = 0; n < subregion->narguments(); n++) {
    auto argument = subregion->argument(n);
    if (is<MemoryStateType>(argument->type()))
      return argument;
  }

  JLM_UNREACHABLE("This should have never happened!");
}

static jive::result *
GetMemoryStateResult(const lambda::node & lambda)
{
  auto subregion = lambda.subregion();

  /*
    FIXME: This function should be part of the lambda node.
  */
  for (size_t n = 0; n < subregion->nresults(); n++) {
    auto result = subregion->result(n);
    if (is<MemoryStateType>(result->type()))
      return result;
  }

  JLM_UNREACHABLE("This should have never happened!");
}

/** \brief A cache for points-to graph memory nodes of pointer outputs.
 *
 */
class MemoryNodeCache final {
private:
  explicit
  MemoryNodeCache(const MemoryNodeProvisioning & memoryNodeProvisioning)
    : MemoryNodeProvisioning_(memoryNodeProvisioning)
  {}

public:
  MemoryNodeCache(const MemoryNodeCache&) = delete;

  MemoryNodeCache(MemoryNodeCache&&) = delete;

  MemoryNodeCache &
  operator=(const MemoryNodeCache&) = delete;

  MemoryNodeCache &
  operator=(MemoryNodeCache&&) = delete;

  bool
  Contains(const jive::output & output) const noexcept
  {
    return MemoryNodeMap_.find(&output) != MemoryNodeMap_.end();
  }

  HashSet<const PointsToGraph::MemoryNode*>
  GetMemoryNodes(const jive::output & output)
  {
    JLM_ASSERT(is<PointerType>(output.type()));

    if (Contains(output))
      return MemoryNodeMap_[&output];

    auto memoryNodes = MemoryNodeProvisioning_.GetOutputNodes(output);

    /*
     * There is no need to cache the memory nodes, if the address is only once used.
     */
    if (output.nusers() <= 1)
      return memoryNodes;

    MemoryNodeMap_[&output] = std::move(memoryNodes);

    return MemoryNodeMap_[&output];
  }

  void
  ReplaceAddress(
    const jive::output & oldAddress,
    const jive::output & newAddress)
  {
    JLM_ASSERT(!Contains(oldAddress));
    JLM_ASSERT(!Contains(newAddress));

    MemoryNodeMap_[&newAddress] = MemoryNodeProvisioning_.GetOutputNodes(oldAddress);
  }

  static std::unique_ptr<MemoryNodeCache>
  Create(const MemoryNodeProvisioning & memoryNodeProvisioning)
  {
    return std::unique_ptr<MemoryNodeCache>(new MemoryNodeCache(memoryNodeProvisioning));
  }

private:
  const MemoryNodeProvisioning & MemoryNodeProvisioning_;
  std::unordered_map<const jive::output*, HashSet<const PointsToGraph::MemoryNode*>> MemoryNodeMap_;
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

    MemoryNodeStatePair(
      const PointsToGraph::MemoryNode & memoryNode,
      jive::output & state)
      : MemoryNode_(&memoryNode)
      , State_(&state)
    {
      JLM_ASSERT(is<MemoryStateType>(state.type()));
    }

  public:
    [[nodiscard]] const PointsToGraph::MemoryNode &
    MemoryNode() const noexcept
    {
      return *MemoryNode_;
    }

    [[nodiscard]] jive::output &
    State() const noexcept
    {
      return *State_;
    }

    void
    ReplaceState(jive::output & state) noexcept
    {
      JLM_ASSERT(State_->region() == state.region());
      JLM_ASSERT(is<MemoryStateType>(state.type()));

      State_ = &state;
    }

    static void
    ReplaceStates(
      const std::vector<MemoryNodeStatePair*> & memoryNodeStatePairs,
      const std::vector<jive::output*> & states)
    {
      JLM_ASSERT(memoryNodeStatePairs.size() == states.size());
      for (size_t n = 0; n < memoryNodeStatePairs.size(); n++)
        memoryNodeStatePairs[n]->ReplaceState(*states[n]);
    }

    static std::vector<jive::output*>
    States(const std::vector<MemoryNodeStatePair*> & memoryNodeStatePairs)
    {
      std::vector<jive::output*> states;
      for (auto & memoryNodeStatePair : memoryNodeStatePairs)
        states.push_back(memoryNodeStatePair->State_);

      return states;
    }

  private:
    const PointsToGraph::MemoryNode * MemoryNode_;
    jive::output * State_;
  };

  StateMap() = default;

  StateMap(const StateMap&) = delete;

  StateMap(StateMap&&) = delete;

  StateMap &
  operator=(const StateMap&) = delete;

  StateMap &
  operator=(StateMap&&) = delete;

  bool
  Contains(const PointsToGraph::MemoryNode & memoryNode) const noexcept
  {
    return states_.find(&memoryNode) != states_.end();
  }

  MemoryNodeStatePair*
  GetState(const PointsToGraph::MemoryNode & memoryNode) noexcept
  {
    JLM_ASSERT(Contains(memoryNode));
    return &states_.at(&memoryNode);
  }

  std::vector<MemoryNodeStatePair*>
  GetStates(const HashSet<const PointsToGraph::MemoryNode*> & memoryNodes)
  {
    std::vector<MemoryNodeStatePair*> memoryNodeStatePairs;
    for (auto & memoryNode : memoryNodes.Items())
      memoryNodeStatePairs.push_back(GetState(*memoryNode));

    return memoryNodeStatePairs;
  }

  void
  InsertState(
    const PointsToGraph::MemoryNode & memoryNode,
    jive::output & state)
  {
    JLM_ASSERT(!Contains(memoryNode));

    auto pair = std::make_pair<const PointsToGraph::MemoryNode*, MemoryNodeStatePair>(&memoryNode, {memoryNode, state});
    states_.insert(pair);
  }

  static std::unique_ptr<StateMap>
  Create()
  {
    return std::make_unique<StateMap>();
  }

private:
  std::unordered_map<const PointsToGraph::MemoryNode*, MemoryNodeStatePair> states_;
};

/** \brief Hash map for mapping Rvsdg regions to StateMap class instances.
*/
class RegionalizedStateMap final {
public:
  ~RegionalizedStateMap()
  {
    /*
     * Ensure that a PopRegion() was invoked for each invocation of a PushRegion().
     */
    JLM_ASSERT(StateMaps_.empty());
    JLM_ASSERT(MemoryNodeCacheMaps_.empty());
  }

  explicit
  RegionalizedStateMap(const MemoryNodeProvisioning & provisioning)
    : MemoryNodeProvisioning_(provisioning)
  {}

  RegionalizedStateMap(const RegionalizedStateMap&) = delete;

  RegionalizedStateMap(RegionalizedStateMap&&) = delete;

  RegionalizedStateMap &
  operator=(const RegionalizedStateMap&) = delete;

  RegionalizedStateMap &
  operator=(RegionalizedStateMap&&) = delete;

  void
  InsertState(
    const PointsToGraph::MemoryNode & memoryNode,
    jive::output & state)
  {
    GetStateMap(*state.region()).InsertState(memoryNode, state);
  }

  void
  ReplaceAddress(
    const jive::output & oldAddress,
    const jive::output & newAddress)
  {
    GetMemoryNodeCache(*oldAddress.region()).ReplaceAddress(oldAddress, newAddress);
  }

  std::vector<StateMap::MemoryNodeStatePair*>
  GetStates(const jive::output & output) noexcept
  {
    auto memoryNodes = GetMemoryNodes(output);
    return memoryNodes.Size() == 0
           ? std::vector<StateMap::MemoryNodeStatePair*>()
           : GetStates(*output.region(), memoryNodes);
  }

  std::vector<StateMap::MemoryNodeStatePair*>
  GetStates(
    const jive::region & region,
    const HashSet<const PointsToGraph::MemoryNode*> & memoryNodes)
  {
    return GetStateMap(region).GetStates(memoryNodes);
  }

  StateMap::MemoryNodeStatePair*
  GetState(
    const jive::region & region,
    const PointsToGraph::MemoryNode & memoryNode)
  {
    return GetStateMap(region).GetState(memoryNode);
  }

  HashSet<const PointsToGraph::MemoryNode*>
  GetMemoryNodes(const jive::output & output)
  {
    auto & memoryNodeCache = GetMemoryNodeCache(*output.region());
    return memoryNodeCache.GetMemoryNodes(output);
  }

  void
  PushRegion(const jive::region & region)
  {
    JLM_ASSERT(StateMaps_.find(&region) == StateMaps_.end());
    JLM_ASSERT(MemoryNodeCacheMaps_.find(&region) == MemoryNodeCacheMaps_.end());

    StateMaps_[&region] = StateMap::Create();
    MemoryNodeCacheMaps_[&region] = MemoryNodeCache::Create(MemoryNodeProvisioning_);
  }

  void
  PopRegion(const jive::region & region)
  {
    JLM_ASSERT(StateMaps_.find(&region) != StateMaps_.end());
    JLM_ASSERT(MemoryNodeCacheMaps_.find(&region) != MemoryNodeCacheMaps_.end());

    StateMaps_.erase(&region);
    MemoryNodeCacheMaps_.erase(&region);
  }

private:
  StateMap &
  GetStateMap(const jive::region & region) const noexcept
  {
    JLM_ASSERT(StateMaps_.find(&region) != StateMaps_.end());
    return *StateMaps_.at(&region);
  }

  MemoryNodeCache &
  GetMemoryNodeCache(const jive::region & region) const noexcept
  {
    JLM_ASSERT(MemoryNodeCacheMaps_.find(&region) != MemoryNodeCacheMaps_.end());
    return *MemoryNodeCacheMaps_.at(&region);
  }

  std::unordered_map<const jive::region*, std::unique_ptr<StateMap>> StateMaps_;
  std::unordered_map<const jive::region*, std::unique_ptr<MemoryNodeCache>> MemoryNodeCacheMaps_;

  const MemoryNodeProvisioning & MemoryNodeProvisioning_;
};

/** \brief Context for the memory state encoder
*/
class MemoryStateEncoder::Context final {
public:
  explicit
  Context(const MemoryNodeProvisioning & provisioning)
    : RegionalizedStateMap_(provisioning)
    , Provisioning_(provisioning)
  {}

  Context(const Context&) = delete;

  Context(Context&&) = delete;

  Context&
  operator=(const Context&) = delete;

  Context&
  operator=(Context&&) = delete;

  RegionalizedStateMap &
  GetRegionalizedStateMap() noexcept
  {
    return RegionalizedStateMap_;
  }

  const MemoryNodeProvisioning &
  GetMemoryNodeProvisioning() const noexcept
  {
    return Provisioning_;
  }

  static std::unique_ptr<MemoryStateEncoder::Context>
  Create(const MemoryNodeProvisioning & provisioning)
  {
    return std::make_unique<Context>(provisioning);
  }

private:
  RegionalizedStateMap RegionalizedStateMap_;
  const MemoryNodeProvisioning & Provisioning_;
};

MemoryStateEncoder::~MemoryStateEncoder() noexcept
= default;

MemoryStateEncoder::MemoryStateEncoder()
= default;

void
MemoryStateEncoder::Encode(
  RvsdgModule & rvsdgModule,
  const MemoryNodeProvisioning & provisioning,
  StatisticsCollector & statisticsCollector)
{
  Context_ = Context::Create(provisioning);
  auto statistics = EncodingStatistics::Create(rvsdgModule.SourceFileName());

  statistics->Start(rvsdgModule.Rvsdg());
  EncodeRegion(*rvsdgModule.Rvsdg().root());
  statistics->Stop();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  /*
   * Remove all nodes that became dead throughout the encoding.
   */
  jlm::DeadNodeElimination deadNodeElimination;
  deadNodeElimination.run(rvsdgModule, statisticsCollector);
}

void
MemoryStateEncoder::EncodeRegion(jive::region & region)
{
  using namespace jive;

  jive::topdown_traverser traverser(&region);
  for (auto & node : traverser) {
    if (auto simpleNode = dynamic_cast<const simple_node*>(node)) {
      EncodeSimpleNode(*simpleNode);
      continue;
    }

    auto structuralNode = AssertedCast<structural_node>(node);
    EncodeStructuralNode(*structuralNode);
  }
}

void
MemoryStateEncoder::EncodeStructuralNode(jive::structural_node & structuralNode)
{
  auto encodeLambda = [](auto & be, auto & n){ be.EncodeLambda(*AssertedCast<lambda::node>(&n));    };
  auto encodeDelta  = [](auto & be, auto & n){ be.EncodeDelta(*AssertedCast<delta::node>(&n));      };
  auto encodePhi    = [](auto & be, auto & n){ be.EncodePhi(*AssertedCast<phi::node>(&n));          };
  auto encodeGamma  = [](auto & be, auto & n){ be.EncodeGamma(*AssertedCast<jive::gamma_node>(&n)); };
  auto encodeTheta  = [](auto & be, auto & n){ be.EncodeTheta(*AssertedCast<jive::theta_node>(&n)); };

  static std::unordered_map<
    std::type_index,
    std::function<void(MemoryStateEncoder&, jive::structural_node&)>
  > nodes
    ({
         {typeid(lambda::operation), encodeLambda }
       , {typeid(delta::operation),  encodeDelta  }
       , {typeid(phi::operation),    encodePhi    }
       , {typeid(jive::gamma_op),    encodeGamma  }
       , {typeid(jive::theta_op),    encodeTheta  }
     });

  auto & operation = structuralNode.operation();
  JLM_ASSERT(nodes.find(typeid(operation)) != nodes.end());
  nodes[typeid(operation)](*this, structuralNode);
}

void
MemoryStateEncoder::EncodeSimpleNode(const jive::simple_node & be)
{
  auto EncodeAlloca = [](auto & be, auto & node){ be.EncodeAlloca(node); };
  auto EncodeMalloc = [](auto & be, auto & node){ be.EncodeMalloc(node); };
  auto EncodeCall   = [](auto & be, auto & node){ be.EncodeCall(*AssertedCast<const CallNode>(&node)); };
  auto EncodeLoad   = [](auto & be, auto & node){ be.EncodeLoad(*AssertedCast<const LoadNode>(&node)); };
  auto EncodeStore  = [](auto & be, auto & node){ be.EncodeStore(*AssertedCast<const StoreNode>(&node)); };
  auto EncodeFree   = [](auto & be, auto & node){ be.EncodeFree(node); };
  auto EncodeMemcpy = [](auto & be, auto & node){ be.EncodeMemcpy(node); };

  static std::unordered_map<
    std::type_index
    , std::function<void(MemoryStateEncoder&, const jive::simple_node&)>
  > nodes
    ({
       {typeid(alloca_op),      EncodeAlloca},
       {typeid(malloc_op),      EncodeMalloc},
       {typeid(LoadOperation),  EncodeLoad},
       {typeid(StoreOperation), EncodeStore},
       {typeid(CallOperation),  EncodeCall},
       {typeid(free_op),        EncodeFree},
       {typeid(Memcpy),         EncodeMemcpy}
     });

  auto & operation = be.operation();
  if (nodes.find(typeid(operation)) == nodes.end())
    return;

  nodes[typeid(operation)](*this, be);
}

void
MemoryStateEncoder::EncodeAlloca(const jive::simple_node & allocaNode)
{
  JLM_ASSERT(is<alloca_op>(&allocaNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto & allocaMemoryNode = Context_->GetMemoryNodeProvisioning().GetPointsToGraph().GetAllocaNode(allocaNode);
  auto memoryNodeStatePair = stateMap.GetState(*allocaNode.region(), allocaMemoryNode);
  memoryNodeStatePair->ReplaceState(*allocaNode.output(1));
}

void
MemoryStateEncoder::EncodeMalloc(const jive::simple_node & mallocNode)
{
  JLM_ASSERT(is<malloc_op>(&mallocNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto & mallocMemoryNode = Context_->GetMemoryNodeProvisioning().GetPointsToGraph().GetMallocNode(mallocNode);

  /**
   * We use a static heap model. This means that multiple invocations of an malloc
   * at runtime can refer to the same abstract memory location. We therefore need to
   * merge the previous and the current state to ensure that the previous state
   * is not just simply replaced and therefore "lost".
   */
  auto memoryNodeStatePair = stateMap.GetState(*mallocNode.region(), mallocMemoryNode);
  auto mallocState = mallocNode.output(1);
  auto mergedState = MemStateMergeOperator::Create({mallocState, &memoryNodeStatePair->State()});
  memoryNodeStatePair->ReplaceState(*mergedState);
}

void
MemoryStateEncoder::EncodeLoad(const LoadNode & loadNode)
{
  auto & loadOperation = loadNode.GetOperation();
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = loadNode.GetAddressInput()->origin();
  auto memoryNodeStatePairs = stateMap.GetStates(*address);
  auto oldResult = loadNode.GetValueOutput();
  auto inStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto outputs = LoadNode::Create(
    address,
    inStates,
    loadOperation.GetLoadedType(),
    loadOperation.GetAlignment());
  oldResult->divert_users(outputs[0]);

  StateMap::MemoryNodeStatePair::ReplaceStates(
    memoryNodeStatePairs,
    {std::next(outputs.begin()), outputs.end()});

  if (is<PointerType>(oldResult->type()))
    stateMap.ReplaceAddress(*oldResult, *outputs[0]);
}

void
MemoryStateEncoder::EncodeStore(const StoreNode & storeNode)
{
  auto & storeOperation = storeNode.GetOperation();
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = storeNode.GetAddressInput()->origin();
  auto value = storeNode.GetValueInput()->origin();
  auto memoryNodeStatePairs = stateMap.GetStates(*address);
  auto inStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto outStates = StoreNode::Create(
    address,
    value,
    inStates,
    storeOperation.GetAlignment());

  StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, outStates);
}

void
MemoryStateEncoder::EncodeFree(const jive::simple_node & freeNode)
{
  JLM_ASSERT(is<free_op>(&freeNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = freeNode.input(0)->origin();
  auto iostate = freeNode.input(freeNode.ninputs() - 1)->origin();
  auto memoryNodeStatePairs = stateMap.GetStates(*address);
  auto inStates = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);

  auto outputs = free_op::create(address, inStates, iostate);
  freeNode.output(freeNode.noutputs() - 1)->divert_users(outputs.back());

  StateMap::MemoryNodeStatePair::ReplaceStates(
    memoryNodeStatePairs,
    {outputs.begin(), std::prev(outputs.end())});
}

void
MemoryStateEncoder::EncodeCall(const CallNode & callNode)
{
  auto EncodeEntry = [this](const CallNode & callNode)
  {
    auto region = callNode.region();
    auto & memoryNodes = Context_->GetMemoryNodeProvisioning().GetCallEntryNodes(callNode);

    auto memoryNodeStatePairs = Context_->GetRegionalizedStateMap().GetStates(*region, memoryNodes);
    auto states = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);
    auto state = CallEntryMemStateOperator::Create(region, states);
    callNode.GetMemoryStateInput()->divert_to(state);
  };

  auto EncodeExit = [this](const CallNode & callNode)
  {
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto & memoryNodes = Context_->GetMemoryNodeProvisioning().GetCallExitNodes(callNode);

    auto states = CallExitMemStateOperator::Create(callNode.GetMemoryStateOutput(), memoryNodes.Size());
    auto memoryNodeStatePairs = stateMap.GetStates(*callNode.region(), memoryNodes);
    StateMap::MemoryNodeStatePair::ReplaceStates(memoryNodeStatePairs, states);
  };

  EncodeEntry(callNode);
  EncodeExit(callNode);
}

void
MemoryStateEncoder::EncodeMemcpy(const jive::simple_node & memcpyNode)
{
  JLM_ASSERT(is<Memcpy>(&memcpyNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto destination = memcpyNode.input(0)->origin();
  auto source = memcpyNode.input(1)->origin();
  auto length = memcpyNode.input(2)->origin();
  auto isVolatile = memcpyNode.input(3)->origin();

  auto destMemoryNodeStatePairs = stateMap.GetStates(*destination);
  auto srcMemoryNodeStatePairs = stateMap.GetStates(*source);

  auto inStates = StateMap::MemoryNodeStatePair::States(destMemoryNodeStatePairs);
  auto srcStates = StateMap::MemoryNodeStatePair::States(srcMemoryNodeStatePairs);
  inStates.insert(inStates.end(), srcStates.begin(), srcStates.end());

  auto outStates = Memcpy::create(destination, source, length, isVolatile, inStates);

  auto end = std::next(outStates.begin(), (ssize_t)destMemoryNodeStatePairs.size());
  StateMap::MemoryNodeStatePair::ReplaceStates(
    destMemoryNodeStatePairs,
    {outStates.begin(), std::next(outStates.begin(), (ssize_t)destMemoryNodeStatePairs.size())});

  JLM_ASSERT((size_t)std::distance(end, outStates.end()) == srcMemoryNodeStatePairs.size());
  StateMap::MemoryNodeStatePair::ReplaceStates(
    srcMemoryNodeStatePairs, {end, outStates.end()});
}

void
MemoryStateEncoder::EncodeLambda(const lambda::node & lambda)
{
  auto EncodeEntry = [this](const lambda::node & lambda)
  {
    auto memoryStateArgument = GetMemoryStateArgument(lambda);
    JLM_ASSERT(memoryStateArgument->nusers() == 1);
    auto memoryStateArgumentUser = *memoryStateArgument->begin();

    auto & memoryNodes = Context_->GetMemoryNodeProvisioning().GetLambdaEntryNodes(lambda);
    auto & stateMap = Context_->GetRegionalizedStateMap();

    stateMap.PushRegion(*lambda.subregion());

    auto states = LambdaEntryMemStateOperator::Create(memoryStateArgument, memoryNodes.Size());

    size_t n = 0;
    for (auto & memoryNode : memoryNodes.Items())
      stateMap.InsertState(*memoryNode, *states[n++]);

    if (!states.empty())
    {
      /*
       * This additional MemStateMergeOperator node makes all other nodes in the function that consume the memory state
       * dependent on this node and therefore transitively on the LambdaEntryMemStateOperator. This ensures that the
       * LambdaEntryMemStateOperator is always visited before all other memory state consuming nodes:
       *
       * ... := LAMBDA[f]
       *   [..., a1, ...]
       *     o1, ..., ox := LambdaEntryMemStateOperator a1
       *     oy = MemStateMergeOperator o1, ..., ox
       *     ....
       *
       * No other memory state consuming node aside from the LambdaEntryMemStateOperator should now consume a1.
       */
      auto state = MemStateMergeOperator::Create(states);
      memoryStateArgumentUser->divert_to(state);
    }
  };

  auto EncodeExit = [this](const lambda::node & lambda)
  {
    auto subregion = lambda.subregion();
    auto & memoryNodes = Context_->GetMemoryNodeProvisioning().GetLambdaExitNodes(lambda);
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto memoryStateResult = GetMemoryStateResult(lambda);

    auto memoryNodeStatePairs = stateMap.GetStates(*subregion, memoryNodes);
    auto states = StateMap::MemoryNodeStatePair::States(memoryNodeStatePairs);
    auto mergedState = LambdaExitMemStateOperator::Create(subregion, states);
    memoryStateResult->divert_to(mergedState);

    stateMap.PopRegion(*lambda.subregion());
  };

  EncodeEntry(lambda);
  EncodeRegion(*lambda.subregion());
  EncodeExit(lambda);
}

void
MemoryStateEncoder::EncodePhi(const phi::node & phi)
{
  EncodeRegion(*phi.subregion());
}

void
MemoryStateEncoder::EncodeDelta(const delta::node & delta)
{
  /* Nothing needs to be done */
}

void
MemoryStateEncoder::EncodeGamma(jive::gamma_node & gamma)
{
  auto EncodeEntry = [this](jive::gamma_node & gamma)
  {
    auto region = gamma.region();
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto memoryNodes = Context_->GetMemoryNodeProvisioning().GetGammaEntryNodes(gamma);

    auto memoryNodeStatePairs = stateMap.GetStates(*region, memoryNodes);
    for (auto & memoryNodeStatePair : memoryNodeStatePairs) {
      auto gammaInput = gamma.add_entryvar(&memoryNodeStatePair->State());
      for (auto & argument : *gammaInput)
        stateMap.InsertState(memoryNodeStatePair->MemoryNode(), argument);
    }
  };

  auto EncodeExit = [this](jive::gamma_node & gamma)
  {
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto memoryNodes = Context_->GetMemoryNodeProvisioning().GetGammaExitNodes(gamma);
    auto memoryNodeStatePairs = stateMap.GetStates(*gamma.region(), memoryNodes);

    for (auto & memoryNodeStatePair : memoryNodeStatePairs)
    {
      std::vector<jive::output*> states;
      for (size_t n = 0; n < gamma.nsubregions(); n++) {
        auto subregion = gamma.subregion(n);

        auto & state = stateMap.GetState(*subregion, memoryNodeStatePair->MemoryNode())->State();
        states.push_back(&state);
      }

      auto state = gamma.add_exitvar(states);
      memoryNodeStatePair->ReplaceState(*state);
    }
  };

  for (size_t n = 0; n < gamma.nsubregions(); n++)
    Context_->GetRegionalizedStateMap().PushRegion(*gamma.subregion(n));

  EncodeEntry(gamma);

  for (size_t n = 0; n < gamma.nsubregions(); n++)
    EncodeRegion(*gamma.subregion(n));

  EncodeExit(gamma);

  for (size_t n = 0; n < gamma.nsubregions(); n++)
    Context_->GetRegionalizedStateMap().PopRegion(*gamma.subregion(n));
}

void
MemoryStateEncoder::EncodeTheta(jive::theta_node & theta)
{
  auto EncodeEntry = [this](jive::theta_node & theta)
  {
    auto region = theta.region();
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto & memoryNodes = Context_->GetMemoryNodeProvisioning().GetThetaEntryExitNodes(theta);

    std::vector<jive::theta_output*> thetaStateOutputs;
    auto memoryNodeStatePairs = stateMap.GetStates(*region, memoryNodes);
    for (auto & memoryNodeStatePair : memoryNodeStatePairs)
    {
      auto thetaStateOutput = theta.add_loopvar(&memoryNodeStatePair->State());
      stateMap.InsertState(memoryNodeStatePair->MemoryNode(), *thetaStateOutput->argument());
      thetaStateOutputs.push_back(thetaStateOutput);
    }

    return thetaStateOutputs;
  };

  auto EncodeExit = [this](
    jive::theta_node & theta,
    const std::vector<jive::theta_output*> & thetaStateOutputs)
  {
    auto subregion = theta.subregion();
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto & memoryNodes = Context_->GetMemoryNodeProvisioning().GetThetaEntryExitNodes(theta);
    auto memoryNodeStatePairs = stateMap.GetStates(*theta.region(), memoryNodes);

    JLM_ASSERT(memoryNodeStatePairs.size() == thetaStateOutputs.size());
    for (size_t n = 0; n < thetaStateOutputs.size(); n++)
    {
      auto thetaStateOutput = thetaStateOutputs[n];
      auto & memoryNodeStatePair = memoryNodeStatePairs[n];
      auto & memoryNode = memoryNodeStatePair->MemoryNode();
      JLM_ASSERT(thetaStateOutput->input()->origin() == &memoryNodeStatePair->State());

      auto & subregionState = stateMap.GetState(*subregion, memoryNode)->State();
      thetaStateOutput->result()->divert_to(&subregionState);
      memoryNodeStatePair->ReplaceState(*thetaStateOutput);
    }
  };

  Context_->GetRegionalizedStateMap().PushRegion(*theta.subregion());

  auto thetaStateOutputs = EncodeEntry(theta);
  EncodeRegion(*theta.subregion());
  EncodeExit(theta, thetaStateOutputs);

  Context_->GetRegionalizedStateMap().PopRegion(*theta.subregion());
}

}