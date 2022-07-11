/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/opt/alias-analyses/BasicEncoder.hpp>
#include <jlm/opt/alias-analyses/BasicMemoryNodeProvider.hpp>
#include <jlm/opt/alias-analyses/Operators.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/DeadNodeElimination.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/strfmt.hpp>
#include <jlm/util/time.hpp>

#include <jive/rvsdg/traverser.hpp>

namespace jlm::aa {

/** \brief Statistics class for basic encoder encoding
 *
 */
class EncodingStatistics final : public Statistics {
public:
  ~EncodingStatistics() override
  = default;

  explicit
  EncodingStatistics(jlm::filepath sourceFile)
  : Statistics(StatisticsDescriptor::StatisticsId::BasicEncoderEncoding)
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
  MemoryNodeCache(const MemoryNodeProvider & memoryNodeProvider)
  : MemoryNodeProvider_(memoryNodeProvider)
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

  std::vector<const PointsToGraph::MemoryNode*>
  GetMemoryNodes(const jive::output & output)
  {
    JLM_ASSERT(is<PointerType>(output.type()));

    if (Contains(output))
      return MemoryNodeMap_[&output];

    auto memoryNodes = MemoryNodeProvider_.GetOutputNodes(output);

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

    MemoryNodeMap_[&newAddress] = MemoryNodeProvider_.GetOutputNodes(oldAddress);
  }

  static std::unique_ptr<MemoryNodeCache>
  Create(const MemoryNodeProvider & memoryNodeProvider)
  {
    return std::unique_ptr<MemoryNodeCache>(new MemoryNodeCache(memoryNodeProvider));
  }

private:
  const MemoryNodeProvider & MemoryNodeProvider_;
  std::unordered_map<const jive::output*, std::vector<const PointsToGraph::MemoryNode*>> MemoryNodeMap_;
};

/** \brief Hash map for mapping points-to graph memory nodes to RVSDG memory states.
*/
class StateMap final {
public:
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

  jive::output *
  GetState(const PointsToGraph::MemoryNode & memoryNode) const noexcept
  {
    JLM_ASSERT(Contains(memoryNode));
    return states_.at(&memoryNode);
  }

  std::vector<jive::output*>
  GetStates(const std::vector<const PointsToGraph::MemoryNode*> & memoryNodes) const
  {
    std::vector<jive::output*> states;
    states.reserve(memoryNodes.size());
    for (auto & memoryNode : memoryNodes)
      states.push_back(GetState(*memoryNode));

    return states;
  }

  void
  InsertState(
    const PointsToGraph::MemoryNode & memoryNode,
    jive::output & state)
  {
    JLM_ASSERT(!Contains(memoryNode));
    JLM_ASSERT(is<MemoryStateType>(state.type()));

    states_[&memoryNode] = &state;
  }

  void
  InsertStates(
    const std::vector<const PointsToGraph::MemoryNode*> & memoryNodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(memoryNodes.size() == states.size());

    for (size_t n = 0; n < memoryNodes.size(); n++)
      InsertState(*memoryNodes[n], *states[n]);
  }

  void
  ReplaceState(
    const PointsToGraph::MemoryNode & memoryNode,
    jive::output & state)
  {
    JLM_ASSERT(Contains(memoryNode));
    JLM_ASSERT(is<MemoryStateType>(state.type()));

    states_[&memoryNode] = &state;
  }

  void
  ReplaceStates(
    const std::vector<const PointsToGraph::MemoryNode*> & memoryNodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(memoryNodes.size() == states.size());
    for (size_t n = 0; n < memoryNodes.size(); n++)
      ReplaceState(*memoryNodes[n], *states[n]);
  }

  static std::unique_ptr<StateMap>
  Create()
  {
    return std::make_unique<StateMap>();
  }

private:
  std::unordered_map<const PointsToGraph::MemoryNode*, jive::output*> states_;
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
  RegionalizedStateMap(const MemoryNodeProvider & memoryNodeProvider)
  : MemoryNodeProvider_(memoryNodeProvider)
  {}

  RegionalizedStateMap(const RegionalizedStateMap&) = delete;

  RegionalizedStateMap(RegionalizedStateMap&&) = delete;

  RegionalizedStateMap &
  operator=(const RegionalizedStateMap&) = delete;

  RegionalizedStateMap &
  operator=(RegionalizedStateMap&&) = delete;

  bool
  Contains(
    const jive::region & region,
    const PointsToGraph::MemoryNode & memoryNode)
  {
    return GetStateMap(region).Contains(memoryNode);
  }

  void
  InsertState(
    const PointsToGraph::MemoryNode & memoryNode,
    jive::output & state)
  {
    GetStateMap(*state.region()).InsertState(memoryNode, state);
  }

  void
  InsertStates(
    const std::vector<const PointsToGraph::MemoryNode*> & memoryNodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(memoryNodes.size() == states.size());
    JLM_ASSERT(!memoryNodes.empty());

    GetStateMap(*states[0]->region()).InsertStates(memoryNodes, states);
  }

  void
  ReplaceAddress(
    const jive::output & oldAddress,
    const jive::output & newAddress)
  {
    GetMemoryNodeCache(*oldAddress.region()).ReplaceAddress(oldAddress, newAddress);
  }

  void
  ReplaceStates(
    const jive::output & output,
    const std::vector<jive::output*> & states)
  {
    auto memoryNodes = GetMemoryNodes(output);
    GetStateMap(*output.region()).ReplaceStates(memoryNodes, states);
  }

  void
  ReplaceState(
    const PointsToGraph::MemoryNode & memoryNode,
    jive::output & state)
  {
    GetStateMap(*state.region()).ReplaceState(memoryNode, state);
  }

  void
  ReplaceStates(
    const std::vector<const PointsToGraph::MemoryNode*> & memoryNodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(memoryNodes.size() == states.size());
    JLM_ASSERT(!memoryNodes.empty());

    GetStateMap(*states[0]->region()).ReplaceStates(memoryNodes, states);
  }

  std::vector<jive::output*>
  GetStates(const jive::output & output) noexcept
  {
    auto memoryNodes = GetMemoryNodes(output);
    return memoryNodes.empty()
           ? std::vector<jive::output*>()
           : GetStates(*output.region(), memoryNodes);
  }

  std::vector<jive::output*>
  GetStates(
    const jive::region & region,
    const std::vector<const PointsToGraph::MemoryNode*> & memoryNodes)
  {
    return GetStateMap(region).GetStates(memoryNodes);
  }

  jive::output *
  GetState(
    const jive::region & region,
    const PointsToGraph::MemoryNode & memoryNode)
  {
    return GetStates(region, {&memoryNode})[0];
  }

  std::vector<const PointsToGraph::MemoryNode*>
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
    MemoryNodeCacheMaps_[&region] = MemoryNodeCache::Create(MemoryNodeProvider_);
  }

  void
  PopRegion(const jive::region & region)
  {
    JLM_ASSERT(StateMaps_.find(&region) != StateMaps_.end());
    JLM_ASSERT(MemoryNodeCacheMaps_.find(&region) != MemoryNodeCacheMaps_.end());

    StateMaps_.erase(&region);
    MemoryNodeCacheMaps_.erase(&region);
  }

  static std::unique_ptr<BasicEncoder::Context>
  Create(const MemoryNodeProvider & memoryNodeProvider)
  {
    return std::make_unique<BasicEncoder::Context>(memoryNodeProvider);
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

  const MemoryNodeProvider & MemoryNodeProvider_;
};

/** \brief Context for the basic encoder
*/
class BasicEncoder::Context final {
public:
  explicit
  Context(const MemoryNodeProvider & memoryNodeProvider)
    : RegionalizedStateMap_(memoryNodeProvider)
    , MemoryNodeProvider_(memoryNodeProvider)
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

  const MemoryNodeProvider &
  GetMemoryNodeProvider() const noexcept
  {
    return MemoryNodeProvider_;
  }

  static std::unique_ptr<BasicEncoder::Context>
  Create(const MemoryNodeProvider & memoryNodeProvider)
  {
    return std::make_unique<Context>(memoryNodeProvider);
  }

private:
  RegionalizedStateMap RegionalizedStateMap_;
  const MemoryNodeProvider & MemoryNodeProvider_;
};

BasicEncoder::~BasicEncoder()
= default;

BasicEncoder::BasicEncoder(PointsToGraph & pointsToGraph)
  : PointsToGraph_(pointsToGraph)
{}

void
BasicEncoder::Encode(
  PointsToGraph & pointsToGraph,
  RvsdgModule & rvsdgModule,
  const StatisticsDescriptor & statisticsDescriptor)
{
  BasicEncoder encoder(pointsToGraph);
  encoder.Encode(rvsdgModule, statisticsDescriptor);
}

void
BasicEncoder::Encode(
  RvsdgModule & rvsdgModule,
  const StatisticsDescriptor & statisticsDescriptor)
{
  jlm::aa::BasicMemoryNodeProvider basicMemoryNodeProvider(GetPointsToGraph());
  Context_ = Context::Create(basicMemoryNodeProvider);

  EncodingStatistics encodingStatistics(rvsdgModule.SourceFileName());
  encodingStatistics.Start(rvsdgModule.Rvsdg());
  EncodeRegion(*rvsdgModule.Rvsdg().root());
  encodingStatistics.Stop();
  statisticsDescriptor.PrintStatistics(encodingStatistics);

  /*
   * Remove all nodes that became dead throughout the encoding.
   */
  jlm::DeadNodeElimination deadNodeElimination;
  deadNodeElimination.run(rvsdgModule, statisticsDescriptor);
}

void
BasicEncoder::EncodeRegion(jive::region & region)
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
BasicEncoder::EncodeStructuralNode(jive::structural_node & structuralNode)
{
  auto encodeLambda = [](auto & be, auto & n){ be.EncodeLambda(*AssertedCast<lambda::node>(&n));    };
  auto encodeDelta  = [](auto & be, auto & n){ be.EncodeDelta(*AssertedCast<delta::node>(&n));      };
  auto encodePhi    = [](auto & be, auto & n){ be.EncodePhi(*AssertedCast<phi::node>(&n));          };
  auto encodeGamma  = [](auto & be, auto & n){ be.EncodeGamma(*AssertedCast<jive::gamma_node>(&n)); };
  auto encodeTheta  = [](auto & be, auto & n){ be.EncodeTheta(*AssertedCast<jive::theta_node>(&n)); };

  static std::unordered_map<
    std::type_index,
    std::function<void(BasicEncoder&, jive::structural_node&)>
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
BasicEncoder::EncodeSimpleNode(const jive::simple_node & be)
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
    , std::function<void(BasicEncoder&, const jive::simple_node&)>
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
BasicEncoder::EncodeAlloca(const jive::simple_node & allocaNode)
{
  JLM_ASSERT(is<alloca_op>(&allocaNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto & allocaMemoryNode = GetPointsToGraph().GetAllocaNode(allocaNode);
  stateMap.ReplaceState(allocaMemoryNode, *allocaNode.output(1));
}

void
BasicEncoder::EncodeMalloc(const jive::simple_node & mallocNode)
{
  JLM_ASSERT(is<malloc_op>(&mallocNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto & mallocMemoryNode = GetPointsToGraph().GetMallocNode(mallocNode);

  /**
   * We use a static heap model. This means that multiple invocations of an malloc
   * at runtime can refer to the same abstract memory location. We therefore need to
   * merge the previous and the current state to ensure that the previous state
   * is not just simply replaced and therefore "lost".
   */
  auto routedState = stateMap.GetState(*mallocNode.region(), mallocMemoryNode);
  auto mallocState = mallocNode.output(1);
  auto mergedState = MemStateMergeOperator::Create({mallocState, routedState});

  stateMap.ReplaceState(mallocMemoryNode, *mergedState);
}

void
BasicEncoder::EncodeLoad(const LoadNode & loadNode)
{
  auto & loadOperation = loadNode.GetOperation();
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = loadNode.GetAddressInput()->origin();
  auto instates = stateMap.GetStates(*address);
  auto oldResult = loadNode.GetValueOutput();

  auto outputs = LoadNode::Create(address, instates, loadOperation.GetAlignment());
  oldResult->divert_users(outputs[0]);

  stateMap.ReplaceStates(*address, {std::next(outputs.begin()), outputs.end()});

  if (is<PointerType>(oldResult->type()))
    stateMap.ReplaceAddress(*oldResult, *outputs[0]);
}

void
BasicEncoder::EncodeStore(const StoreNode & storeNode)
{
  auto & storeOperation = storeNode.GetOperation();
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = storeNode.GetAddressInput()->origin();
  auto value = storeNode.GetValueInput()->origin();
  auto inStates = stateMap.GetStates(*address);

  auto outStates = StoreNode::Create(
    address,
    value,
    inStates,
    storeOperation.GetAlignment());

  stateMap.ReplaceStates(*address, outStates);
}

void
BasicEncoder::EncodeFree(const jive::simple_node & freeNode)
{
  JLM_ASSERT(is<free_op>(&freeNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto address = freeNode.input(0)->origin();
  auto iostate = freeNode.input(freeNode.ninputs() - 1)->origin();
  auto instates = stateMap.GetStates(*address);

  auto outputs = free_op::create(address, instates, iostate);
  freeNode.output(freeNode.noutputs() - 1)->divert_users(outputs.back());

  stateMap.ReplaceStates(*address, {outputs.begin(), std::prev(outputs.end())});
}

void
BasicEncoder::EncodeCall(const CallNode & callNode)
{
  auto EncodeEntry = [this](const CallNode & callNode)
  {
    auto region = callNode.region();
    auto & memoryNodes = Context_->GetMemoryNodeProvider().GetCallEntryNodes(callNode);

    auto states = Context_->GetRegionalizedStateMap().GetStates(*region, memoryNodes);
    auto state = CallEntryMemStateOperator::Create(region, states);
    callNode.GetMemoryStateInput()->divert_to(state);
  };

  auto EncodeExit = [this](const CallNode & callNode)
  {
    auto & memoryNodes = Context_->GetMemoryNodeProvider().GetCallExitNodes(callNode);

    auto states = CallExitMemStateOperator::Create(callNode.GetMemoryStateOutput(), memoryNodes.size());
    Context_->GetRegionalizedStateMap().ReplaceStates(memoryNodes, states);
  };

  EncodeEntry(callNode);
  EncodeExit(callNode);
}

void
BasicEncoder::EncodeMemcpy(const jive::simple_node & memcpyNode)
{
  JLM_ASSERT(is<Memcpy>(&memcpyNode));
  auto & stateMap = Context_->GetRegionalizedStateMap();

  auto destination = memcpyNode.input(0)->origin();
  auto source = memcpyNode.input(1)->origin();
  auto length = memcpyNode.input(2)->origin();
  auto isVolatile = memcpyNode.input(3)->origin();

  auto destinationStates = stateMap.GetStates(*destination);
  auto sourceStates = stateMap.GetStates(*source);

  auto inStates = destinationStates;
  inStates.insert(inStates.end(), sourceStates.begin(), sourceStates.end());

  auto outStates = Memcpy::create(destination, source, length, isVolatile, inStates);

  auto begin = outStates.begin();
  auto end = std::next(outStates.begin(), (ssize_t)destinationStates.size());
  stateMap.ReplaceStates(*destination, {begin, end});

  JLM_ASSERT((size_t)std::distance(end, outStates.end()) == sourceStates.size());
  stateMap.ReplaceStates(*source, {end, outStates.end()});
}

void
BasicEncoder::EncodeLambda(const lambda::node & lambda)
{
  auto EncodeEntry = [this](const lambda::node & lambda)
  {
    auto memoryStateArgument = GetMemoryStateArgument(lambda);
    auto & memoryNodes = Context_->GetMemoryNodeProvider().GetLambdaEntryNodes(lambda);
    auto & stateMap = Context_->GetRegionalizedStateMap();

    stateMap.PushRegion(*lambda.subregion());

    auto states = LambdaEntryMemStateOperator::Create(memoryStateArgument, memoryNodes.size());
    stateMap.InsertStates(memoryNodes, states);
  };

  auto EncodeExit = [this](const lambda::node & lambda)
  {
    auto subregion = lambda.subregion();
    auto & memoryNodes = Context_->GetMemoryNodeProvider().GetLambdaExitNodes(lambda);
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto memoryStateResult = GetMemoryStateResult(lambda);

    auto states = stateMap.GetStates(*subregion, memoryNodes);
    auto mergedState = LambdaExitMemStateOperator::Create(subregion, states);
    memoryStateResult->divert_to(mergedState);

    stateMap.PopRegion(*lambda.subregion());
  };

  EncodeEntry(lambda);
  EncodeRegion(*lambda.subregion());
  EncodeExit(lambda);
}

void
BasicEncoder::EncodePhi(const phi::node & phi)
{
  EncodeRegion(*phi.subregion());
}

void
BasicEncoder::EncodeDelta(const delta::node & delta)
{
  /* Nothing needs to be done */
}

void
BasicEncoder::EncodeGamma(jive::gamma_node & gamma)
{
  auto EncodeEntry = [this](jive::gamma_node & gamma)
  {
    auto region = gamma.region();
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto memoryNodes = Context_->GetMemoryNodeProvider().GetGammaEntryNodes(gamma);

    auto states = stateMap.GetStates(*region, memoryNodes);
    for (size_t n = 0; n < states.size(); n++) {
      auto state = states[n];
      auto memoryNode = memoryNodes[n];

      auto gammaInput = gamma.add_entryvar(state);
      for (auto & argument : *gammaInput)
        stateMap.InsertState(*memoryNode, argument);
    }
  };

  auto EncodeExit = [this](jive::gamma_node & gamma)
  {
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto memoryNodes = Context_->GetMemoryNodeProvider().GetGammaExitNodes(gamma);

    for (auto & memoryNode : memoryNodes) {
      std::vector<jive::output*> states;
      for (size_t n = 0; n < gamma.nsubregions(); n++) {
        auto subregion = gamma.subregion(n);

        auto state = stateMap.GetState(*subregion, *memoryNode);
        states.push_back(state);
      }

      auto state = gamma.add_exitvar(states);
      stateMap.ReplaceState(*memoryNode, *state);
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
BasicEncoder::EncodeTheta(jive::theta_node & theta)
{
  auto EncodeEntry = [this](jive::theta_node & theta)
  {
    auto region = theta.region();
    auto & stateMap = Context_->GetRegionalizedStateMap();
    auto & memoryNodes = Context_->GetMemoryNodeProvider().GetThetaEntryExitNodes(theta);

    std::vector<jive::theta_output*> thetaStateOutputs;
    auto states = stateMap.GetStates(*region, memoryNodes);
    for (size_t n = 0; n < states.size(); n++) {
      auto state = states[n];
      auto memoryNode = memoryNodes[n];

      auto thetaStateOutput = theta.add_loopvar(state);
      stateMap.InsertState(*memoryNode, *thetaStateOutput->argument());
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
    auto & memoryNodes = Context_->GetMemoryNodeProvider().GetThetaEntryExitNodes(theta);

    JLM_ASSERT(memoryNodes.size() == thetaStateOutputs.size());
    for (size_t n = 0; n < thetaStateOutputs.size(); n++) {
      auto thetaOutput = thetaStateOutputs[n];
      auto memoryNode = memoryNodes[n];

      auto state = stateMap.GetState(*subregion, *memoryNode);
      thetaOutput->result()->divert_to(state);
      stateMap.ReplaceState(*memoryNode, *thetaOutput);
    }
  };

  Context_->GetRegionalizedStateMap().PushRegion(*theta.subregion());

  auto thetaStateOutputs = EncodeEntry(theta);
  EncodeRegion(*theta.subregion());
  EncodeExit(theta, thetaStateOutputs);

  Context_->GetRegionalizedStateMap().PopRegion(*theta.subregion());
}

}

