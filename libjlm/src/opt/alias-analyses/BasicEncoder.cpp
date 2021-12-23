/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/opt/alias-analyses/BasicEncoder.hpp>
#include <jlm/opt/alias-analyses/Operators.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/dne.hpp>

#include <jive/arch/addresstype.hpp>
#include <jive/rvsdg/traverser.hpp>

namespace jlm {
namespace aa {

static jive::input *
call_memstate_input(const jive::simple_node & node)
{
  JLM_ASSERT(is<call_op>(&node));

  /*
    FIXME: This function should be part of the call node.
  */
  for (size_t n = 0; n < node.ninputs(); n++) {
    auto input = node.input(n);
    if (jive::is<jive::memtype>(input->type()))
      return input;
  }

  JLM_ASSERT(0 && "This should have never happened!");
}

static jive::output *
call_memstate_output(const jive::simple_node & node)
{
  JLM_ASSERT(is<call_op>(&node));

  /*
    FIXME: This function should be part of the call node.
  */
  for (size_t n = 0; n < node.noutputs(); n++) {
    auto output = node.output(n);
    if (jive::is<jive::memtype>(output->type()))
      return output;
  }

  JLM_UNREACHABLE("This should have never happened!");
}

static jive::argument *
lambda_memstate_argument(const lambda::node & lambda)
{
  auto subregion = lambda.subregion();

  /*
    FIXME: This function should be part of the lambda node.
  */
  for (size_t n = 0; n < subregion->narguments(); n++) {
    auto argument = subregion->argument(n);
    if (jive::is<jive::memtype>(argument->type()))
      return argument;
  }

  JLM_UNREACHABLE("This should have never happened!");
}

static jive::result *
lambda_memstate_result(const lambda::node & lambda)
{
  auto subregion = lambda.subregion();

  /*
    FIXME: This function should be part of the lambda node.
  */
  for (size_t n = 0; n < subregion->nresults(); n++) {
    auto result = subregion->result(n);
    if (jive::is<jive::memtype>(result->type()))
      return result;
  }

  JLM_UNREACHABLE("This should have never happened!");
}

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
  contains(const PointsToGraph::memnode * node) const noexcept
  {
    return states_.find(node) != states_.end();
  }

  jive::output *
  state(const PointsToGraph::memnode * node) const noexcept
  {
    JLM_ASSERT(contains(node));
    return states_.at(node);
  }

  std::vector<jive::output*>
  states(const std::vector<const PointsToGraph::memnode*> & nodes) const
  {
    std::vector<jive::output*> states;
    states.reserve(nodes.size());
    for (auto & node : nodes)
      states.push_back(state(node));

    return states;
  }

  void
  insert(
    const PointsToGraph::memnode * node,
    jive::output * state)
  {
    JLM_ASSERT(!contains(node));
    JLM_ASSERT(is<jive::memtype>(state->type()));

    states_[node] = state;
  }

  void
  insert(
    const std::vector<const PointsToGraph::memnode*> & nodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(nodes.size() == states.size());

    for (size_t n = 0; n < nodes.size(); n++)
      insert(nodes[n], states[n]);
  }

  void
  replace(
    const PointsToGraph::memnode * node,
    jive::output * state)
  {
    JLM_ASSERT(contains(node));
    JLM_ASSERT(is<jive::memtype>(state->type()));

    states_[node] = state;
  }

  void
  replace(
    const std::vector<const PointsToGraph::memnode*> & nodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(nodes.size() == states.size());
    for (size_t n = 0; n < nodes.size(); n++)
      replace(nodes[n], states[n]);
  }

  static std::unique_ptr<StateMap>
  Create()
  {
    return std::make_unique<StateMap>();
  }

private:
  std::unordered_map<const PointsToGraph::memnode*, jive::output*> states_;
};

/** FIXME: write documentation
*/
class RegionalizedStateMap final {
public:
  explicit
  RegionalizedStateMap(const jlm::aa::PointsToGraph & ptg)
  {
    CollectAddressMemNodes(ptg);
  }

  RegionalizedStateMap(const RegionalizedStateMap&) = delete;

  RegionalizedStateMap(RegionalizedStateMap&&) = delete;

  RegionalizedStateMap &
  operator=(const RegionalizedStateMap&) = delete;

  RegionalizedStateMap &
  operator=(RegionalizedStateMap&&) = delete;

  bool
  contains(
    const jive::region & region,
    const PointsToGraph::memnode * node)
  {
    return GetOrInsertStateMap(region).contains(node);
  }

  void
  insert(
    const PointsToGraph::memnode * node,
    jive::output * state)
  {
    GetOrInsertStateMap(*state->region()).insert(node, state);
  }

  void
  insert(
    const std::vector<const PointsToGraph::memnode*> & nodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(nodes.size() == states.size());
    JLM_ASSERT(!nodes.empty());

    GetOrInsertStateMap(*states[0]->region()).insert(nodes, states);
  }

  void
  ReplaceAddress(
    const jive::output * oldAddress,
    const jive::output * newAddress)
  {
    JLM_ASSERT(AddressMemNodeMap_.find(oldAddress) != AddressMemNodeMap_.end());
    JLM_ASSERT(AddressMemNodeMap_.find(newAddress) == AddressMemNodeMap_.end());

    AddressMemNodeMap_[newAddress] = AddressMemNodeMap_[oldAddress];
    AddressMemNodeMap_.erase(oldAddress);
  }

  void
  replace(
    const jive::output * output,
    const std::vector<jive::output*> & states)
  {
    auto nodes = memnodes(output);
    GetOrInsertStateMap(*output->region()).replace(nodes, states);
  }

  void
  replace(
    const PointsToGraph::memnode * node,
    jive::output * state)
  {
    GetOrInsertStateMap(*state->region()).replace(node, state);
  }

  void
  replace(
    const std::vector<const PointsToGraph::memnode*> & nodes,
    const std::vector<jive::output*> & states)
  {
    JLM_ASSERT(nodes.size() == states.size());
    JLM_ASSERT(!nodes.empty());

    GetOrInsertStateMap(*states[0]->region()).replace(nodes, states);
  }

  std::vector<jive::output*>
  states(const jive::output * output) noexcept
  {
    auto nodes = memnodes(output);
    return states(*output->region(), nodes);
  }

  std::vector<jive::output*>
  states(
    const jive::region & region,
    const std::vector<const PointsToGraph::memnode*> & nodes)
  {
    return GetOrInsertStateMap(region).states(nodes);
  }

  jive::output *
  state(
    const jive::region & region,
    const PointsToGraph::memnode & memnode)
  {
    return states(region, {&memnode})[0];
  }

  std::vector<const PointsToGraph::memnode*>
  memnodes(const jive::output * output)
  {
    JLM_ASSERT(is<ptrtype>(output->type()));
    JLM_ASSERT(AddressMemNodeMap_.find(output) != AddressMemNodeMap_.end());
    JLM_ASSERT(!AddressMemNodeMap_[output].empty());

    return AddressMemNodeMap_[output];
  }

  static std::unique_ptr<BasicEncoder::Context>
  Create(const jlm::aa::PointsToGraph & ptg)
  {
    return std::make_unique<BasicEncoder::Context>(ptg);
  }

private:
  StateMap &
  GetOrInsertStateMap(const jive::region & region) noexcept
  {
    if (StateMaps_.find(&region) == StateMaps_.end())
      StateMaps_[&region] = StateMap::Create();

    return *StateMaps_[&region];
  }

  void
  CollectAddressMemNodes(const jlm::aa::PointsToGraph & ptg)
  {
    for (auto & regnode : ptg.regnodes()) {
      auto output = regnode.first;
      auto memNodes = PointsToGraph::regnode::allocators(*regnode.second);

      AddressMemNodeMap_[output] = memNodes;
    }
  }

  std::unordered_map<const jive::output*, std::vector<const PointsToGraph::memnode*>> AddressMemNodeMap_;
  std::unordered_map<const jive::region*, std::unique_ptr<StateMap>> StateMaps_;
};

/* BasicEncoder class */

/** FIXME: write documentation
*/
class BasicEncoder::Context final {
public:
  explicit
  Context(const jlm::aa::PointsToGraph & ptg)
    : StateMap_(ptg)
  {
    collect_memnodes(ptg);
  }

  Context(const Context&) = delete;

  Context(Context&&) = delete;

  Context&
  operator=(const Context&) = delete;

  Context&
  operator=(Context&&) = delete;

  RegionalizedStateMap &
  StateMap() noexcept
  {
    return StateMap_;
  }

  const std::vector<const PointsToGraph::memnode*> &
  MemoryNodes()
  {
    return MemoryNodes_;
  }

  static std::unique_ptr<BasicEncoder::Context>
  Create(const jlm::aa::PointsToGraph & ptg)
  {
    return std::make_unique<Context>(ptg);
  }

private:
  void
  collect_memnodes(const jlm::aa::PointsToGraph & ptg)
  {
    for (auto & pair : ptg.allocnodes())
      MemoryNodes_.push_back(pair.second.get());

    for (auto & pair : ptg.impnodes())
      MemoryNodes_.push_back(static_cast<const PointsToGraph::memnode*>(pair.second.get()));
  }

  RegionalizedStateMap StateMap_;
  std::vector<const PointsToGraph::memnode*> MemoryNodes_;
};

BasicEncoder::~BasicEncoder() = default;

BasicEncoder::BasicEncoder(jlm::aa::PointsToGraph & ptg)
  : Ptg_(ptg)
{
  UnlinkMemUnknown(Ptg_);
}

void
BasicEncoder::UnlinkMemUnknown(jlm::aa::PointsToGraph & ptg)
{
  /*
    FIXME: There should be a kind of memory nodes iterator in the points-to graph.
  */
  std::vector<PointsToGraph::Node*> memNodes;
  for (auto & node : ptg.allocnodes())
    memNodes.push_back(node.second.get());
  for (auto & node : ptg.impnodes())
    memNodes.push_back(node.second.get());

  auto & memUnknown = ptg.memunknown();
  while (memUnknown.nsources() != 0) {
    auto & source = *memUnknown.sources().begin();
    for (auto & memNode : memNodes)
      source.add_edge(memNode);
    source.remove_edge(&memUnknown);
  }
}

void
BasicEncoder::Encode(
  jlm::aa::PointsToGraph & ptg,
  rvsdg_module & module)
{
  jlm::aa::BasicEncoder encoder(ptg);
  encoder.Encode(module);
}

void
BasicEncoder::Encode(rvsdg_module & module)
{
  Context_ = Context::Create(Ptg());

  MemoryStateEncoder::Encode(*module.graph()->root());

  /*
    Remove all nodes that became dead throughout the encoding.
  */
  jlm::dne dne;
  dne.run(*module.graph()->root());

}

void
BasicEncoder::EncodeAlloca(const jive::simple_node & node)
{
  JLM_ASSERT(is<alloca_op>(&node));

  auto & memnode = Ptg().find(&node);
  Context_->StateMap().replace(&memnode, node.output(1));
}

void
BasicEncoder::EncodeMalloc(const jive::simple_node & node)
{
  JLM_ASSERT(is<malloc_op>(&node));

  auto & memnode = Ptg().find(&node);
  Context_->StateMap().replace(&memnode, node.output(1));
}

void
BasicEncoder::EncodeLoad(const jive::simple_node & node)
{
  JLM_ASSERT(is<load_op>(&node));
  auto & op = *static_cast<const load_op*>(&node.operation());
  auto & smap = Context_->StateMap();

  auto address = node.input(0)->origin();
  auto instates = smap.states(address);
  auto oldResult = node.output(0);

  auto outputs = load_op::create(address, instates, op.alignment());
  oldResult->divert_users(outputs[0]);

  smap.replace(address, {std::next(outputs.begin()), outputs.end()});

  if (is<ptrtype>(oldResult->type()))
    smap.ReplaceAddress(oldResult, outputs[0]);
}

void
BasicEncoder::EncodeStore(const jive::simple_node & node)
{
  JLM_ASSERT(is<store_op>(&node));
  auto & op = *static_cast<const store_op*>(&node.operation());
  auto & smap = Context_->StateMap();

  auto address = node.input(0)->origin();
  auto value = node.input(1)->origin();
  auto instates = smap.states(address);

  auto outstates = store_op::create(address, value, instates, op.alignment());

  smap.replace(address, outstates);
}

void
BasicEncoder::EncodeFree(const jive::simple_node & node)
{
  JLM_ASSERT(is<free_op>(&node));
  auto & smap = Context_->StateMap();

  auto address = node.input(0)->origin();
  auto iostate = node.input(node.ninputs()-1)->origin();
  auto instates = smap.states(address);

  auto outputs = free_op::create(address, instates, iostate);
  node.output(node.noutputs()-1)->divert_users(outputs.back());

  smap.replace(address, {outputs.begin(), std::prev(outputs.end())});
}

void
BasicEncoder::EncodeCall(const jive::simple_node & node)
{
  JLM_ASSERT(is<call_op>(&node));

  auto EncodeEntry = [this](const jive::simple_node & node)
  {
    auto region = node.region();
    auto & memnodes = Context_->MemoryNodes();
    auto meminput = call_memstate_input(node);

    auto states = Context_->StateMap().states(*region, memnodes);
    auto state = CallEntryMemStateOperator::Create(region, states);
    meminput->divert_to(state);
  };

  auto EncodeExit = [this](const jive::simple_node & node)
  {
    auto memoutput = call_memstate_output(node);
    auto & memnodes = Context_->MemoryNodes();

    auto states = CallExitMemStateOperator::Create(memoutput, memnodes.size());
    Context_->StateMap().replace(memnodes, states);
  };

  EncodeEntry(node);
  EncodeExit(node);
}

void
BasicEncoder::EncodeMemcpy(const jive::simple_node & node)
{
  JLM_ASSERT(is<Memcpy>(&node));
  auto & smap = Context_->StateMap();

  auto destination = node.input(0)->origin();
  auto source = node.input(1)->origin();
  auto length = node.input(2)->origin();
  auto isVolatile = node.input(3)->origin();

  auto destinationStates = smap.states(destination);
  auto sourceStates = smap.states(source);

  auto inStates = destinationStates;
  inStates.insert(inStates.end(), sourceStates.begin(), sourceStates.end());

  auto outStates = Memcpy::create(destination, source, length, isVolatile, inStates);

  auto begin = outStates.begin();
  auto end = std::next(outStates.begin(), destinationStates.size());
  smap.replace(destination, {begin, end});

  JLM_ASSERT((size_t)std::distance(end, outStates.end()) == sourceStates.size());
  smap.replace(source, {end, outStates.end()});
}

void
BasicEncoder::Encode(const lambda::node & lambda)
{
  auto EncodeEntry = [this](const lambda::node & lambda)
  {
    auto memstate = lambda_memstate_argument(lambda);
    auto & memnodes = Context_->MemoryNodes();

    auto states = LambdaEntryMemStateOperator::Create(memstate, memnodes.size());
    Context_->StateMap().insert(memnodes, states);
  };

  auto EncodeExit = [this](const lambda::node & lambda)
  {
    auto subregion = lambda.subregion();
    auto & memnodes = Context_->MemoryNodes();
    auto memresult = lambda_memstate_result(lambda);

    auto states = Context_->StateMap().states(*subregion, memnodes);
    auto state = LambdaExitMemStateOperator::Create(subregion, states);
    memresult->divert_to(state);
  };

  EncodeEntry(lambda);
  MemoryStateEncoder::Encode(*lambda.subregion());
  EncodeExit(lambda);
}

void
BasicEncoder::Encode(const jive::phi::node & phi)
{
  MemoryStateEncoder::Encode(*phi.subregion());
}

void
BasicEncoder::Encode(const delta::node & delta)
{
  /* Nothing needs to be done */
}

void
BasicEncoder::Encode(jive::gamma_node & gamma)
{
  auto EncodeEntry = [this](jive::gamma_node & gamma)
  {
    auto region = gamma.region();
    auto & memNodes = Context_->MemoryNodes();

    auto states = Context_->StateMap().states(*region, memNodes);
    for (size_t n = 0; n < states.size(); n++) {
      auto state = states[n];
      auto memNode = memNodes[n];

      auto ev = gamma.add_entryvar(state);
      for (auto & argument : *ev)
        Context_->StateMap().insert(memNode, &argument);
    }
  };

  auto EncodeExit = [this](jive::gamma_node & gamma)
  {
    auto & memNodes = Context_->MemoryNodes();

    for (auto & memNode : memNodes) {
      std::vector<jive::output*> states;
      for (size_t n = 0; n < gamma.nsubregions(); n++) {
        auto subregion = gamma.subregion(n);

        auto state = Context_->StateMap().state(*subregion, *memNode);
        states.push_back(state);
      }

      auto state = gamma.add_exitvar(states);
      Context_->StateMap().replace(memNode, state);
    }
  };

  EncodeEntry(gamma);
  for (size_t n = 0; n < gamma.nsubregions(); n++)
    MemoryStateEncoder::Encode(*gamma.subregion(n));
  EncodeExit(gamma);
}

void
BasicEncoder::Encode(jive::theta_node & theta)
{
  auto EncodeEntry = [this](jive::theta_node & theta)
  {
    auto region = theta.region();
    auto & memNodes = Context_->MemoryNodes();

    std::vector<jive::theta_output*> loopvars;
    auto states = Context_->StateMap().states(*region, memNodes);
    for (size_t n = 0; n < states.size(); n++) {
      auto state = states[n];
      auto memNode = memNodes[n];

      auto lv = theta.add_loopvar(state);
      Context_->StateMap().insert(memNode, lv->argument());
      loopvars.push_back(lv);
    }

    return loopvars;
  };

  auto EncodeExit = [this](
    jive::theta_node & theta,
    const std::vector<jive::theta_output*> & loopvars)
  {
    auto subregion = theta.subregion();
    auto & memNodes = Context_->MemoryNodes();

    JLM_ASSERT(memNodes.size() == loopvars.size());
    for (size_t n = 0; n < loopvars.size(); n++) {
      auto loopvar = loopvars[n];
      auto memNode = memNodes[n];

      auto state = Context_->StateMap().state(*subregion, *memNode);
      loopvar->result()->divert_to(state);
      Context_->StateMap().replace(memNode, loopvar);
    }
  };

  auto loopvars = EncodeEntry(theta);
  MemoryStateEncoder::Encode(*theta.subregion());
  EncodeExit(theta, loopvars);
}

}}

