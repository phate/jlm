/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/opt/alias-analyses/encoders.hpp>
#include <jlm/opt/alias-analyses/operators.hpp>
#include <jlm/opt/alias-analyses/pointsto-graph.hpp>
#include <jlm/opt/dne.hpp>

#include <jive/arch/addresstype.hpp>
#include <jive/rvsdg/traverser.hpp>

namespace jlm {
namespace aa {

MemoryStateEncoder::~MemoryStateEncoder()
{}

static std::vector<std::string>
dbgstrs(const std::vector<const ptg::memnode*> & memnodes)
{
	std::vector<std::string> strs;
	for (auto memnode : memnodes)
		strs.push_back(memnode->debug_string());

	return strs;
}

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

/** \brief Hash map for mapping point-to graph memory nodes to RVSDG memory states.
*/
class StateMap final {
public:
	StateMap()
	{}

	StateMap(const StateMap&) = delete;

	StateMap(StateMap&&) = delete;

	StateMap &
	operator=(const StateMap&) = delete;

	StateMap &
	operator=(StateMap&&) = delete;

	bool
	contains(const ptg::memnode * node) const noexcept
	{
		return states_.find(node) != states_.end();
	}

	jive::output *
	state(const ptg::memnode * node) const noexcept
	{
		JLM_ASSERT(contains(node));
		return states_.at(node);
	}

	std::vector<jive::output*>
	states(const std::vector<const ptg::memnode*> & nodes)
	{
		std::vector<jive::output*> states;
		for (auto & node : nodes)
			states.push_back(state(node));

		return states;
	}

	void
	insert(
		const ptg::memnode * node,
		jive::output * state)
	{
		JLM_ASSERT(!contains(node));
		JLM_ASSERT(is<jive::memtype>(state->type()));

		states_[node] = state;
	}

	void
	insert(
		const std::vector<const ptg::memnode*> & nodes,
		const std::vector<jive::output*> & states)
	{
		JLM_ASSERT(nodes.size() == states.size());

		for (size_t n = 0; n < nodes.size(); n++)
			insert(nodes[n], states[n]);
	}

	void
	replace(
		const ptg::memnode * node,
		jive::output * state)
	{
		JLM_ASSERT(contains(node));
		JLM_ASSERT(is<jive::memtype>(state->type()));

		states_[node] = state;
	}

	void
	replace(
		const std::vector<const ptg::memnode*> nodes,
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
	std::unordered_map<const ptg::memnode*, jive::output*> states_;
};

/** FIXME: write documentation
*/
class RegionalizedStateMap final {
public:
	RegionalizedStateMap(const jlm::aa::ptg & ptg)
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
		const ptg::memnode * node)
	{
		return GetOrInsertStateMap(region).contains(node);
	}

	void
	insert(
		const ptg::memnode * node,
		jive::output * state)
	{
		GetOrInsertStateMap(*state->region()).insert(node, state);
	}

	void
	insert(
		const std::vector<const ptg::memnode*> & nodes,
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
		const ptg::memnode * node,
		jive::output * state)
	{
		GetOrInsertStateMap(*state->region()).replace(node, state);
	}

	void
	replace(
		const std::vector<const ptg::memnode*> & nodes,
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
		const std::vector<const ptg::memnode*> & nodes)
	{
		return GetOrInsertStateMap(region).states(nodes);
	}

	jive::output *
	state(
		const jive::region & region,
		const ptg::memnode & memnode)
	{
		return states(region, {&memnode})[0];
	}

	std::vector<const ptg::memnode*>
	memnodes(const jive::output * output)
	{
		JLM_ASSERT(is<ptrtype>(output->type()));
		JLM_ASSERT(AddressMemNodeMap_.find(output) != AddressMemNodeMap_.end());
		JLM_ASSERT(AddressMemNodeMap_[output].size() != 0);

		return AddressMemNodeMap_[output];
	}

	std::unique_ptr<BasicEncoder::Context>
	Create(const jlm::aa::ptg & ptg)
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
	CollectAddressMemNodes(const jlm::aa::ptg & ptg)
	{
		for (auto & regnode : ptg.regnodes()) {
			auto output = regnode.first;
			auto memNodes = ptg::regnode::allocators(*regnode.second);

			AddressMemNodeMap_[output] = memNodes;
		}
	}

	std::unordered_map<const jive::output*, std::vector<const ptg::memnode*>> AddressMemNodeMap_;
	std::unordered_map<const jive::region*, std::unique_ptr<StateMap>> StateMaps_;
};

/* MemoryStateEncoder class */

void
MemoryStateEncoder::Encode(const jive::simple_node & node)
{
	static std::unordered_map<
		std::type_index
	, std::function<void(MemoryStateEncoder&, const jive::simple_node&)>
	> nodes({
	  {typeid(alloca_op), [](auto & mse, auto & node){ mse.EncodeAlloca(node); }}
	, {typeid(malloc_op), [](auto & mse, auto & node){ mse.EncodeMalloc(node); }}
	, {typeid(load_op),   [](auto & mse, auto & node){ mse.EncodeLoad(node);   }}
	, {typeid(store_op),  [](auto & mse, auto & node){ mse.EncodeStore(node);  }}
	, {typeid(call_op),   [](auto & mse, auto & node){ mse.EncodeCall(node);   }}
	, {typeid(free_op),   [](auto & mse, auto & node){ mse.EncodeFree(node);   }}
	, {typeid(Memcpy),    [](auto & mse, auto & node){ mse.EncodeMemcpy(node); }}
	});

	auto & op = node.operation();
	if (nodes.find(typeid(op)) == nodes.end())
		return;

	nodes[typeid(op)](*this, node);
}

void
MemoryStateEncoder::Encode(jive::structural_node & node)
{
	auto encodeLambda = [](auto & mse, auto & n){mse.Encode(*static_cast<lambda::node*>(&n));     };
	auto encodeDelta  = [](auto & mse, auto & n){mse.Encode(*static_cast<delta::node*>(&n));      };
	auto encodePhi    = [](auto & mse, auto & n){mse.Encode(*static_cast<jive::phi::node*>(&n));  };
	auto encodeGamma  = [](auto & mse, auto & n){mse.Encode(*static_cast<jive::gamma_node*>(&n)); };
	auto encodeTheta  = [](auto & mse, auto & n){mse.Encode(*static_cast<jive::theta_node*>(&n)); };

	static std::unordered_map<
		std::type_index,
		std::function<void(MemoryStateEncoder&, jive::structural_node&)>
	> nodes({
	  {typeid(lambda::operation),    encodeLambda }
	, {typeid(delta::operation),     encodeDelta  }
	, {typeid(jive::phi::operation), encodePhi    }
	, {typeid(jive::gamma_op),       encodeGamma  }
	, {typeid(jive::theta_op),       encodeTheta  }
	});

	auto & op = node.operation();
	JLM_ASSERT(nodes.find(typeid(op)) != nodes.end());
	nodes[typeid(op)](*this, node);
}

void
MemoryStateEncoder::Encode(jive::region & region)
{
	using namespace jive;

	topdown_traverser traverser(&region);
	for (auto & node : traverser) {
		if (auto simpnode = dynamic_cast<const simple_node*>(node)) {
			Encode(*simpnode);
			continue;
		}

		JLM_ASSERT(is<structural_op>(node));
		auto structnode = static_cast<structural_node*>(node);
		Encode(*structnode);
	}
}

/* BasicEncoder class */

/** FIXME: write documentation
*/
class BasicEncoder::Context final {
public:
	Context(const jlm::aa::ptg & ptg)
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

	const std::vector<const ptg::memnode*> &
	MemoryNodes()
	{
		return MemoryNodes_;
	}

	static std::unique_ptr<BasicEncoder::Context>
	Create(const jlm::aa::ptg & ptg)
	{
		return std::make_unique<Context>(ptg);
	}

private:
	void
	collect_memnodes(const jlm::aa::ptg & ptg)
	{
		for (auto & pair : ptg.allocnodes())
			MemoryNodes_.push_back(pair.second.get());

		for (auto & pair : ptg.impnodes())
			MemoryNodes_.push_back(static_cast<const ptg::memnode*>(pair.second.get()));
	}

	RegionalizedStateMap StateMap_;
	std::vector<const ptg::memnode*> MemoryNodes_;
};

BasicEncoder::~BasicEncoder()
{}

BasicEncoder::BasicEncoder(jlm::aa::ptg & ptg)
: Ptg_(ptg)
{
	UnlinkMemUnknown(Ptg_);
}

void
BasicEncoder::UnlinkMemUnknown(jlm::aa::ptg & ptg)
{
	/*
		FIXME: There should be a kind of memory nodes iterator in the points-to graph.
	*/
	std::vector<ptg::node*> memNodes;
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
	jlm::aa::ptg & ptg,
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
		auto state = CallEntryMemStateOperator::Create(region, states, dbgstrs(memnodes));
		meminput->divert_to(state);
	};

	auto EncodeExit = [this](const jive::simple_node & node)
	{
		auto memoutput = call_memstate_output(node);
		auto & memnodes = Context_->MemoryNodes();

		auto states = CallExitMemStateOperator::Create(memoutput, memnodes.size(), dbgstrs(memnodes));
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

		auto states = LambdaEntryMemStateOperator::Create(memstate, memnodes.size(), dbgstrs(memnodes));
		Context_->StateMap().insert(memnodes, states);
	};

	auto EncodeExit = [this](const lambda::node & lambda)
	{
		auto subregion = lambda.subregion();
		auto & memnodes = Context_->MemoryNodes();
		auto memresult = lambda_memstate_result(lambda);

		auto states = Context_->StateMap().states(*subregion, memnodes);
		auto state = LambdaExitMemStateOperator::Create(subregion, states, dbgstrs(memnodes));
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

#if 0
/** FIXME: write documentation
*/
class statemap final {
public:
	statemap(const jlm::aa::ptg & ptg)
	: ptg_(&ptg)
	{}

	statemap(const statemap&) = delete;

	statemap(statemap&&) = delete;

	statemap &
	operator=(const statemap&) = delete;

	statemap &
	operator=(statemap&&) = delete;

	const jlm::aa::ptg &
	ptg() const noexcept
	{
		return *ptg_;
	}

	bool
	contains(
		const jive::region & region,
		const ptg::memnode * node)
	{
		return getOrInsertMemoryNodeMap(region).contains(node);
	}

	void
	insert(
		const ptg::memnode * node,
		jive::output * state)
	{
		getOrInsertMemoryNodeMap(*state->region()).insert(node, state);
	}

	void
	insert(
		const std::vector<const ptg::memnode*> & nodes,
		const std::vector<jive::output*> & states)
	{
		JLM_ASSERT(nodes.size() == states.size());
		JLM_ASSERT(!nodes.empty());

		getOrInsertMemoryNodeMap(*states[0]->region()).insert(nodes, states);
	}

	void
	replace(
		const jive::output * output,
		const std::vector<jive::output*> & states)
	{
		auto nodes = memnodes(output);
		getOrInsertMemoryNodeMap(*output->region()).replace(nodes, states);
	}

	void
	replace(
		const ptg::memnode * node,
		jive::output * state)
	{
		getOrInsertMemoryNodeMap(*state->region()).replace(node, state);
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
		const std::vector<const ptg::memnode*> & nodes)
	{
		return getOrInsertMemoryNodeMap(region).states(nodes);
	}

	jive::output *
	state(
		const jive::region & region,
		const ptg::memnode & memnode)
	{
		return states(region, {&memnode})[0];
	}

	std::vector<const ptg::memnode*>
	memnodes(const jive::output * output)
	{
		JLM_ASSERT(jive::is<ptrtype>(output->type()));

		if (memnodes_.find(output) != memnodes_.end())
			return memnodes_[output];

		auto & regnode = ptg_->find_regnode(output);
		auto nodes = ptg::regnode::allocators(regnode);
		memnodes_[output] = nodes;

		return nodes;
	}

private:
	MemoryNodeMap &
	getOrInsertMemoryNodeMap(const jive::region & region) noexcept
	{
		if (amaps_.find(&region) == amaps_.end())
			amaps_[&region] = allocatormap();

		return amaps_[&region];
	}

	const jlm::aa::ptg * ptg_;
	std::unordered_map<const jive::region*, MemoryNodeMap> amaps_;
	std::unordered_map<const jive::output*, std::vector<const ptg::memnode*>> memnodes_;
};

class lambda_memnodes final
{
	using memnode_vector = std::vector<const ptg::memnode*>;
	using memnode_set = std::unordered_set<const ptg::memnode*>;

public:
	lambda_memnodes(
		const memnode_set & entry,
		const memnode_set & exit)
	: entry_(entry.begin(), entry.end())
	, exit_(exit.begin(), exit.end())
	{}

	const memnode_vector &
	entry() const noexcept
	{
		return entry_;
	}

	const memnode_vector &
	exit() const noexcept
	{
		return exit_;
	}

	bool
	has_empty_entry() const noexcept
	{
		return entry_.empty();
	}

	bool
	has_empty_exit() const noexcept
	{
		return exit_.empty();
	}

	bool
	is_empty() const noexcept
	{
		return has_empty_entry()
		    && has_empty_exit();
	}

	static std::unique_ptr<lambda_memnodes>
	create(
		const lambda::node & lambda,
		statemap & smap,
		const memnode_set & deltas,
		const memnode_set & imports,
		const memnode_set & lambdas)
	{
		auto entry = create_entry(lambda, smap, deltas, imports, lambdas);
		auto exit = create_exit(lambda, smap, entry);

		return std::make_unique<lambda_memnodes>(entry, exit);
	}

private:
	static memnode_set
	create_entry(
		const lambda::node & lambda,
		statemap & smap,
		const memnode_set & deltas,
		const memnode_set & imports,
		const memnode_set & lambdas)
	{
		memnode_set memnodes;
		for (auto & argument : lambda.fctarguments()) {
			if (!jive::is<ptrtype>(argument.type()))
				continue;

			auto nodes = smap.memnodes(&argument);
			memnodes.insert(nodes.begin(), nodes.end());
		}

		memnodes.insert(deltas.begin(), deltas.end());
		memnodes.insert(imports.begin(), imports.end());
		memnodes.insert(lambdas.begin(), lambdas.end());

		return memnodes;
	}

	static memnode_set
	create_exit(
		const lambda::node & lambda,
		statemap & smap,
		const memnode_set entry)
	{
		memnode_set memnodes;
		for (auto & result : lambda.fctresults()) {
			if (!jive::is<ptrtype>(result.type()))
				continue;

			auto nodes = smap.memnodes(result.origin());
			for (auto & node : nodes) {
				if (entry.find(node) != entry.end())
					continue;

				memnodes.insert(node);
			}
		}

		return memnodes;
	}

	memnode_vector entry_;
	memnode_vector exit_;
};

/** \brief FIXME: write documentation
*/
class context final {
	using memnode_set = std::unordered_set<const ptg::memnode*>;
	using lambda_memnode_map = std::unordered_map<
		const lambda::node*,
		std::unique_ptr<lambda_memnodes>>;

public:
	context(const jlm::aa::ptg & ptg)
	: smap_(ptg)
	{
		collect_delta_allocators(ptg);
		collect_import_allocators(ptg);
		collect_lambda_memnodes(ptg);
	}

	context(const context&) = delete;

	context(context&&) = delete;

	context&
	operator=(const context&) = delete;

	context&
	operator=(context&&) = delete;

	const jlm::aa::ptg &
	ptg() const noexcept
	{
		return smap_.ptg();
	}

	jlm::aa::statemap &
	statemap() noexcept
	{
		return smap_;
	}

	const memnode_set &
	delta_allocators() const
	{
		return deltas_;
	}

	const memnode_set &
	import_allocators() const
	{
		return imports_;
	}

	const memnode_set &
	lambda_allocators() const
	{
		return lambdas_;
	}

	const lambda_memnodes &
	memnodes(const lambda::node & lambda)
	{
		if (lmmap_.find(&lambda) != lmmap_.end())
			return *lmmap_[&lambda];

		auto deltas = delta_allocators();
		auto imports = import_allocators();
		auto lambdas = lambda_allocators();

		auto lm = lambda_memnodes::create(lambda, statemap(), deltas, imports, lambdas);
		lmmap_[&lambda] = std::move(lm);

		return *lmmap_[&lambda];
	}

private:
	void
	collect_delta_allocators(const jlm::aa::ptg & ptg)
	{
		for (auto & pair : ptg.allocnodes()) {
			if (jive::is<delta_op>(pair.first))
				deltas_.insert(pair.second.get());
		}
	}

	void
	collect_import_allocators(const jlm::aa::ptg & ptg)
	{
		for (auto & pair : ptg.impnodes())
			imports_.insert(static_cast<const ptg::memnode*>(pair.second.get()));
	}

	void
	collect_lambda_memnodes(const aa::ptg & ptg)
	{
		for (auto & pair : ptg.allocnodes()) {
			if (jive::is<lambda::operation>(pair.first))
				lambdas_.insert(pair.second.get());
		}
	}

	memnode_set deltas_;
	memnode_set imports_;
	memnode_set lambdas_;
	jlm::aa::statemap smap_;
	lambda_memnode_map lmmap_;
};

static void
encode(jive::region&, context&);

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

static std::vector<std::string>
dbgstrs(const std::vector<const ptg::memnode*> & memnodes)
{
	std::vector<std::string> strs;
	for (auto memnode : memnodes)
		strs.push_back(memnode->debug_string());

	return strs;
}

static void
encode(
	const lambda::node & lambda,
	context & ctx)
{
	auto handle_lambda_entry = [](const lambda::node & lambda, context & ctx)
	{
		auto memstate = lambda_memstate_argument(lambda);

		/*
			FIXME: We need to check whether all call sides can be resolved, i.e. all call sides are
			direct calls. We need to be way more conservative with indirect calls.
		*/

		auto & memnodes = ctx.memnodes(lambda).entry();
		if (memnodes.empty())
			/*
				The lambda has no pointer arguments or context variables.
				Nothing needs to be done.
			*/
			return;

		auto lambdastates = lambda_aamux_op::create_entry(memstate, memnodes.size(),
			dbgstrs(memnodes));
		ctx.statemap().insert(memnodes, lambdastates);
	};

	auto handle_lambda_exit = [](const lambda::node & lambda, context & ctx)
	{
		auto entry = ctx.memnodes(lambda).entry();
		auto & exit = ctx.memnodes(lambda).exit();

		if (entry.empty() && exit.empty())
			/*
				The lambda has no pointer arguments, context variables, or results.
				Nothing needs to be done.
			*/
			return;

		entry.insert(entry.end(), exit.begin(), exit.end());

		auto states = ctx.statemap().states(*lambda.subregion(), entry);
		auto state = lambda_aamux_op::create_exit(lambda.subregion(), states, dbgstrs(entry));

		auto memresult = lambda_memstate_result(lambda);
		memresult->divert_to(state);
	};

	handle_lambda_entry(lambda, ctx);
	encode(*lambda.subregion(), ctx);
	handle_lambda_exit(lambda, ctx);
}

static void
encode(
	const delta_node & delta,
	context & ctx)
{
	/* Nothing needs to be done. */
}

template<class T> static void
encode(
	jive::structural_node & node,
	context & ctx)
{
	JLM_ASSERT(dynamic_cast<T*>(&node));
	encode(*static_cast<T*>(&node), ctx);
}

static void
encode(
	jive::structural_node & node,
	context & ctx)
{
	static std::unordered_map<
		std::type_index,
		std::function<void(jive::structural_node&, context&)>
	> nodes({
		{typeid(lambda::operation), encode<lambda::node>}
	, {typeid(delta_op),          encode<delta_node>}
	});

	auto & op = node.operation();
	JLM_ASSERT(nodes.find(typeid(op)) != nodes.end());
	nodes[typeid(op)](node, ctx);
}

static void
encode_alloca(
	const jive::simple_node & node,
	context & ctx)
{
	JLM_ASSERT(is<alloca_op>(&node));

	auto memnode = &ctx.ptg().find(&node);
	ctx.statemap().insert(memnode, node.output(1));
}

static void
encode_malloc(
	const jive::simple_node & node,
	context & ctx)
{
	JLM_ASSERT(is<malloc_op>(&node));

	auto memnode = &ctx.ptg().find(&node);
	ctx.statemap().insert(memnode, node.output(1));
}

static void
encode_load(
	const jive::simple_node & node,
	context & ctx)
{
	JLM_ASSERT(is<load_op>(&node));
	auto & op = *static_cast<const load_op*>(&node.operation());

	auto address = node.input(0)->origin();
	auto instates = ctx.statemap().states(address);

	auto outputs = load_op::create(address, instates, op.alignment());
	node.output(0)->divert_users(outputs[0]);

	ctx.statemap().replace(address, {std::next(outputs.begin()), outputs.end()});
}

static void
encode_store(
	const jive::simple_node & node,
	context & ctx)
{
	JLM_ASSERT(is<store_op>(&node));
	auto & op = *static_cast<const store_op*>(&node.operation());

	auto address = node.input(0)->origin();
	auto value = node.input(1)->origin();
	auto instates = ctx.statemap().states(address);

	auto outstates = store_op::create(address, value, instates, op.alignment());

	ctx.statemap().replace(address, outstates);
}

static std::vector<const ptg::memnode*>
indcall_memnodes(
	const jive::simple_node & node,
	const context & ctx)
{
	JLM_ASSERT(is<call_op>(&node));

	std::unordered_set<const ptg::memnode*> memnodes;
	for (size_t n = 0; n < node.ninputs(); n++) {
		auto origin = node.input(n)->origin();

		if (!jive::is<ptrtype>(origin->type()))
			continue;

		auto & regnode = ctx.ptg().find_regnode(origin);
		/* FIXME: We need to cast the targets to memnodes. Then we can simply to
		insert(regnode.begin(), regnode.end()) */
		for (auto & target : regnode) {
			auto memnode = static_cast<const ptg::memnode*>(&target);
			memnodes.insert(memnode);
		}
	}

	auto deltas = ctx.delta_allocators();
	auto imports = ctx.import_allocators();
	auto lambdas = ctx.lambda_allocators();

	memnodes.insert(deltas.begin(), deltas.end());
	memnodes.insert(imports.begin(), imports.end());
	memnodes.insert(lambdas.begin(), lambdas.end());

	return {memnodes.begin(), memnodes.end()};
}

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

static void
encode_direct_call(
	const jive::simple_node & node,
	const lambda::node & lambda,
	context & ctx)
{
	auto handle_call_entry = [](
		const jive::simple_node & node,
		const lambda::node & lambda,
		context & ctx,
		std::vector<std::string> & dbgstrs)
	{
		auto region = node.region();

		auto & entry = ctx.memnodes(lambda).entry();

		std::vector<jive::output*> instates;
		for (auto & memnode : entry) {
			if (ctx.statemap().contains(*region, memnode)) {
				auto state = ctx.statemap().states(*region, {memnode})[0];
				instates.push_back(state);
				dbgstrs.push_back(memnode->debug_string());
				continue;
			}

			auto state = undef_constant_op::create(region, jive::memtype::instance());
			instates.push_back(state);
			dbgstrs.push_back("*");
		};

		auto state = call_aamux_op::create_entry(region, instates, dbgstrs);
		auto meminput = call_memstate_input(node);
		meminput->divert_to(state);
	};

	auto handle_call_exit = [](
		const jive::simple_node & node,
		const lambda::node & lambda,
		context & ctx,
		std::vector<std::string> & dbgstrs)
	{
		auto & entry = ctx.memnodes(lambda).entry();
		auto & exit = ctx.memnodes(lambda).exit();

		JLM_ASSERT(entry.size() == dbgstrs.size());

		for (auto & memnode : exit)
			dbgstrs.push_back(memnode->debug_string());

		auto memoutput = call_memstate_output(node);
		auto outstates = call_aamux_op::create_exit(memoutput, entry.size() + exit.size(), dbgstrs);

		size_t n;
		auto & smap = ctx.statemap();
		JLM_ASSERT(dbgstrs.size() == outstates.size());
		for (n = 0; n < entry.size(); n++) {
			/*
				FIXME: uff this is really ugly
			*/
			if (dbgstrs[n] == "*")
				continue;

			smap.replace(entry[n], outstates[n]);
		}
		for (; n < outstates.size(); n++) {
			auto & region = *node.region();
			auto outstate = outstates[n];
			auto memnode = exit[n];

			if (smap.contains(region, memnode)) {
				auto other = smap.states(region, {memnode})[0];
				auto operands = std::vector<jive::output*>({other, outstate});
				auto outstate = memstatemux_op::create_merge(operands);
				smap.replace(memnode, outstate);
			} else
				smap.insert(memnode, outstate);
		}
	};

	JLM_ASSERT(is<call_op>(&node));

	std::vector<std::string> dbgstrs;
	handle_call_entry(node, lambda, ctx, dbgstrs);
	handle_call_exit(node, lambda, ctx, dbgstrs);
}

static void
encode_indirect_call(
	const jive::simple_node & node,
	context & ctx)
{
	auto handle_call_entry = [](
		const jive::simple_node & node,
		context & ctx,
		std::vector<std::string> & dbgstrs)
	{
		auto & region = *node.region();

		auto memnodes = indcall_memnodes(node, ctx.ptg());

		std::vector<jive::output*> instates;
		for (auto & memnode : memnodes) {
			auto instate = ctx.statemap().state(region, *memnode);
			instates.push_back(instate);
			dbgstrs.push_back(memnode->debug_string());
		}

		auto state = call_aamux_op::create_entry(&region, instates, dbgstrs);
		auto meminput = call_memstate_input(node);
		meminput->divert_to(state);
	};

	auto handle_call_exit = [](
		const jive::simple_node & node,
		context & ctx,
		std::vector<std::string> & dbgstrs)
	{
		auto memnodes = indcall_memnodes(node, ctx.ptg());
		/*
			FIXME: What about indirect call returning an address.
		*/

		auto memoutput = call_memstate_output(node);
		auto outstates = call_aamux_op::create_exit(memoutput, memnodes.size(), dbgstrs);

		for (size_t n = 0; n < memnodes.size(); n++)
			ctx.statemap().replace(memnodes[n], outstates[n]);
	};

	JLM_ASSERT(is<call_op>(&node));

	std::vector<std::string> dbgstrs;
	handle_call_entry(node, ctx, dbgstrs);
	handle_call_exit(node, ctx, dbgstrs);
}

static void
encode_call(
	const jive::simple_node & node,
	context & ctx)
{
	JLM_ASSERT(is<call_op>(&node));

	if (auto lambda = is_direct_call(node))
		encode_direct_call(node, *lambda, ctx);
	else {
		/*
			FIXME: We might already know here that this indirect call only calls one specific lambda
			and it would therefore be possible to convert the indirect to a direct call.
		*/
		encode_indirect_call(node, ctx);
	}
}

static void
encode_free(
	const jive::simple_node & node,
	context & ctx)
{
	JLM_ASSERT(is<free_op>(&node));

	auto & smap = ctx.statemap();
	auto address = node.input(0)->origin();
	auto iostate = node.input(node.ninputs()-1)->origin();

	auto memnodes = smap.memnodes(address);
	auto instates = smap.states(*node.region(), memnodes);

	auto outputs = free_op::create(address, instates, iostate);

	node.output(node.noutputs()-1)->divert_users(outputs.back());
	smap.replace(address, {outputs.begin(), std::prev(outputs.end())});
}

static void
encode(const jive::simple_node & node, context & ctx)
{
	static std::unordered_map<
		std::type_index
	, std::function<void(const jive::simple_node&, context&)>
	> nodes({
	  {typeid(alloca_op), encode_alloca}
	, {typeid(malloc_op), encode_malloc}
	, {typeid(load_op),   encode_load}
	, {typeid(store_op),  encode_store}
	, {typeid(call_op),   encode_call}
	, {typeid(free_op),   encode_free}
	});

	auto & op = node.operation();
	if (nodes.find(typeid(op)) == nodes.end())
		return;

	nodes[typeid(op)](node, ctx);
}

static void
encode(jive::region & region, context & ctx)
{
	using namespace jive;

	topdown_traverser traverser(&region);
	for (auto & node : traverser) {
		if (auto simpnode = dynamic_cast<const simple_node*>(node)) {
			encode(*simpnode, ctx);
			continue;
		}

		JLM_ASSERT(is<structural_op>(node));
		auto structnode = static_cast<structural_node*>(node);
		encode(*structnode, ctx);
	}
}

void
encode(const jlm::aa::ptg & ptg, rvsdg_module & module)
{
	context ctx(ptg);
	/*
		FIXME: handle imports
	*/
	jlm::aa::encode(*module.graph()->root(), ctx);

	/*
		Remove all nodes that were redenered dead throughout the encoding.
	*/
	jlm::dne dne;
	dne.run(*module.graph()->root());
}
#endif
}}
