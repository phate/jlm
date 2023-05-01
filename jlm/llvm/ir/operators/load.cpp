/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/store.hpp>

namespace jlm {

LoadOperation::~LoadOperation() noexcept
= default;

bool
LoadOperation::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const LoadOperation*>(&other);
  return op
      && op->narguments() == narguments()
      && op->GetPointerType() == GetPointerType()
      && op->GetAlignment() == GetAlignment();
}

std::string
LoadOperation::debug_string() const
{
	return "LOAD";
}

std::unique_ptr<jive::operation>
LoadOperation::copy() const
{
	return std::unique_ptr<jive::operation>(new LoadOperation(*this));
}

/* load normal form */

/*
	sx1 = MemStateMerge si1 ... siM
	v sl1 = load_op a sx1
	=>
	v sl1 ... slM = load_op a si1 ... siM
	sx1 = MemStateMerge sl1 ... slM

  FIXME: The reduction can be generalized: A load node can have multiple operands from different merge nodes.
*/
static bool
is_load_mux_reducible(const std::vector<jive::output*> & operands)
{
	JLM_ASSERT(operands.size() >= 2);

	auto memStateMergeNode = jive::node_output::node(operands[1]);
	if (!is<MemStateMergeOperator>(memStateMergeNode))
		return false;

  if (operands.size() != 2)
    return false;

	return true;
}

/*
	If the producer of a load's address is an alloca, then we can remove
	all state edges originating from other allocas.

	a1 s1 = alloca_op ...
	a2 s2 = alloca_op ...
	s3 = mux_op s1
	v sl1 sl2 sl3 = load_op a1 s1 s2 s3
	=>
	...
	v sl1 sl3 = load_op a1 s1 s3
*/
static bool
is_load_alloca_reducible(const std::vector<jive::output*> & operands)
{
	auto address = operands[0];

	auto allocanode = jive::node_output::node(address);
	if (!is<alloca_op>(allocanode))
		return false;

	for (size_t n = 1; n < operands.size(); n++) {
		auto node = jive::node_output::node(operands[n]);
		if (is<alloca_op>(node) && node != allocanode)
			return true;
	}

	return false;
}

/*
	a s = alloca_op
	ss = store_op a ... s
	v sl = load_op a ss
	=>
	v s
*/
static bool
is_load_store_alloca_reducible(const std::vector<jive::output*> & operands)
{
	if (operands.size() != 2)
		return false;

	auto address = operands[0];
	auto state = operands[1];

	auto alloca = jive::node_output::node(address);
	if (!is<alloca_op>(alloca))
		return false;

	auto store = jive::node_output::node(state);
	if (!is<StoreOperation>(store))
		return false;

	if (store->input(0)->origin() != alloca->output(0))
		return false;

	if (store->ninputs() != 3)
		return false;

	if (store->input(2)->origin() != alloca->output(1))
		return false;

	if (alloca->output(1)->nusers() != 1)
		return false;

	if (address->nusers() != 2)
		return false;

	return true;
}

static bool
is_reducible_state(const jive::output * state, const jive::node * loadalloca)
{
	if (is<StoreOperation>(jive::node_output::node(state))) {
		auto storenode = jive::node_output::node(state);
		auto addressnode = jive::node_output::node(storenode->input(0)->origin());
		if (is<alloca_op>(addressnode) && addressnode != loadalloca)
			return true;
	}

	return false;
}

/*
	a1 sa1 = alloca_op ...
	a2 sa2 = alloca_op ...
	ss1 = store_op a1 ... sa1
	ss2 = store_op a2 ... sa2
	... = load_op a1 ss1 ss2
	=>
	...
	... = load_op a1 ss1
*/
static bool
is_load_store_state_reducible(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto address = operands[0];

	if (operands.size() == 2)
		return false;

	auto allocanode = jive::node_output::node(address);
	if (!is<alloca_op>(allocanode))
		return false;

	size_t redstates = 0;
	for (size_t n = 1; n < operands.size(); n++) {
		auto state = operands[n];
		if (is_reducible_state(state, allocanode))
			redstates++;
	}

	return redstates == op.NumStates() || redstates == 0 ? false : true;
}

/*
	v so1 so2 so3 = load_op a si1 si1 si1
	=>
	v so1 = load_op a si1
*/
static bool
is_multiple_origin_reducible(const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> states(std::next(operands.begin()), operands.end());
	return states.size() != operands.size()-1;
}

static bool
is_load_store_reducible(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	JLM_ASSERT(operands.size() > 1);

	auto storenode = jive::node_output::node(operands[1]);
	if (!is<StoreOperation>(storenode))
		return false;

	auto sop = static_cast<const StoreOperation*>(&storenode->operation());
	if (sop->NumStates() != op.NumStates())
		return false;

	/* check for same address */
	if (operands[0] != storenode->input(0)->origin())
		return false;

	for (size_t n = 1; n < operands.size(); n++) {
		if (jive::node_output::node(operands[n]) != storenode)
			return false;
	}

	JLM_ASSERT(op.GetAlignment() == sop->GetAlignment());
	return true;
}

static std::vector<jive::output*>
perform_load_store_reduction(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto storenode = jive::node_output::node(operands[1]);

	std::vector<jive::output*> results(1, storenode->input(1)->origin());
	results.insert(results.end(), std::next(operands.begin()), operands.end());

	return results;
}

static std::vector<jive::output*>
perform_load_store_alloca_reduction(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto allocanode = jive::node_output::node(operands[0]);
	auto storenode = jive::node_output::node(operands[1]);

	return {storenode->input(1)->origin(), allocanode->output(1)};
}

static std::vector<jive::output*>
perform_load_mux_reduction(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto memStateMergeNode = jive::node_output::node(operands[1]);

	auto ld = LoadNode::Create(
    operands[0],
    jive::operands(memStateMergeNode),
    op.GetLoadedType(),
    op.GetAlignment());

	std::vector<jive::output*> states = {std::next(ld.begin()), ld.end()};
	auto mx = MemStateMergeOperator::Create(states);

	return {ld[0], mx};
}

static std::vector<jive::output*>
perform_load_alloca_reduction(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto allocanode = jive::node_output::node(operands[0]);

	std::vector<jive::output*> loadstates;
	std::vector<jive::output*> otherstates;
	for (size_t n = 1; n < operands.size(); n++) {
		auto node = jive::node_output::node(operands[n]);
		if (!is<alloca_op>(node) || node == allocanode)
			loadstates.push_back(operands[n]);
		else
			otherstates.push_back(operands[n]);
	}

	auto ld = LoadNode::Create(
    operands[0],
    loadstates,
    op.GetLoadedType(),
    op.GetAlignment());

	std::vector<jive::output*> results(1, ld[0]);
	results.insert(results.end(), std::next(ld.begin()), ld.end());
	results.insert(results.end(), otherstates.begin(), otherstates.end());
	return results;
}

static std::vector<jive::output*>
perform_load_store_state_reduction(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	auto address = operands[0];
	auto allocanode = jive::node_output::node(address);

	std::vector<jive::output*> new_loadstates;
	std::vector<jive::output*> results(operands.size(), nullptr);
	for (size_t n = 1; n < operands.size(); n++) {
		auto state = operands[n];
		if (is_reducible_state(state, allocanode))
			results[n] = state;
		else new_loadstates.push_back(state);
	}

	auto ld = LoadNode::Create(
    operands[0],
    new_loadstates,
    op.GetLoadedType(),
    op.GetAlignment());

	results[0] = ld[0];
	for (size_t n = 1, s = 1; n < results.size(); n++) {
		if (results[n] == nullptr)
			results[n] = ld[s++];
	}

	return results;
}

static std::vector<jive::output*>
perform_multiple_origin_reduction(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	std::vector<jive::output*> new_loadstates;
	std::unordered_set<jive::output*> seen_state;
	std::vector<jive::output*> results(operands.size(), nullptr);
	for (size_t n = 1; n < operands.size(); n++) {
		auto state = operands[n];
		if (seen_state.find(state) != seen_state.end())
			results[n] = state;
		else
			new_loadstates.push_back(state);

		seen_state.insert(state);
	}

	auto ld = LoadNode::Create(
    operands[0],
    new_loadstates,
    op.GetLoadedType(),
    op.GetAlignment());

	results[0] = ld[0];
	for (size_t n = 1, s = 1; n < results.size(); n++) {
		if (results[n] == nullptr)
			results[n] = ld[s++];
	}

	return results;
}

/*
	_ so1 = load_op _ si1
	_ so2 = load_op _ so1
	_ so3 = load_op _ so2
	=>
	_ so1 = load_op _ si1
	_ so2 = load_op _ si1
	_ so3 = load_op _ si1
*/
static bool
is_load_load_state_reducible(const std::vector<jive::output*> & operands)
{
	JLM_ASSERT(operands.size() >= 2);

	for (size_t n = 1; n < operands.size(); n++) {
		if (is<LoadOperation>(jive::node_output::node(operands[n])))
			return true;
	}

	return false;
}

static std::vector<jive::output*>
perform_load_load_state_reduction(
	const LoadOperation & op,
	const std::vector<jive::output*> & operands)
{
	size_t nstates = operands.size()-1;

	auto load_state_input = [](jive::output * result)
	{
		auto ld = jive::node_output::node(result);
		JLM_ASSERT(is<LoadOperation>(ld));

		/*
			FIXME: This function returns the corresponding state input for a state output of a load
			node. It should be part of a load node class.
		*/
		for (size_t n = 1; n < ld->noutputs(); n++) {
			if (result == ld->output(n))
				return ld->input(n);
		}

		JLM_UNREACHABLE("This should have never happened!");
	};

	std::function<jive::output*(size_t, jive::output*, std::vector<std::vector<jive::output*>>&)>
	reduce_state = [&](size_t index, jive::output * operand, auto & mxstates)
	{
		JLM_ASSERT(jive::is<jive::statetype>(operand->type()));

		if (!is<LoadOperation>(jive::node_output::node(operand)))
			return operand;

		mxstates[index].push_back(operand);
		return reduce_state(index, load_state_input(operand)->origin(), mxstates);
	};

	std::vector<jive::output*> ldstates;
	std::vector<std::vector<jive::output*>> mxstates(nstates);
	for (size_t n = 1; n < operands.size(); n++)
		ldstates.push_back(reduce_state(n-1, operands[n], mxstates));

	auto ld = LoadNode::Create(
    operands[0],
    ldstates,
    op.GetLoadedType(),
    op.GetAlignment());
	for (size_t n = 0; n < mxstates.size(); n++) {
		auto & states = mxstates[n];
		if (!states.empty()) {
			states.push_back(ld[n+1]);
			ld[n+1] = MemStateMergeOperator::Create(states);
		}
	}

	return ld;
}

load_normal_form::~load_normal_form()
{}

load_normal_form::load_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: simple_normal_form(opclass, parent, graph)
, enable_load_mux_(false)
, enable_load_store_(false)
, enable_load_alloca_(false)
, enable_load_load_state_(false)
, enable_multiple_origin_(false)
, enable_load_store_state_(false)
, enable_load_store_alloca_(false)
{}

bool
load_normal_form::normalize_node(jive::node * node) const
{
	JLM_ASSERT(is<LoadOperation>(node->operation()));
	auto op = static_cast<const LoadOperation*>(&node->operation());
	auto operands = jive::operands(node);

	if (!get_mutable())
		return true;

	if (get_load_mux_reducible() && is_load_mux_reducible(operands)) {
		divert_users(node, perform_load_mux_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_load_store_reducible() && is_load_store_reducible(*op, operands)) {
		divert_users(node, perform_load_store_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_load_alloca_reducible() && is_load_alloca_reducible(operands)) {
		divert_users(node, perform_load_alloca_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_load_store_state_reducible() && is_load_store_state_reducible(*op, operands)) {
		divert_users(node, perform_load_store_state_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands)) {
		divert_users(node, perform_multiple_origin_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_load_store_alloca_reducible() && is_load_store_alloca_reducible(operands)) {
		divert_users(node, perform_load_store_alloca_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_load_load_state_reducible() && is_load_load_state_reducible(operands)) {
		divert_users(node, perform_load_load_state_reduction(*op, operands));
		remove(node);
		return false;
	}

	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
load_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & operands) const
{
	JLM_ASSERT(is<LoadOperation>(op));
	auto lop = static_cast<const LoadOperation*>(&op);

	if (!get_mutable())
		return simple_normal_form::normalized_create(region, op, operands);

	if (get_load_mux_reducible() && is_load_mux_reducible(operands))
		return perform_load_mux_reduction(*lop, operands);

	if (get_load_store_reducible() && is_load_store_reducible(*lop, operands))
		return perform_load_store_reduction(*lop, operands);

	if (get_load_alloca_reducible() && is_load_alloca_reducible(operands))
		return perform_load_alloca_reduction(*lop, operands);

	if (get_load_store_state_reducible() && is_load_store_state_reducible(*lop, operands))
		return perform_load_store_state_reduction(*lop, operands);

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands))
		return perform_multiple_origin_reduction(*lop, operands);

	if (get_load_store_alloca_reducible() && is_load_store_alloca_reducible(operands))
		return perform_load_store_alloca_reduction(*lop, operands);

	if (get_load_load_state_reducible() && is_load_load_state_reducible(operands))
		return perform_load_load_state_reduction(*lop, operands);

	return simple_normal_form::normalized_create(region, op, operands);
}

}

namespace {

static jive::node_normal_form *
create_load_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jlm::load_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_normal_form()
{
	jive::node_normal_form::register_factory(typeid(jlm::LoadOperation), create_load_normal_form);
}

}
