/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/addresstype.h>
#include <jive/rvsdg/statemux.h>

#include <jlm/ir/operators/alloca.hpp>
#include <jlm/ir/operators/load.hpp>
#include <jlm/ir/operators/store.hpp>

namespace jlm {

/* load operator */

load_op::~load_op() noexcept
{}

bool
load_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const load_op*>(&other);
	if (!op || op->narguments() != narguments())
		return false;

	return op->argument(0) == argument(0) && op->alignment() == alignment();
}

std::string
load_op::debug_string() const
{
	return "LOAD";
}

std::unique_ptr<jive::operation>
load_op::copy() const
{
	return std::unique_ptr<jive::operation>(new load_op(*this));
}

/* load normal form */

static bool
is_load_mux_reducible(const std::vector<jive::output*> & operands)
{
	auto muxnode = operands[1]->node();
	if (!muxnode || !is_mux_op(muxnode->operation()))
		return false;

	for (size_t n = 1; n < operands.size(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::memtype*>(&operands[n]->type()));
		if (operands[n]->node() && operands[n]->node() != muxnode)
			return false;
	}

	return true;
}

static std::vector<jive::output*>
is_load_alloca_reducible(const std::vector<jive::output*> & operands)
{
	auto address = operands[0];

	auto allocanode = address->node();
	if (!allocanode || !is_alloca_op(allocanode->operation()))
		return {std::next(operands.begin()), operands.end()};

	std::vector<jive::output*> new_states;
	for (size_t n = 1; n < operands.size(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::memtype*>(&operands[n]->type()));
		auto node = operands[n]->node();
		if (node && is_alloca_op(node->operation()) && node != allocanode)
			continue;

		new_states.push_back(operands[n]);
	}

	JLM_DEBUG_ASSERT(!new_states.empty());
	return new_states;
}

static bool
is_load_store_alloca_reducible(const std::vector<jive::output*> & operands)
{
	if (operands.size() != 2)
		return false;

	auto address = operands[0];
	auto state = operands[1];

	auto alloca = address->node();
	if (!alloca || !is_alloca_op(alloca->operation()))
		return false;

	auto store = state->node();
	if (!store || !is<store_op>(store->operation()))
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

static std::vector<jive::output*>
is_load_store_state_reducible(const std::vector<jive::output*> & operands)
{
	auto address = operands[0];

	if (operands.size() == 2)
		return {operands[1]};

	auto allocanode = address->node();
	if (!allocanode || !is_alloca_op(allocanode->operation()))
		return {std::next(operands.begin()), operands.end()};

	std::vector<jive::output*> new_states;
	for (size_t n = 1; n < operands.size(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::memtype*>(&operands[n]->type()));
		auto node = operands[n]->node();
		if (node && is<store_op>(node->operation())) {
			auto addressnode = node->input(0)->origin()->node();
			if (addressnode && is_alloca_op(addressnode->operation()) && addressnode != allocanode)
				continue;
		}

		new_states.push_back(operands[n]);
	}

	if (new_states.empty())
		return {std::next(operands.begin()), operands.end()};

	return new_states;
}

static bool
is_multiple_origin_reducible(const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> states(std::next(operands.begin()), operands.end());
	return states.size() != operands.size()-1;
}

static bool
is_load_store_reducible(
	const load_op & op,
	const std::vector<jive::output*> & operands)
{
	JLM_DEBUG_ASSERT(operands.size() > 1);

	auto storenode = operands[1]->node();
	if (!is<store_op>(storenode))
		return false;

	auto sop = static_cast<const store_op*>(&storenode->operation());
	if (sop->nstates() != op.nstates())
		return false;

	/* check for same address */
	if (operands[0] != storenode->input(0)->origin())
		return false;

	for (size_t n = 1; n < operands.size(); n++) {
		if (operands[n]->node() != storenode)
			return false;
	}

	JLM_DEBUG_ASSERT(op.alignment() == sop->alignment());
	return true;
}

static std::vector<jive::output*>
perform_load_store_reduction(
	const jlm::load_op & op,
	const std::vector<jive::output*> & operands)
{
	JLM_DEBUG_ASSERT(is_load_store_reducible(op, operands));
	auto storenode = operands[1]->node();

	return {storenode->input(1)->origin()};
}

static std::vector<jive::output*>
perform_load_store_alloca_reduction(
	const jlm::load_op & op,
	const std::vector<jive::output*> & operands)
{
	return {operands[1]->node()->input(1)->origin()};
}

static std::vector<jive::output*>
perform_load_mux_reduction(
	const jlm::load_op & op,
	const std::vector<jive::output*> & operands)
{
	auto muxnode = operands[1]->node();
	return {create_load(operands[0], jive::operands(muxnode), op.alignment())};
}

static std::vector<jive::output*>
perform_load_alloca_reduction(
	const jlm::load_op & op,
	const std::vector<jive::output*> & operands,
	const std::vector<jive::output*> & new_states)
{
	JLM_DEBUG_ASSERT(!new_states.empty());
	return {create_load(operands[0], new_states, op.alignment())};
}

static std::vector<jive::output*>
perform_load_store_state_reduction(
	const jlm::load_op & op,
	const std::vector<jive::output*> & operands,
	const std::vector<jive::output*> & new_states)
{
	JLM_DEBUG_ASSERT(!new_states.empty());
	return {create_load(operands[0], new_states, op.alignment())};
}

static std::vector<jive::output*>
perform_multiple_origin_reduction(
	const jlm::load_op & op,
	const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> states(std::next(operands.begin()), operands.end());
	return {create_load(operands[0], {states.begin(), states.end()}, op.alignment())};
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
, enable_multiple_origin_(false)
, enable_load_store_state_(false)
, enable_load_store_alloca_(false)
{}

bool
load_normal_form::normalize_node(jive::node * node) const
{
	JLM_DEBUG_ASSERT(is<load_op>(node->operation()));
	auto op = static_cast<const jlm::load_op*>(&node->operation());
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

	auto new_states = is_load_alloca_reducible(operands);
	if (get_load_alloca_reducible() && new_states.size() != operands.size()-1) {
		divert_users(node, perform_load_alloca_reduction(*op, operands, new_states));
		remove(node);
		return false;
	}

	new_states = is_load_store_state_reducible(operands);
	if (get_load_store_state_reducible() && new_states.size() != operands.size()-1) {
		divert_users(node, perform_load_store_state_reduction(*op, operands, new_states));
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

	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
load_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & operands) const
{
	JLM_DEBUG_ASSERT(is<load_op>(op));
	auto lop = static_cast<const jlm::load_op*>(&op);

	if (!get_mutable())
		return simple_normal_form::normalized_create(region, op, operands);

	if (get_load_mux_reducible() && is_load_mux_reducible(operands))
		return perform_load_mux_reduction(*lop, operands);

	if (get_load_store_reducible() && is_load_store_reducible(*lop, operands))
		return perform_load_store_reduction(*lop, operands);

	auto new_states = is_load_alloca_reducible(operands);
	if (get_load_alloca_reducible() && new_states.size() != operands.size()-1)
		return perform_load_alloca_reduction(*lop, operands, new_states);

	new_states = is_load_store_state_reducible(operands);
	if (get_load_store_state_reducible() && new_states.size() != operands.size()-1)
		return perform_load_store_state_reduction(*lop, operands, new_states);

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands))
		return perform_multiple_origin_reduction(*lop, operands);

	if (get_load_store_alloca_reducible() && is_load_store_alloca_reducible(operands))
		return perform_load_store_alloca_reduction(*lop, operands);

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
	jive::node_normal_form::register_factory(typeid(jlm::load_op), create_load_normal_form);
}

}
