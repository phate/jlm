/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/statemux.h>

#include <jlm/ir/operators/alloca.hpp>
#include <jlm/ir/operators/store.hpp>

namespace jlm {

/* store operator */

store_op::~store_op() noexcept
{}

bool
store_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const store_op*>(&other);
	return op
	    && op->nstates_ == nstates_
	    && op->aport_ == aport_
	    && op->vport_ == vport_
	    && op->alignment_ == alignment_;
}

size_t
store_op::narguments() const noexcept
{
	return 2 + nstates();
}

const jive::port &
store_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return aport_;

	if (index == 1)
		return vport_;

	static const jive::port p(jive::memtype::instance());
	return p;
}

size_t
store_op::nresults() const noexcept
{
	return nstates();
}

const jive::port &
store_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	static const jive::port p(jive::memtype::instance());
	return p;
}

std::string
store_op::debug_string() const
{
	return "STORE";
}

std::unique_ptr<jive::operation>
store_op::copy() const
{
	return std::unique_ptr<jive::operation>(new store_op(*this));
}

/* store normal form */

static bool
is_store_mux_reducible(const std::vector<jive::output*> & operands)
{
	JLM_DEBUG_ASSERT(operands.size() > 2);

	auto muxnode = operands[2]->node();
	if (!muxnode || !is_mux_op(muxnode->operation()))
		return false;

	for (size_t n = 2; n < operands.size(); n++) {
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::memtype*>(&operands[n]->type()));
		if (operands[n]->node() && operands[n]->node() != muxnode)
			return false;
	}

	return true;
}

static bool
is_store_store_reducible(
	const store_op & op,
	const std::vector<jive::output*> & operands)
{
	JLM_DEBUG_ASSERT(operands.size() > 2);

	auto storenode = operands[2]->node();
	if (!is_store_node(storenode))
		return false;

	if (op.nstates() != storenode->noutputs())
		return false;

	/* check for same address */
	if (operands[0] != storenode->input(0)->origin())
		return false;

	for (size_t n = 2; n < operands.size(); n++) {
		if (operands[n]->node() != storenode || operands[n]->nusers() != 1)
			return false;
	}

	auto other = static_cast<const store_op*>(&storenode->operation());
	JLM_DEBUG_ASSERT(op.alignment() == other->alignment());
	return true;
}

static bool
is_store_alloca_reducible(const std::vector<jive::output*> & operands)
{
	if (operands.size() == 3)
		return false;

	auto alloca = operands[0]->node();
	if (!alloca || !is_alloca_op(alloca->operation()))
		return false;

	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());
	if (states.find(alloca->output(1)) == states.end())
		return false;

	if (alloca->output(1)->nusers() != 1)
		return false;

	return true;
}

static bool
is_multiple_origin_reducible(const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());
	return states.size() != operands.size()-2;
}

static std::vector<jive::output*>
perform_store_mux_reduction(
	const jlm::store_op & op,
	const std::vector<jive::output*> & operands)
{
	auto muxnode = operands[2]->node();
	auto states = create_store(operands[0], operands[1], jive::operands(muxnode), op.alignment());
	return jive::create_state_mux(muxnode->input(0)->type(), states, op.nstates());
}

static std::vector<jive::output*>
perform_store_store_reduction(
	const jlm::store_op & op,
	const std::vector<jive::output*> & operands)
{
	JLM_DEBUG_ASSERT(is_store_store_reducible(op, operands));
	auto storenode = operands[2]->node();

	auto storeops = jive::operands(storenode);
	std::vector<jive::output*> states(std::next(std::next(storeops.begin())), storeops.end());
	return jlm::create_store(operands[0], operands[1], states, op.alignment());
}

static std::vector<jive::output*>
perform_store_alloca_reduction(
	const jlm::store_op & op,
	const std::vector<jive::output*> & operands)
{
	auto value = operands[1];
	auto address = operands[0];
	auto alloca_state = address->node()->output(1);
	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());

	auto outputs = create_store(address, value, {alloca_state}, op.alignment());
	states.erase(alloca_state);
	states.insert(outputs[0]);
	return {states.begin(), states.end()};
}

static std::vector<jive::output*>
perform_multiple_origin_reduction(
	const jlm::store_op & op,
	const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> states(std::next(std::next(operands.begin())), operands.end());
	return jlm::create_store(operands[0], operands[1], {states.begin(), states.end()},
		op.alignment());
}

store_normal_form::~store_normal_form()
{}

store_normal_form::store_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: simple_normal_form(opclass, parent, graph)
, enable_store_mux_(false)
, enable_store_store_(false)
, enable_store_alloca_(false)
, enable_multiple_origin_(false)
{
	if (auto p = dynamic_cast<const store_normal_form*>(parent)) {
		enable_multiple_origin_ = p->enable_multiple_origin_;
		enable_store_store_ = p->enable_store_store_;
		enable_store_mux_ = p->enable_store_mux_;
	}
}

bool
store_normal_form::normalize_node(jive::node * node) const
{
	JLM_DEBUG_ASSERT(is_store_op(node->operation()));
	auto op = static_cast<const jlm::store_op*>(&node->operation());
	auto operands = jive::operands(node);

	if (!get_mutable())
		return true;

	if (get_store_mux_reducible() && is_store_mux_reducible(operands)) {
		replace(node, perform_store_mux_reduction(*op, operands));
		node->region()->remove_node(node);
		return false;
	}

	if (get_store_store_reducible() && is_store_store_reducible(*op, operands)) {
		replace(node, perform_store_store_reduction(*op, operands));
		remove(node);
		return false;
	}

	if (get_store_alloca_reducible() && is_store_alloca_reducible(operands)) {
		replace(node, perform_store_alloca_reduction(*op, operands));
		node->region()->remove_node(node);
		return false;
	}

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands)) {
		auto outputs = perform_multiple_origin_reduction(*op, operands);
		auto new_node = outputs[0]->node();

		std::unordered_map<jive::output*, jive::output*> origin2output;
		for (size_t n = 0; n < outputs.size(); n++) {
			auto origin = new_node->input(n+2)->origin();
			JLM_DEBUG_ASSERT(origin2output.find(origin) == origin2output.end());
			origin2output[origin] = outputs[n];
		}

		for (size_t n = 2; n < node->ninputs(); n++) {
			auto origin = node->input(n)->origin();
			JLM_DEBUG_ASSERT(origin2output.find(origin) != origin2output.end());
			node->output(n-2)->replace(origin2output[origin]);
		}
		remove(node);
		return false;
	}

	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
store_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & ops) const
{
	JLM_DEBUG_ASSERT(is_store_op(op));
	auto sop = static_cast<const jlm::store_op*>(&op);

	if (!get_mutable())
		return simple_normal_form::normalized_create(region, op, ops);

	auto operands = ops;
	if (get_store_mux_reducible() && is_store_mux_reducible(operands))
		return perform_store_mux_reduction(*sop, operands);

	if (get_store_alloca_reducible() && is_store_alloca_reducible(operands))
		return perform_store_alloca_reduction(*sop, operands);

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands))
		return perform_multiple_origin_reduction(*sop, operands);

	return simple_normal_form::normalized_create(region, op, operands);
}

void
store_normal_form::set_store_mux_reducible(bool enable)
{
	if (get_store_mux_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_store_mux_reducible>(enable);

	enable_store_mux_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
store_normal_form::set_store_store_reducible(bool enable)
{
	if (get_store_store_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_store_store_reducible>(enable);

	enable_store_store_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
store_normal_form::set_store_alloca_reducible(bool enable)
{
	if (get_store_alloca_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_store_alloca_reducible>(enable);

	enable_store_alloca_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
store_normal_form::set_multiple_origin_reducible(bool enable)
{
	if (get_multiple_origin_reducible() == enable)
		return;

	children_set<store_normal_form, &store_normal_form::set_multiple_origin_reducible>(enable);

	enable_multiple_origin_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

}

namespace {

static jive::node_normal_form *
create_store_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jlm::store_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_normal_form()
{
	jive::node_normal_form::register_factory(typeid(jlm::store_op), create_store_normal_form);
}

}
