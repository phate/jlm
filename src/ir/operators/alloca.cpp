/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/vsdg/graph.h>

#include <jlm/ir/operators/alloca.hpp>

namespace jlm {

/* alloca operator */

alloca_op::~alloca_op() noexcept
{}

bool
alloca_op::operator==(const operation & other) const noexcept
{
	/* Avoid CNE for alloca operators */
	return false;
}

size_t
alloca_op::narguments() const noexcept
{
	return 2;
}

const jive::port &
alloca_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return bport_;

	static const jive::port p(jive::mem::type::instance());
	return p;
}

size_t
alloca_op::nresults() const noexcept
{
	return 2;
}

const jive::port &
alloca_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	if (index == 0)
		return aport_;

	static const jive::port p(jive::mem::type::instance());
	return p;
}

std::string
alloca_op::debug_string() const
{
	return "ALLOCA";
}

std::unique_ptr<jive::operation>
alloca_op::copy() const
{
	return std::unique_ptr<jive::operation>(new alloca_op(*this));
}

/* alloca normal form */

static bool
is_alloca_alloca_reducible(const jive::output * state)
{
	return state->node()
	    && is_alloca_op(state->node()->operation())
	    && state->nusers() == 1;
}

static bool
is_alloca_mux_reducible(const jive::output * state)
{
	auto muxnode = state->node();
	if (!muxnode)
		return false;

	std::vector<jive::node*> allocas;
	for (size_t n = 0; n < muxnode->ninputs(); n++) {
		auto node = muxnode->input(n)->origin()->node();
		if (!node || !is_alloca_op(node->operation()))
			return false;
		allocas.push_back(node);
	}
	JLM_DEBUG_ASSERT(!allocas.empty());

	state = allocas[0]->input(1)->origin();
	for (const auto & node : allocas) {
		if (node->input(1)->origin() != state)
			return false;
	}

	return true;
}

static std::vector<jive::output*>
perform_alloca_alloca_reduction(
	const jlm::alloca_op & op,
	jive::output * size,
	jive::output * state)
{
	JLM_DEBUG_ASSERT(is_alloca_op(state->node()->operation()));
	auto origin = state->node()->input(1)->origin();
	auto type = static_cast<const jive::state::type*>(&origin->type());

	auto ops = jive::create_normalized(size->region(), op, {size, origin});
	state = jive::create_state_merge(*type, {state, ops[1]});

	return {ops[0], state};
}

static std::vector<jive::output*>
perform_alloca_mux_reduction(
	const jlm::alloca_op & op,
	jive::output * size,
	jive::output * state)
{
	auto muxnode = state->node();
	auto allocanode = muxnode->input(0)->origin()->node();
	auto type = static_cast<const jive::state::type*>(&state->type());

	auto ops = jive::create_normalized(size->region(), op, {size, allocanode->input(1)->origin()});
	state = jive::create_state_merge(*type, {state, ops[1]});

	return {ops[0], state};
}

alloca_normal_form::~alloca_normal_form()
{}

alloca_normal_form::alloca_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: simple_normal_form(opclass, parent, graph)
, enable_alloca_mux_(false)
, enable_alloca_alloca_(false)
{
	if (auto p = dynamic_cast<const alloca_normal_form*>(parent)) {
		enable_alloca_mux_ = p->enable_alloca_mux_;
		enable_alloca_alloca_ = p->enable_alloca_alloca_;
	}
}

bool
alloca_normal_form::normalize_node(jive::node * node) const
{
	JLM_DEBUG_ASSERT(is_alloca_op(node->operation()));
	auto op = static_cast<const jlm::alloca_op*>(&node->operation());

	if (!get_mutable())
		return true;

	auto size = node->input(0)->origin();
	auto state = node->input(1)->origin();
	if (get_alloca_alloca_reducible() && is_alloca_alloca_reducible(state)) {
		auto outputs = perform_alloca_alloca_reduction(*op, size, state);
		node->output(0)->replace(outputs[0]);
		node->output(1)->replace(outputs[1]);
		return false;
	}

	if (get_alloca_mux_reducible() && is_alloca_mux_reducible(state)) {
		auto outputs = perform_alloca_mux_reduction(*op, size, state);
		node->output(0)->replace(outputs[0]);
		node->output(1)->replace(outputs[1]);
		return false;
	}

	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
alloca_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & ops) const
{
	JLM_DEBUG_ASSERT(ops.size() == 2);
	JLM_DEBUG_ASSERT(is_alloca_op(op));
	auto aop = static_cast<const jlm::alloca_op*>(&op);

	if (!get_mutable())
		return simple_normal_form::normalized_create(region, op, ops);

	std::vector<jive::output*> operands = ops;
	if (get_alloca_alloca_reducible() && is_alloca_alloca_reducible(operands[1]))
		operands = perform_alloca_alloca_reduction(*aop, operands[0], operands[1]);

	if (get_alloca_mux_reducible() && is_alloca_alloca_reducible(operands[1]))
		operands = perform_alloca_mux_reduction(*aop, operands[0], operands[1]);

	return simple_normal_form::normalized_create(region, op, operands);
}

void
alloca_normal_form::set_alloca_mux_reducible(bool enable)
{
	if (get_alloca_mux_reducible() == enable)
		return;

	children_set<alloca_normal_form, &alloca_normal_form::set_alloca_mux_reducible>(enable);

	enable_alloca_mux_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
alloca_normal_form::set_alloca_alloca_reducible(bool enable)
{
	if (get_alloca_alloca_reducible() == enable)
		return;

	children_set<alloca_normal_form, &alloca_normal_form::set_alloca_alloca_reducible>(enable);

	enable_alloca_alloca_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

}

namespace {

static jive::node_normal_form *
create_alloca_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jlm::alloca_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_normal_form()
{
	jive::node_normal_form::register_factory(typeid(jlm::alloca_op), create_alloca_normal_form);
}

}
