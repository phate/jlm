/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/memorytype.h>
#include <jive/vsdg/statemux.h>

#include <jlm/ir/operators/load.hpp>

namespace jlm {

/* load operator */

load_op::~load_op() noexcept
{}

bool
load_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const load_op*>(&other);
	return op
	    && op->nstates_ == nstates_
	    && op->aport_ == aport_
	    && op->vport_ == vport_
	    && op->alignment_ == alignment_;
}

size_t
load_op::narguments() const noexcept
{
	return 1 + nstates();
}

const jive::port &
load_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return aport_;

	static const jive::port p(jive::mem::type::instance());
	return p;
}

size_t
load_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
load_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return vport_;
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
		JLM_DEBUG_ASSERT(dynamic_cast<const jive::mem::type*>(&operands[n]->type()));
		if (operands[n]->node() && operands[n]->node() != muxnode)
			return false;
	}

	return true;
}

static std::vector<jive::output*>
perform_load_mux_reduction(
	const jlm::load_op & op,
	const std::vector<jive::output*> & operands)
{
	auto muxnode = operands[1]->node();
	return {create_load(operands[0], jive::operands(muxnode), op.alignment())};
}

load_normal_form::~load_normal_form()
{}

load_normal_form::load_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: simple_normal_form(opclass, parent, graph)
, enable_load_mux_(false)
{}

bool
load_normal_form::normalize_node(jive::node * node) const
{
	JLM_DEBUG_ASSERT(is_load_op(node->operation()));
	auto op = static_cast<const jlm::load_op*>(&node->operation());
	auto operands = jive::operands(node);

	if (!get_mutable())
		return true;

	if (get_load_mux_reducible() && is_load_mux_reducible(operands)) {
		replace(node, perform_load_mux_reduction(*op, operands));
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
	JLM_DEBUG_ASSERT(is_load_op(op));
	auto lop = static_cast<const jlm::load_op*>(&op);

	if (!get_mutable())
		return simple_normal_form::normalized_create(region, op, operands);

	if (get_load_mux_reducible() && is_load_mux_reducible(operands))
		return perform_load_mux_reduction(*lop, operands);

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
