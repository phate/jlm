/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/statemux.hpp>

namespace jive {

/* mux operator */

mux_op::~mux_op() noexcept {}

bool
mux_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const mux_op*>(&other);
	return op
	    && op->narguments() == narguments()
	    && op->nresults() == nresults()
	    && op->result(0) == result(0);
}

std::string
mux_op::debug_string() const
{
	return "STATEMUX";
}

std::unique_ptr<jive::operation>
mux_op::copy() const
{
	return std::unique_ptr<jive::operation>(new mux_op(*this));
}

/* mux normal form */

static jive::node *
is_mux_mux_reducible(const std::vector<jive::output*> & ops)
{
	std::unordered_set<jive::output*> operands(ops.begin(), ops.end());

	for (const auto & operand : operands) {
		auto node = node_output::node(operand);
		if (!node || !is_mux_op(node->operation()))
			continue;

		size_t n;
		for (n = 0; n < node->noutputs(); n++) {
			auto output = node->output(n);
			if (operands.find(output) == operands.end() || output->nusers() != 1)
				break;
		}
		if (n == node->noutputs())
			return node;
	}

	return nullptr;
}

static bool
is_multiple_origin_reducible(const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> set(operands.begin(), operands.end());
	return set.size() != operands.size();
}

static std::vector<jive::output*>
perform_multiple_origin_reduction(
	const jive::mux_op & op,
	const std::vector<jive::output*> & operands)
{
	std::unordered_set<jive::output*> set(operands.begin(), operands.end());
	return create_state_mux(op.result(0).type(), {set.begin(), set.end()}, op.nresults());
}

static std::vector<jive::output*>
perform_mux_mux_reduction(
	const jive::mux_op & op,
	const jive::node * muxnode,
	const std::vector<jive::output*> & old_operands)
{
	JIVE_DEBUG_ASSERT(is_mux_op(muxnode->operation()));

	bool reduced = false;
	std::vector<jive::output*> new_operands;
	for (const auto & operand : old_operands) {
		if (jive::node_output::node(operand) == muxnode && !reduced) {
			reduced = true;
			auto tmp = operands(muxnode);
			new_operands.insert(new_operands.end(), tmp.begin(), tmp.end());
			continue;
		}

		if (jive::node_output::node(operand) != muxnode)
			new_operands.push_back(operand);
	}

	return create_state_mux(op.result(0).type(), new_operands, op.nresults());
}

mux_normal_form::~mux_normal_form() noexcept
{}

mux_normal_form::mux_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: simple_normal_form(opclass, parent, graph)
, enable_mux_mux_(false)
, enable_multiple_origin_(false)
{
	if (auto p = dynamic_cast<const mux_normal_form*>(parent))
		enable_mux_mux_ = p->enable_mux_mux_;
}

bool
mux_normal_form::normalize_node(jive::node * node) const
{
	JIVE_DEBUG_ASSERT(dynamic_cast<const jive::mux_op*>(&node->operation()));
	auto op = static_cast<const jive::mux_op*>(&node->operation());

	if (!get_mutable())
		return true;

	auto muxnode = is_mux_mux_reducible(operands(node));
	if (get_mux_mux_reducible() && muxnode) {
		divert_users(node, perform_mux_mux_reduction(*op, muxnode, operands(node)));
		remove(node);
		return false;
	}

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands(node))) {
		divert_users(node, perform_multiple_origin_reduction(*op, operands(node)));
		remove(node);
		return false;
	}

	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
mux_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & operands) const
{
	JIVE_DEBUG_ASSERT(dynamic_cast<const jive::mux_op*>(&op));
	auto mop = static_cast<const jive::mux_op*>(&op);

	if (!get_mutable())
		return simple_normal_form::normalized_create(region, op, operands);

	auto muxnode = is_mux_mux_reducible(operands);
	if (get_mux_mux_reducible() && muxnode)
		return perform_mux_mux_reduction(*mop, muxnode, operands);

	if (get_multiple_origin_reducible() && is_multiple_origin_reducible(operands))
		return perform_multiple_origin_reduction(*mop, operands);

	return simple_normal_form::normalized_create(region, op, operands);
}

void
mux_normal_form::set_mux_mux_reducible(bool enable)
{
	if (get_mux_mux_reducible() == enable)
		return;

	children_set<mux_normal_form, &mux_normal_form::set_mux_mux_reducible>(enable);

	enable_mux_mux_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

void
mux_normal_form::set_multiple_origin_reducible(bool enable)
{
	if (get_multiple_origin_reducible() == enable)
		return;

	children_set<mux_normal_form, &mux_normal_form::set_multiple_origin_reducible>(enable);

	enable_multiple_origin_ = enable;
	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

}

namespace {

static jive::node_normal_form *
create_mux_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jive::mux_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
	jive::node_normal_form::register_factory(typeid(jive::mux_op), create_mux_normal_form);
}

}
