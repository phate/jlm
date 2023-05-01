/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>

static jive::node *
node_cse(
	jive::region * region,
	const jive::operation & op,
	const std::vector<jive::output*> & arguments)
{
	auto cse_test = [&](const jive::node * node)
	{
		return node->operation() == op && arguments == jive::operands(node);
	};

	if (!arguments.empty()) {
		for (const auto & user : *arguments[0]) {
			if (!jive::is<jive::node_input>(*user))
				continue;

			auto node = static_cast<jive::node_input*>(user)->node();
			if (cse_test(node))
				return node;
		}
	} else {
		for (auto & node : region->top_nodes) {
			if (cse_test(&node))
				return &node;
		}
	}

	return nullptr;
}

namespace jive {

simple_normal_form::~simple_normal_form() noexcept
{}

simple_normal_form::simple_normal_form(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
	: node_normal_form(operator_class, parent, graph)
	, enable_cse_(true)
{
	if (auto p = dynamic_cast<simple_normal_form*>(parent))
		enable_cse_ = p->get_cse();
}

bool
simple_normal_form::normalize_node(jive::node * node) const
{
	if (!get_mutable())
		return true;

	if (get_cse()) {
		auto new_node = node_cse(node->region(), node->operation(), operands(node));
		JIVE_DEBUG_ASSERT(new_node);
		if (new_node != node) {
			divert_users(node, outputs(new_node));
			remove(node);
			return false;
		}
	}

	return true;
}

std::vector<jive::output*>
simple_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & arguments) const
{
	jive::node * node = nullptr;
	if (get_mutable() && get_cse())
		node = node_cse(region, op, arguments);
	if (!node)
		node = simple_node::create(region, op, arguments);

	return outputs(node);
}

void
simple_normal_form::set_cse(bool enable)
{
	if (enable == enable_cse_)
		return;

	enable_cse_ = enable;
	children_set<simple_normal_form, &simple_normal_form::set_cse>(enable);

	if (get_mutable() && enable)
		graph()->mark_denormalized();
}

}

static jive::node_normal_form *
get_default_normal_form(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jive::simple_normal_form(operator_class, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
	jive::node_normal_form::register_factory(typeid(jive::simple_op), get_default_normal_form);
}
