/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-normal-form.hpp>

namespace jive {

structural_normal_form::~structural_normal_form() noexcept
{}

structural_normal_form::structural_normal_form(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
	: node_normal_form(operator_class, parent, graph)
{}

}

static jive::node_normal_form *
get_default_normal_form(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return new jive::structural_normal_form(operator_class, parent, graph);
}

static void __attribute__((constructor))
register_node_normal_form(void)
{
	jive::node_normal_form::register_factory(typeid(jive::structural_op), get_default_normal_form);
}
