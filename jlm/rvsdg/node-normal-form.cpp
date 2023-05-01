/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <cxxabi.h>
#include <typeindex>
#include <unordered_map>

#include <jlm/rvsdg/graph.hpp>

namespace jive {

node_normal_form::~node_normal_form() noexcept
{
}

bool
node_normal_form::normalize_node(jive::node * node) const
{
	return true;
}

void
node_normal_form::set_mutable(bool enable)
{
	if (enable_mutable_ == enable) {
		return;
	}

	children_set<node_normal_form, &node_normal_form::set_mutable>(enable);

	enable_mutable_ = enable;
	if (enable)
		graph()->mark_denormalized();
}

namespace {

typedef jive::node_normal_form *(*create_node_normal_form_functor)(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph);

typedef std::unordered_map<std::type_index, create_node_normal_form_functor>
	node_normal_form_registry;

std::unique_ptr<node_normal_form_registry> registry;

create_node_normal_form_functor
lookup_factory_functor(const std::type_info * info)
{
	if (!registry) {
		registry.reset(new node_normal_form_registry());
	}

	node_normal_form_registry::const_iterator i;
	for(;;) {
		auto i = registry->find(std::type_index(*info));
		if (i != registry->end()) {
			return i->second;
		}
		const auto& cinfo = dynamic_cast<const abi::__si_class_type_info &>(
			*info);
		info = cinfo.__base_type;
	}
}

}

void
node_normal_form::register_factory(
	const std::type_info & operator_class,
	jive::node_normal_form *(*fn)(
		const std::type_info & operator_class,
		jive::node_normal_form * parent,
		jive::graph * graph))
{
	if (!registry) {
		registry.reset(new node_normal_form_registry());
	}

	(*registry)[std::type_index(operator_class)] = fn;
}

node_normal_form *
node_normal_form::create(
	const std::type_info & operator_class,
	jive::node_normal_form * parent,
	jive::graph * graph)
{
	return lookup_factory_functor(&operator_class)(operator_class, parent, graph);
}

}
