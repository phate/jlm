/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

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

	static const jive::port p(jive::mem::type::instance());
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
	static const jive::port p(jive::mem::type::instance());
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

store_normal_form::~store_normal_form()
{}

store_normal_form::store_normal_form(
	const std::type_info & opclass,
	jive::node_normal_form * parent,
	jive::graph * graph) noexcept
: simple_normal_form(opclass, parent, graph)
{}

bool
store_normal_form::normalize_node(jive::node * node) const
{
	return simple_normal_form::normalize_node(node);
}

std::vector<jive::output*>
store_normal_form::normalized_create(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & operands) const
{
	return simple_normal_form::normalized_create(region, op, operands);
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
