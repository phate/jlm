/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>
#include <jlm/rvsdg/structural-normal-form.hpp>

namespace jive {

/* port */

port::~port()
{}

port::port(const jive::type & type)
: port(type.copy())
{}

port::port(std::unique_ptr<jive::type> type)
: type_(std::move(type))
{}

bool
port::operator==(const port & other) const noexcept
{
	return *type_ == *other.type_;
}

std::unique_ptr<port>
port::copy() const
{
	return std::make_unique<port>(*this);
}

/* operation */

operation::~operation() noexcept
{}

jive::node_normal_form *
operation::normal_form(jive::graph * graph) noexcept
{
	return graph->node_normal_form(typeid(operation));
}

/* simple operation */

simple_op::~simple_op()
{}

size_t
simple_op::narguments() const noexcept
{
	return operands_.size();
}

const jive::port &
simple_op::argument(size_t index) const noexcept
{
	JIVE_DEBUG_ASSERT(index < narguments());
	return operands_[index];
}

size_t
simple_op::nresults() const noexcept
{
	return results_.size();
}

const jive::port &
simple_op::result(size_t index) const noexcept
{
	JIVE_DEBUG_ASSERT(index < nresults());
	return results_[index];
}

jive::simple_normal_form *
simple_op::normal_form(jive::graph * graph) noexcept
{
	return static_cast<jive::simple_normal_form*>(graph->node_normal_form(typeid(simple_op)));
}

/* structural operation */

bool
structural_op::operator==(const operation & other) const noexcept
{
	return typeid(*this) == typeid(other);
}

jive::structural_normal_form *
structural_op::normal_form(jive::graph * graph) noexcept
{
	return static_cast<structural_normal_form*>(graph->node_normal_form(typeid(structural_op)));
}

}
