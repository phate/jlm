/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/alloca.hpp>

namespace jlm {

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

}
