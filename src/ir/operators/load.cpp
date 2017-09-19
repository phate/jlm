/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/arch/memorytype.h>

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

}
