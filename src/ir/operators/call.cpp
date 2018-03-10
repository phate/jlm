/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/call.hpp>

namespace jlm {

/* call operator */

call_op::~call_op() noexcept
{}

bool
call_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const call_op*>(&other);
	return op
	    && op->arguments_ == arguments_
	    && op->results_ == results_;
}

size_t
call_op::narguments() const noexcept
{
	return arguments_.size();
}

const jive::port &
call_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	return arguments_[index];
}

size_t
call_op::nresults() const noexcept
{
	return results_.size();
}

const jive::port &
call_op::result(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < nresults());
	return results_[index];
}

std::string
call_op::debug_string() const
{
	return "CALL";
}

std::unique_ptr<jive::operation>
call_op::copy() const
{
	return std::unique_ptr<jive::operation>(new call_op(*this));
}

}
