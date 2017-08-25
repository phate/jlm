/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"

namespace jlm {

test_op::~test_op()
{}

bool
test_op::operator==(const operation & o) const noexcept
{
	auto other = dynamic_cast<const test_op*>(&o);
	if (!other) return false;

	if (narguments() != other->narguments() || nresults() != other->nresults())
		return false;

	for (size_t n = 0; n < narguments(); n++) {
		if (argument(n) != other->argument(n))
			return false;
	}

	for (size_t n = 0; n < nresults(); n++) {
		if (result(n) != other->result(n))
			return false;
	}

	return true;
}

size_t
test_op::narguments() const noexcept
{
	return arguments_.size();
}

const jive::port &
test_op::argument(size_t n) const noexcept
{
	JLM_DEBUG_ASSERT(n < narguments());
	return arguments_[n];
}

size_t
test_op::nresults() const noexcept
{
	return results_.size();
}

const jive::port &
test_op::result(size_t n) const noexcept
{
	JLM_DEBUG_ASSERT(n < nresults());
	return results_[n];
}

std::string
test_op::debug_string() const
{
	return "test_op";
}

std::unique_ptr<jive::operation>
test_op::copy() const
{
	return std::unique_ptr<operation>(new test_op(*this));
}

}
