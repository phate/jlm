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
		if (argument_type(n) != other->argument_type(n))
			return false;
	}

	for (size_t n = 0; n < nresults(); n++) {
		if (result_type(n) != other->result_type(n))
			return false;
	}

	return true;
}

size_t
test_op::narguments() const noexcept
{
	return argument_types_.size();
}

const jive::base::type &
test_op::argument_type(size_t n) const noexcept
{
	return *argument_types_[n];
}

size_t
test_op::nresults() const noexcept
{
	return result_types_.size();
}

const jive::base::type &
test_op::result_type(size_t n) const noexcept
{
	return *result_types_[n];
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
