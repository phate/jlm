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
	if (!op || op->narguments() != narguments() || op->nresults() != nresults())
		return false;

	for (size_t n = 0; n < narguments(); n++) {
		if (op->argument(n) != argument(n))
			return false;
	}

	for (size_t n = 0; n < nresults(); n++) {
		if (op->result(n) != result(n))
			return false;
	}

	return true;
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
