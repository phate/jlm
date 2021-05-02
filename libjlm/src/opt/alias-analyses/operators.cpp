/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/opt/alias-analyses/operators.hpp>

namespace jlm {
namespace aa {

static std::string
dbgstr(const std::vector<std::string> & dbgstrs)
{
	/*
		FIXME: This is a string join. It is about time that we get our own string type.
	*/
	std::string dbgstr = "";
	for (size_t n = 0; n < dbgstrs.size(); n++) {
		if (n != 0)
			dbgstr += "|";

		dbgstr += dbgstrs[n];
	}

	return dbgstr;
}

/* lambda_aamux_op class */

lambda_aamux_op::~lambda_aamux_op()
{}

bool
lambda_aamux_op::operator==(const jive::operation & other) const noexcept
{
	auto op = dynamic_cast<const lambda_aamux_op*>(&other);
	return op
	    && op->is_entry_ == is_entry_;
}

std::string
lambda_aamux_op::debug_string() const
{
	if (is_entry_)
		return strfmt("LAMBDA_ENTRY_AAMUX[", dbgstr(dbgstrs_), "]");

	return strfmt("LAMBDA_EXIT_AAMUX[", dbgstr(dbgstrs_), "]");
}

std::unique_ptr<jive::operation>
lambda_aamux_op::copy() const
{
	return std::unique_ptr<jive::operation>(new lambda_aamux_op(*this));
}

/* call_aamux_op class */

call_aamux_op::~call_aamux_op()
{}

bool
call_aamux_op::operator==(const jive::operation & other) const noexcept
{
	auto op = dynamic_cast<const call_aamux_op*>(&other);
	return op
	    && op->is_entry_ == is_entry_;
}

std::string
call_aamux_op::debug_string() const
{
	if (is_entry_)
		return strfmt("CALL_ENTRY_AAMUX[", dbgstr(dbgstrs_), "]");

	return strfmt("CALL_EXIT_AAMUX[", dbgstr(dbgstrs_), "]");
}

std::unique_ptr<jive::operation>
call_aamux_op::copy() const
{
	return std::unique_ptr<jive::operation>(new call_aamux_op(*this));
}

}
}
