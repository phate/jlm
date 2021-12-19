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

/* LambdaEntryMemStateOperator class */

LambdaEntryMemStateOperator::~LambdaEntryMemStateOperator() = default;

bool
LambdaEntryMemStateOperator::operator==(const jive::operation & other) const noexcept
{
    return is<LambdaEntryMemStateOperator>(other);
}

std::string
LambdaEntryMemStateOperator::debug_string() const
{
    return strfmt("LambdaEntryMemState[", dbgstr(dbgstrs_), "]");
}

std::unique_ptr<jive::operation>
LambdaEntryMemStateOperator::copy() const
{
    return std::unique_ptr<jive::operation>(new LambdaEntryMemStateOperator(*this));
}

/* LambdaExitMemStateOperator class */

LambdaExitMemStateOperator::~LambdaExitMemStateOperator() = default;

bool
LambdaExitMemStateOperator::operator==(const jive::operation & other) const noexcept
{
    return is<LambdaExitMemStateOperator>(other);
}

std::string
LambdaExitMemStateOperator::debug_string() const
{
    return strfmt("LambdaExitMemState[", dbgstr(dbgstrs_), "]");
}

std::unique_ptr<jive::operation>
LambdaExitMemStateOperator::copy() const
{
    return std::unique_ptr<jive::operation>(new LambdaExitMemStateOperator(*this));
}

/* CallEntryMemStateOperator class */

CallEntryMemStateOperator::~CallEntryMemStateOperator() = default;

bool
CallEntryMemStateOperator::operator==(const jive::operation & other) const noexcept
{
  return is<CallEntryMemStateOperator>(other);
}

std::string
CallEntryMemStateOperator::debug_string() const
{
  return strfmt("CallEntryMemState[", dbgstr(dbgstrs_), "]");
}

std::unique_ptr<jive::operation>
CallEntryMemStateOperator::copy() const
{
  return std::unique_ptr<jive::operation>(new CallEntryMemStateOperator(*this));
}

/* CallExitMemStateOperator class */

CallExitMemStateOperator::~CallExitMemStateOperator() = default;

bool
CallExitMemStateOperator::operator==(const jive::operation & other) const noexcept
{
  return is<CallExitMemStateOperator>(other);
}

std::string
CallExitMemStateOperator::debug_string() const
{
  return strfmt("CallExitMemState[", dbgstr(dbgstrs_), "]");
}

std::unique_ptr<jive::operation>
CallExitMemStateOperator::copy() const
{
  return std::unique_ptr<jive::operation>(new CallExitMemStateOperator(*this));
}

}
}
