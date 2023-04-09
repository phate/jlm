/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/GetElementPtr.hpp>

namespace jlm {

/* getelementptr operator */

GetElementPtrOperation::~GetElementPtrOperation()
{}

bool
GetElementPtrOperation::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::GetElementPtrOperation*>(&other);
	if (!op || op->narguments() != narguments())
		return false;

	for (size_t n = 0; n < narguments(); n++) {
		if (op->argument(n) != argument(n))
			return false;
	}

	return op->result(0) == result(0);
}

std::string
GetElementPtrOperation::debug_string() const
{
	return "GETELEMENTPTR";
}

std::unique_ptr<jive::operation>
GetElementPtrOperation::copy() const
{
	return std::unique_ptr<jive::operation>(new GetElementPtrOperation(*this));
}

}
