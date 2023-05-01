/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>

namespace jlm {

/* alloca operator */

alloca_op::~alloca_op() noexcept
{}

bool
alloca_op::operator==(const operation & other) const noexcept
{
	/* Avoid CNE for alloca operators */
	return this == &other;
}

std::string
alloca_op::debug_string() const
{
	return "ALLOCA[" + value_type().debug_string() + "]";
}

std::unique_ptr<jive::operation>
alloca_op::copy() const
{
	return std::unique_ptr<jive::operation>(new alloca_op(*this));
}

}
