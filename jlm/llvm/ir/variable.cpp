/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/variable.hpp>

namespace jlm {

/* variable */

variable::~variable() noexcept
{}

std::string
variable::debug_string() const
{
	return name();
}

/* top level variable */

gblvariable::~gblvariable()
{}

}
