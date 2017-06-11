/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/variable.hpp>

namespace jlm {

variable::~variable() noexcept
{}

std::string
variable::debug_string() const
{
	return name();
}

const jive::base::type &
variable::type() const noexcept
{
	return *type_;
}


global_variable::~global_variable() noexcept
{}

}
