/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/variable.hpp>

namespace jlm {

variable::~variable() noexcept
{}

const jive::base::type &
variable::type() const noexcept
{
	return *type_;
}

}
