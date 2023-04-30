/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>

namespace jlm {

error::~error()
{}

}

namespace jive {

compiler_error::~compiler_error() noexcept
{
}

type_error::~type_error() noexcept
{
}

}
