/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/util/common.hpp>

namespace jlm::util
{

error::~error()
{}

TypeError::~TypeError() noexcept = default;

}
