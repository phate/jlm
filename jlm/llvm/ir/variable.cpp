/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/variable.hpp>

namespace jlm::llvm
{

Variable::~Variable() noexcept = default;

std::string
Variable::debug_string() const
{
  return name();
}

/* top level variable */

gblvariable::~gblvariable()
{}

}
