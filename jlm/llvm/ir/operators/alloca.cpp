/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>

namespace jlm::llvm
{

/* alloca operator */

alloca_op::~alloca_op() noexcept
{}

bool
alloca_op::operator==(const Operation & other) const noexcept
{
  /* Avoid CNE for alloca operators */
  return this == &other;
}

std::string
alloca_op::debug_string() const
{
  return "ALLOCA[" + value_type().debug_string() + "]";
}

std::unique_ptr<rvsdg::Operation>
alloca_op::copy() const
{
  return std::make_unique<alloca_op>(*this);
}

}
