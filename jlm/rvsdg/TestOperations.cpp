/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/TestOperations.hpp>

namespace jlm::rvsdg
{

TestUnaryOperation::~TestUnaryOperation() noexcept = default;

bool
TestUnaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const TestUnaryOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

unop_reduction_path_t
TestUnaryOperation::can_reduce_operand(const Output *) const noexcept
{
  return unop_reduction_none;
}

Output *
TestUnaryOperation::reduce_operand(unop_reduction_path_t, Output *) const
{
  return nullptr;
}

std::string
TestUnaryOperation::debug_string() const
{
  return "TestUnaryOperation";
}

std::unique_ptr<Operation>
TestUnaryOperation::copy() const
{
  return std::make_unique<TestUnaryOperation>(*this);
}

}
