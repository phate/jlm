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

TestBinaryOperation::~TestBinaryOperation() noexcept = default;

bool
TestBinaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const TestBinaryOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

binop_reduction_path_t
TestBinaryOperation::can_reduce_operand_pair(const Output *, const Output *) const noexcept
{
  return binop_reduction_none;
}

Output *
TestBinaryOperation::reduce_operand_pair(binop_reduction_path_t, Output *, Output *) const
{
  return nullptr;
}

enum BinaryOperation::flags
TestBinaryOperation::flags() const noexcept
{
  return flags_;
}

std::string
TestBinaryOperation::debug_string() const
{
  return "TestBinaryOperation";
}

std::unique_ptr<Operation>
TestBinaryOperation::copy() const
{
  return std::make_unique<TestBinaryOperation>(*this);
}

TestOperation::~TestOperation() noexcept = default;

bool
TestOperation::operator==(const Operation & other) const noexcept
{
  const auto testOperation = dynamic_cast<const TestOperation *>(&other);
  if (!testOperation)
    return false;

  if (narguments() != testOperation->narguments() || nresults() != testOperation->nresults())
    return false;

  for (size_t n = 0; n < narguments(); n++)
  {
    if (argument(n) != testOperation->argument(n))
      return false;
  }

  for (size_t n = 0; n < nresults(); n++)
  {
    if (result(n) != testOperation->result(n))
      return false;
  }

  return true;
}

std::string
TestOperation::debug_string() const
{
  return "TestOperation";
}

std::unique_ptr<Operation>
TestOperation::copy() const
{
  return std::make_unique<TestOperation>(*this);
}

}
