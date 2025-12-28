/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TESTOPERATIONS_HPP
#define JLM_RVSDG_TESTOPERATIONS_HPP

#include <jlm/rvsdg/nullary.hpp>

namespace jlm::rvsdg
{

class TestNullaryOperation final : public NullaryOperation
{
public:
  explicit TestNullaryOperation(const std::shared_ptr<const Type> & resultType)
      : NullaryOperation(resultType)
  {}

  bool
  operator==(const Operation & other) const noexcept override
  {
    const auto nullaryOperation = dynamic_cast<const TestNullaryOperation *>(&other);
    return nullaryOperation && *result(0) == *nullaryOperation->result(0);
  }

  [[nodiscard]] std::string
  debug_string() const override
  {
    return "NullaryOperation";
  }

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override
  {
    return std::make_unique<TestNullaryOperation>(this->result(0));
  }
};

}

#endif
