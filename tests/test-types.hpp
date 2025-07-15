/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TEST_TEST_TYPES_HPP
#define TEST_TEST_TYPES_HPP

#include <jlm/rvsdg/type.hpp>

namespace jlm::tests
{

class ValueType final : public rvsdg::ValueType
{
public:
  ~ValueType() noexcept override;

  constexpr ValueType() noexcept = default;

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const ValueType>
  Create();
};

class StateType final : public rvsdg::StateType
{
public:
  ~StateType() noexcept override;

  constexpr StateType() noexcept = default;

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const StateType>
  Create();
};

}

#endif
