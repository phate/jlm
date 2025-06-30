/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TEST_TEST_TYPES_HPP
#define TEST_TEST_TYPES_HPP

#include <jlm/rvsdg/type.hpp>

namespace jlm::tests
{

class valuetype final : public rvsdg::ValueType
{
public:
  virtual ~valuetype();

  inline constexpr valuetype() noexcept
      : rvsdg::ValueType()
  {}

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const valuetype>
  Create();
};

class StateType final : public rvsdg::StateType
{
public:
  ~StateType() noexcept override;

  constexpr StateType() noexcept = default;

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const StateType>
  Create();
};

}

#endif
