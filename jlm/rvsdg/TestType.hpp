/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TESTTYPE_HPP
#define JLM_RVSDG_TESTTYPE_HPP

#include <jlm/rvsdg/type.hpp>

namespace jlm::rvsdg
{

/**
 * A configurable type that can be used for testing.
 */
class TestType final : public Type
{
public:
  ~TestType() noexcept override;

  explicit constexpr TestType(const TypeKind kind) noexcept
      : kind_(kind)
  {}

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  TypeKind
  Kind() const noexcept override;

  static std::shared_ptr<const TestType>
  createStateType();

  static std::shared_ptr<const TestType>
  createValueType();

private:
  TypeKind kind_;
};

}

#endif
