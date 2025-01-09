/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_FUNCTION_TYPE_HPP
#define JLM_RVSDG_FUNCTION_TYPE_HPP

#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>

#include <memory>
#include <vector>

namespace jlm::rvsdg
{

/**
 * \brief Function type class
 *
 * Represents the type of a callable function.
 */
class FunctionType final : public ValueType
{
public:
  ~FunctionType() noexcept override;

  FunctionType(
      std::vector<std::shared_ptr<const jlm::rvsdg::Type>> argumentTypes,
      std::vector<std::shared_ptr<const jlm::rvsdg::Type>> resultTypes);

  [[nodiscard]] const std::vector<std::shared_ptr<const jlm::rvsdg::Type>> &
  Arguments() const noexcept
  {
    return ArgumentTypes_;
  }

  [[nodiscard]] const std::vector<std::shared_ptr<const jlm::rvsdg::Type>> &
  Results() const noexcept
  {
    return ResultTypes_;
  }

  [[nodiscard]] size_t
  NumResults() const noexcept
  {
    return ResultTypes_.size();
  }

  [[nodiscard]] size_t
  NumArguments() const noexcept
  {
    return ArgumentTypes_.size();
  }

  [[nodiscard]] const jlm::rvsdg::Type &
  ResultType(size_t index) const noexcept
  {
    JLM_ASSERT(index < ResultTypes_.size());
    return *ResultTypes_[index];
  }

  [[nodiscard]] const jlm::rvsdg::Type &
  ArgumentType(size_t index) const noexcept
  {
    JLM_ASSERT(index < ArgumentTypes_.size());
    return *ArgumentTypes_[index];
  }

  std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const FunctionType>
  Create(
      std::vector<std::shared_ptr<const jlm::rvsdg::Type>> argumentTypes,
      std::vector<std::shared_ptr<const jlm::rvsdg::Type>> resultTypes);

private:
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> ArgumentTypes_;
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> ResultTypes_;
};

}

#endif
