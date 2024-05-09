/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_OPERATORS_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_OPERATORS_HPP

#include <jlm/llvm/ir/operators.hpp>

namespace jlm::llvm
{

namespace aa
{

/** \brief LambdaEntryMemStateOperator class
 */
class LambdaEntryMemStateOperator final : public MemStateOperator
{
public:
  ~LambdaEntryMemStateOperator() override;

public:
  explicit LambdaEntryMemStateOperator(size_t nresults)
      : MemStateOperator(1, nresults)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::vector<jlm::rvsdg::output *>
  Create(jlm::rvsdg::output * output, size_t nresults)
  {
    auto region = output->region();
    LambdaEntryMemStateOperator op(nresults);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { output });
  }
};

/** \brief LambdaExitMemStateOperator class
 */
class LambdaExitMemStateOperator final : public MemStateOperator
{
public:
  ~LambdaExitMemStateOperator() override;

public:
  explicit LambdaExitMemStateOperator(size_t noperands)
      : MemStateOperator(noperands, 1)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region * region, const std::vector<jlm::rvsdg::output *> & operands)
  {
    LambdaExitMemStateOperator op(operands.size());
    return jlm::rvsdg::simple_node::create_normalized(region, op, operands)[0];
  }
};

/** \brief CallEntryMemStateOperator class
 */
class CallEntryMemStateOperator final : public MemStateOperator
{
public:
  ~CallEntryMemStateOperator() override;

public:
  explicit CallEntryMemStateOperator(size_t noperands)
      : MemStateOperator(noperands, 1)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static jlm::rvsdg::output *
  Create(jlm::rvsdg::region * region, const std::vector<jlm::rvsdg::output *> & operands)
  {
    CallEntryMemStateOperator op(operands.size());
    return jlm::rvsdg::simple_node::create_normalized(region, op, operands)[0];
  }
};

/** \brief CallExitMemStateOperator class
 */
class CallExitMemStateOperator final : public MemStateOperator
{
public:
  ~CallExitMemStateOperator() override;

public:
  explicit CallExitMemStateOperator(size_t nresults)
      : MemStateOperator(1, nresults)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::vector<jlm::rvsdg::output *>
  Create(jlm::rvsdg::output * output, size_t nresults)
  {
    auto region = output->region();
    CallExitMemStateOperator op(nresults);
    return jlm::rvsdg::simple_node::create_normalized(region, op, { output });
  }
};

}
}

#endif
