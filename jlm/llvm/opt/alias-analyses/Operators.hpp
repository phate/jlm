/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_OPERATORS_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_OPERATORS_HPP

#include <jlm/llvm/ir/operators.hpp>

namespace jlm {
namespace aa {

/** \brief LambdaEntryMemStateOperator class
*/
class LambdaEntryMemStateOperator final : public MemStateOperator {
public:
  ~LambdaEntryMemStateOperator() override;

private:
  explicit
  LambdaEntryMemStateOperator(
    size_t nresults)
    : MemStateOperator(1, nresults)
  {}

public:
  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jive::operation>
  copy() const override;

  static std::vector<jive::output*>
  Create(
    jive::output * output,
    size_t nresults)
  {
    auto region = output->region();
    LambdaEntryMemStateOperator op(nresults);
    return jive::simple_node::create_normalized(region, op, {output});
  }
};

/** \brief LambdaExitMemStateOperator class
*/
class LambdaExitMemStateOperator final : public MemStateOperator {
public:
  ~LambdaExitMemStateOperator() override;

private:
  explicit
  LambdaExitMemStateOperator(
    size_t noperands)
    : MemStateOperator(noperands, 1)
  {}

public:
  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jive::operation>
  copy() const override;

  static jive::output *
  Create(
    jive::region * region,
    const std::vector<jive::output*> & operands)
  {
    LambdaExitMemStateOperator op(operands.size());
    return jive::simple_node::create_normalized(region, op, operands)[0];
  }
};

/** \brief CallEntryMemStateOperator class
*/
class CallEntryMemStateOperator final : public MemStateOperator {
public:
  ~CallEntryMemStateOperator() override;

private:
  explicit
  CallEntryMemStateOperator(
    size_t noperands)
    : MemStateOperator(noperands, 1)
  {}

public:
  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jive::operation>
  copy() const override;

  static jive::output *
  Create(
    jive::region * region,
    const std::vector<jive::output*> & operands)
  {
    CallEntryMemStateOperator op(operands.size());
    return jive::simple_node::create_normalized(region, op, operands)[0];
  }
};

/** \brief CallExitMemStateOperator class
*/
class CallExitMemStateOperator final : public MemStateOperator {
public:
  ~CallExitMemStateOperator() override;

private:
  explicit
  CallExitMemStateOperator(
    size_t nresults)
    : MemStateOperator(1, nresults)
  {}

public:
  bool
  operator==(const operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<jive::operation>
  copy() const override;

  static std::vector<jive::output*>
  Create(
    jive::output * output,
    size_t nresults)
  {
    auto region = output->region();
    CallExitMemStateOperator op(nresults);
    return jive::simple_node::create_normalized(region, op, {output});
  }
};

}
}

#endif
