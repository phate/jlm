/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_OPERATORS_HPP
#define JLM_OPT_ALIAS_ANALYSES_OPERATORS_HPP

#include <jlm/ir/operators.hpp>

namespace jlm {
namespace aa {

/** \brief LambdaEntryMemStateOperator class
*/
class LambdaEntryMemStateOperator final : public MemStateOperator {
public:
  ~LambdaEntryMemStateOperator() override;

private:
  LambdaEntryMemStateOperator(
    size_t nresults,
    std::vector<std::string>  dbgstrs)
    : MemStateOperator(1, nresults)
    , dbgstrs_(std::move(dbgstrs))
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
    size_t nresults,
    const std::vector<std::string> & dbgstrs)
  {
    if (nresults != dbgstrs.size())
      throw error("Insufficient number of state debug strings.");

    auto region = output->region();
    LambdaEntryMemStateOperator op(nresults, dbgstrs);
    return jive::simple_node::create_normalized(region, op, {output});
  }

private:
  std::vector<std::string> dbgstrs_;
};

/** \brief LambdaExitMemStateOperator class
*/
class LambdaExitMemStateOperator final : public MemStateOperator {
public:
  ~LambdaExitMemStateOperator() override;

private:
  LambdaExitMemStateOperator(
    size_t noperands,
    std::vector<std::string>  dbgstrs)
    : MemStateOperator(noperands, 1)
    , dbgstrs_(std::move(dbgstrs))
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
    const std::vector<jive::output*> & operands,
    const std::vector<std::string> & dbgstrs)
  {
    if (operands.size() != dbgstrs.size())
      throw error("Insufficient number of state debug strings.");

    LambdaExitMemStateOperator op(operands.size(), dbgstrs);
    return jive::simple_node::create_normalized(region, op, operands)[0];
  }

private:
  std::vector<std::string> dbgstrs_;
};

/** \brief CallEntryMemStateOperator class
*/
class CallEntryMemStateOperator final : public MemStateOperator {
public:
  ~CallEntryMemStateOperator() override;

private:
  CallEntryMemStateOperator(
    size_t noperands,
    std::vector<std::string>  dbgstrs)
    : MemStateOperator(noperands, 1)
    , dbgstrs_(std::move(dbgstrs))
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
    const std::vector<jive::output*> & operands,
    const std::vector<std::string> & dbgstrs)
  {
    if (operands.size() != dbgstrs.size())
      throw error("Insufficient number of state debug strings.");

    CallEntryMemStateOperator op(operands.size(), dbgstrs);
    return jive::simple_node::create_normalized(region, op, operands)[0];
  }

private:
  std::vector<std::string> dbgstrs_;
};

/** \brief CallExitMemStateOperator class
*/
class CallExitMemStateOperator final : public MemStateOperator {
public:
  ~CallExitMemStateOperator() override;

private:
  CallExitMemStateOperator(
    size_t nresults,
    std::vector<std::string>  dbgstrs)
    : MemStateOperator(1, nresults)
    , dbgstrs_(std::move(dbgstrs))
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
    size_t nresults,
    const std::vector<std::string> & dbgstrs)
  {
    if (nresults != dbgstrs.size())
      throw error("Insufficient number of state debug strings.");

    auto region = output->region();
    CallExitMemStateOperator op(nresults, dbgstrs);
    return jive::simple_node::create_normalized(region, op, {output});
  }

private:
  std::vector<std::string> dbgstrs_;
};

}
}

#endif
