/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_MEMORYSTATEOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_MEMORYSTATEOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Abstract base class for all memory state operations.
 */
class MemoryStateOperation : public rvsdg::simple_op
{
protected:
  MemoryStateOperation(size_t numOperands, size_t numResults)
      : simple_op(CreatePorts(numOperands), CreatePorts(numResults))
  {}

private:
  static std::vector<rvsdg::port>
  CreatePorts(size_t size)
  {
    return { size, rvsdg::port(MemoryStateType()) };
  }
};

/**
 * A memory state merge operation takes multiple states as input and merges them together into a
 * single output state.
 *
 * The operation has no equivalent LLVM instruction.
 */
class MemoryStateMergeOperation final : public MemoryStateOperation
{
public:
  ~MemoryStateMergeOperation() noexcept override;

  explicit MemoryStateMergeOperation(size_t numOperands)
      : MemoryStateOperation(numOperands, 1)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  static rvsdg::output *
  Create(const std::vector<rvsdg::output *> & operands)
  {
    if (operands.empty())
      throw util::error("Insufficient number of operands.");

    MemoryStateMergeOperation operation(operands.size());
    auto region = operands.front()->region();
    return rvsdg::simple_node::create_normalized(region, operation, operands)[0];
  }

  static std::unique_ptr<tac>
  Create(const std::vector<const variable *> & operands)
  {
    if (operands.empty())
      throw util::error("Insufficient number of operands.");

    MemoryStateMergeOperation operation(operands.size());
    return tac::create(operation, operands);
  }
};

/** \brief MemStateSplit operator
 */
class MemStateSplitOperator final : public MemoryStateOperation
{
public:
  ~MemStateSplitOperator() override;

  MemStateSplitOperator(size_t nresults)
      : MemoryStateOperation(1, nresults)
  {}

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<jlm::rvsdg::operation>
  copy() const override;

  static std::vector<jlm::rvsdg::output *>
  Create(jlm::rvsdg::output * operand, size_t nresults)
  {
    if (nresults == 0)
      throw jlm::util::error("Insufficient number of results.");

    MemStateSplitOperator op(nresults);
    return jlm::rvsdg::simple_node::create_normalized(operand->region(), op, { operand });
  }
};

/** \brief LambdaEntryMemStateOperator class
 */
class LambdaEntryMemStateOperator final : public MemoryStateOperation
{
public:
  ~LambdaEntryMemStateOperator() override;

public:
  explicit LambdaEntryMemStateOperator(size_t nresults)
      : MemoryStateOperation(1, nresults)
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
class LambdaExitMemStateOperator final : public MemoryStateOperation
{
public:
  ~LambdaExitMemStateOperator() override;

public:
  explicit LambdaExitMemStateOperator(size_t noperands)
      : MemoryStateOperation(noperands, 1)
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
class CallEntryMemStateOperator final : public MemoryStateOperation
{
public:
  ~CallEntryMemStateOperator() override;

public:
  explicit CallEntryMemStateOperator(size_t noperands)
      : MemoryStateOperation(noperands, 1)
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
class CallExitMemStateOperator final : public MemoryStateOperation
{
public:
  ~CallExitMemStateOperator() override;

public:
  explicit CallExitMemStateOperator(size_t nresults)
      : MemoryStateOperation(1, nresults)
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

#endif
