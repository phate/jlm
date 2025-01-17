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
class MemoryStateOperation : public rvsdg::SimpleOperation
{
protected:
  MemoryStateOperation(size_t numOperands, size_t numResults)
      : SimpleOperation(
            { numOperands, MemoryStateType::Create() },
            { numResults, MemoryStateType::Create() })
  {}
};

/**
 * A memory state merge operation takes multiple states as input and merges them together to a
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
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static rvsdg::output *
  Create(const std::vector<rvsdg::output *> & operands)
  {
    if (operands.empty())
      throw util::error("Insufficient number of operands.");

    return rvsdg::CreateOpNode<MemoryStateMergeOperation>(operands, operands.size()).output(0);
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

/**
 * A memory state split operation takes a single input state and splits it into multiple output
 * states.
 *
 * The operation has no equivalent LLVM instruction.
 */
class MemoryStateSplitOperation final : public MemoryStateOperation
{
public:
  ~MemoryStateSplitOperation() noexcept override;

  explicit MemoryStateSplitOperation(size_t numResults)
      : MemoryStateOperation(1, numResults)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::vector<rvsdg::output *>
  Create(rvsdg::output & operand, size_t numResults)
  {
    if (numResults == 0)
      throw util::error("Insufficient number of results.");

    return outputs(&rvsdg::CreateOpNode<MemoryStateSplitOperation>({ &operand }, numResults));
  }
};

/**
 * A lambda entry memory state split operation takes a single input state and splits it into
 * multiple output states. In contrast to the MemoryStateSplitOperation, this operation is allowed
 * to have zero output states. The operation's input is required to be connected to the memory state
 * argument of a lambda.
 *
 * The operation has no equivalent LLVM instruction.
 *
 * @see LambdaExitMemoryStateMergeOperation
 */
class LambdaEntryMemoryStateSplitOperation final : public MemoryStateOperation
{
public:
  ~LambdaEntryMemoryStateSplitOperation() noexcept override;

  explicit LambdaEntryMemoryStateSplitOperation(size_t numResults)
      : MemoryStateOperation(1, numResults)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::vector<jlm::rvsdg::output *>
  Create(rvsdg::output & output, size_t numResults)
  {
    return outputs(
        &rvsdg::CreateOpNode<LambdaEntryMemoryStateSplitOperation>({ &output }, numResults));
  }
};

/**
 * A lambda exit memory state merge operation takes multiple states as input and merges them
 * together to a single output state. In contrast to the MemoryStateMergeOperation, this operation
 * is allowed to have zero input states. The operation's output is required to be connected to the
 * memory state result of a lambda.
 *
 * The operation has no equivalent LLVM instruction.
 *
 * @see LambdaEntryMemoryStateMergeOperation
 */
class LambdaExitMemoryStateMergeOperation final : public MemoryStateOperation
{
public:
  ~LambdaExitMemoryStateMergeOperation() noexcept override;

  explicit LambdaExitMemoryStateMergeOperation(size_t numOperands)
      : MemoryStateOperation(numOperands, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static rvsdg::output &
  Create(rvsdg::Region & region, const std::vector<jlm::rvsdg::output *> & operands)
  {
    return operands.empty()
             ? *rvsdg::CreateOpNode<LambdaExitMemoryStateMergeOperation>(region, operands.size())
                    .output(0)
             : *rvsdg::CreateOpNode<LambdaExitMemoryStateMergeOperation>(operands, operands.size())
                    .output(0);
  }
};

/**
 * A call entry memory state merge operation takes multiple states as input and merges them together
 * to a single output state. In contrast to the MemoryStateMergeOperation, this operation is allowed
 * to have zero input states. The operation's output is required to be connected to the memory state
 * argument of a call.
 *
 * The operation has no equivalent LLVM instruction.
 *
 * @see CallExitMemoryStateSplitOperation
 */
class CallEntryMemoryStateMergeOperation final : public MemoryStateOperation
{
public:
  ~CallEntryMemoryStateMergeOperation() noexcept override;

  explicit CallEntryMemoryStateMergeOperation(size_t numOperands)
      : MemoryStateOperation(numOperands, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static rvsdg::output &
  Create(rvsdg::Region & region, const std::vector<rvsdg::output *> & operands)
  {
    return *rvsdg::CreateOpNode<CallEntryMemoryStateMergeOperation>(operands, operands.size())
                .output(0);
  }
};

/**
 * A call exit memory state split operation takes a single input state and splits it into
 * multiple output states. In contrast to the MemoryStateSplitOperation, this operation is allowed
 * to have zero output states. The operation's input is required to be connected to the memory state
 * result of a call.
 *
 * The operation has no equivalent LLVM instruction.
 *
 * @see CallEntryMemoryStateMergeOperation
 */
class CallExitMemoryStateSplitOperation final : public MemoryStateOperation
{
public:
  ~CallExitMemoryStateSplitOperation() noexcept override;

  explicit CallExitMemoryStateSplitOperation(size_t numResults)
      : MemoryStateOperation(1, numResults)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::vector<rvsdg::output *>
  Create(rvsdg::output & output, size_t numResults)
  {
    return outputs(
        &rvsdg::CreateOpNode<CallExitMemoryStateSplitOperation>({ &output }, numResults));
  }
};

}

#endif
