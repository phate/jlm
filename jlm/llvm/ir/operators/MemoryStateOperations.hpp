/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_MEMORYSTATEOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_MEMORYSTATEOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/util/BijectiveMap.hpp>

namespace jlm::llvm
{

using MemoryNodeId = std::size_t;

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
  {
    if (numOperands == 0)
      throw util::Error("Insufficient number of operands.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /** \brief Removes the MemoryStateMergeOperation as it has only a single operand, i.e., no
   * merging is performed.
   *
   * so = MemoryStateMergeOperation si
   * ... = AnyOperation so
   * =>
   * ... = AnyOperation si
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeSingleOperand(
      const MemoryStateMergeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /** \brief Removes duplicated operands from the MemoryStateMergeOperation.
   *
   * so = MemoryStateMergeOperation si0 si0 si1 si1 si2
   * =>
   * so = MemoryStateMergeOperation si0 si1 si2
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeDuplicateOperands(
      const MemoryStateMergeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /** \brief Fuses nested merges into a single merge
   *
   * o1 = MemoryStateMergeOperation i1 i2
   * o2 = MemoryStateMergeOperation o1 i3
   * =>
   * o2 = MemoryStateMergeOperation i1 i2 i3
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeNestedMerges(
      const MemoryStateMergeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /** \brief Fuses nested splits into a single merge
   *
   * o1, o2, o3 = MemoryStateSplitOperation i1
   * o4 = MemoryStateMergeOperation i2 o1 o2 o3 i3
   * =>
   * o4 = MemoryStateMergeOperation i2 i1 i1 i1 i3
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeMergeSplit(
      const MemoryStateMergeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  static rvsdg::SimpleNode &
  CreateNode(const std::vector<rvsdg::Output *> & operands)
  {
    return rvsdg::CreateOpNode<MemoryStateMergeOperation>(operands, operands.size());
  }

  static rvsdg::Output *
  Create(const std::vector<rvsdg::Output *> & operands)
  {
    return CreateNode(operands).output(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  Create(const std::vector<const Variable *> & operands)
  {
    if (operands.empty())
      throw util::Error("Insufficient number of operands.");

    MemoryStateMergeOperation operation(operands.size());
    return ThreeAddressCode::create(operation, operands);
  }
};

/**
 * A memory state join operation takes multiple states that represent the same abstract memory
 * location as input and joins them together to a single output state.
 *
 * The operation has no equivalent LLVM instruction.
 */
class MemoryStateJoinOperation final : public MemoryStateOperation
{
public:
  ~MemoryStateJoinOperation() noexcept override;

  explicit MemoryStateJoinOperation(const size_t numOperands)
      : MemoryStateOperation(numOperands, 1)
  {
    if (numOperands == 0)
      throw std::logic_error("Insufficient number of operands.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /** \brief Removes the MemoryStateJoinOperation as it has only a single operand, i.e., no
   * joining is performed.
   *
   * so = MemoryStateJoinOperation si
   * ... = AnyOperation so
   * =>
   * ... = AnyOperation si
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeSingleOperand(
      const MemoryStateJoinOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /** \brief Removes duplicated operands from the MemoryStateJoinOperation.
   *
   * so = MemoryStateJoinOperation si0 si0 si1 si1 si2
   * =>
   * so = MemoryStateJoinOperation si0 si1 si2
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeDuplicateOperands(
      const MemoryStateJoinOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /** \brief Fuses nested \ref MemoryStateJoinOperation nodes into a single node
   *
   * o1 = MemoryStateJoinOperation i1 i2
   * o2 = MemoryStateJoinOperation o1 i3
   * =>
   * o2 = MemoryStateJoinOperation i1 i2 i3
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeNestedJoins(
      const MemoryStateJoinOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  static rvsdg::SimpleNode &
  CreateNode(const std::vector<rvsdg::Output *> & operands)
  {
    return rvsdg::CreateOpNode<MemoryStateJoinOperation>(operands, operands.size());
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

  explicit MemoryStateSplitOperation(const size_t numResults)
      : MemoryStateOperation(1, numResults)
  {
    if (numResults == 0)
      throw util::Error("Insufficient number of results.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /** \brief Removes the MemoryStateSplitOperation as it has only a single result, i.e., no
   * splitting is performed.
   *
   * so = MemoryStateSplitOperation si
   * ... = AnyOperation so
   * =>
   * ... = AnyOperation si
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeSingleResult(
      const MemoryStateSplitOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /** \brief Fuses nested splits into a single split
   *
   * o1 o2 o3 = MemoryStateSplitOperation i1
   * o4 o5 = MemoryStateSplitOperation o2
   * =>
   * o1 o4 o5 o3 = MemoryStateSplitOperation i1
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeNestedSplits(
      const MemoryStateSplitOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /** \brief Removes an idempotent split-merge pair
   *
   * o1 = MemoryStateMergeOperation i1 i2 i3
   * o2 o3 o4 = MemoryStateSplitOperation o1
   * ... = AnyOperation o2 o3 o4
   * =>
   * ... = AnyOperation i1 i2 i3
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeSplitMerge(
      const MemoryStateSplitOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  static rvsdg::SimpleNode &
  CreateNode(rvsdg::Output & operand, const size_t numResults)
  {
    return rvsdg::CreateOpNode<MemoryStateSplitOperation>({ &operand }, numResults);
  }

  static std::vector<rvsdg::Output *>
  Create(rvsdg::Output & operand, const size_t numResults)
  {
    return outputs(&CreateNode(operand, numResults));
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

  explicit LambdaEntryMemoryStateSplitOperation(const std::vector<MemoryNodeId> & memoryNodeIds);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /**
   * @return The \ref MemoryNodeId for each of the operation's results.
   */
  [[nodiscard]] std::vector<MemoryNodeId>
  getMemoryNodeIds() const noexcept
  {
    std::vector<MemoryNodeId> memoryNodeIds(nresults());
    for (auto [memoryNodeId, index] : memoryNodeIdToIndexMap_)
    {
      JLM_ASSERT(index < nresults());
      memoryNodeIds[index] = memoryNodeId;
    }

    return memoryNodeIds;
  }

  /**
   * Maps a memory node identifier to the respective output of a
   * LambdaEntryMemoryStateSplitOperation node.
   *
   * @param node A LambdaEntryMemoryStateSplitOperation node.
   * @param memoryNodeId A memory node identifier.
   * @return The respective output if the memory node identifier maps to one, otherwise nullptr.
   */
  [[nodiscard]] static rvsdg::Output *
  tryMapMemoryNodeIdToOutput(const rvsdg::SimpleNode & node, MemoryNodeId memoryNodeId);

  /**
   * Maps the output a \ref LambdaEntryMemoryStateSplitOperation node to the respective \ref
   * MemoryNodeId.
   *
   * @param output A output of \ref LambdaEntryMemoryStateSplitOperation node.
   * @return The \ref MemoryNodeId.
   *
   * \pre The output is assumed to belong to a \ref LambdaEntryMemoryStateSplitOperation node.
   */
  [[nodiscard]] static MemoryNodeId
  mapOutputToMemoryNodeId(const rvsdg::Output & output);

  /**
   * Perform the following transformation:
   *
   * oN = CallEntryMemoryStateMergeOperation o0 ... oK
   * oX ... oZ = LambdaEntryMemoryStateSplitOperation oN
   * ... = AnyOp oX ... oZ
   * =>
   * ... = AnyOp o0 ... oK
   *
   * This transformation can occur after function inlining, i.e., a \ref CallOperation has been
   * replaced with the body of its respective \ref rvsdg::LambdaNode.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeCallEntryMemoryStateMerge(
      const LambdaEntryMemoryStateSplitOperation & lambdaEntrySplitOperation,
      const std::vector<rvsdg::Output *> & operands);

  static rvsdg::SimpleNode &
  CreateNode(rvsdg::Output & operand, std::vector<MemoryNodeId> memoryNodeIds)
  {
    return rvsdg::CreateOpNode<LambdaEntryMemoryStateSplitOperation>(
        { &operand },
        std::move(memoryNodeIds));
  }

private:
  util::BijectiveMap<MemoryNodeId, size_t> memoryNodeIdToIndexMap_{};
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

  explicit LambdaExitMemoryStateMergeOperation(const std::vector<MemoryNodeId> & memoryNodeIds);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /**
   * @return The \ref MemoryNodeId for each of the operation's operands.
   */
  [[nodiscard]] std::vector<MemoryNodeId>
  getMemoryNodeIds() const noexcept
  {
    std::vector<MemoryNodeId> memoryNodeIds(narguments());
    for (auto [memoryNodeId, index] : MemoryNodeIdToIndex_)
    {
      JLM_ASSERT(index < narguments());
      memoryNodeIds[index] = memoryNodeId;
    }

    return memoryNodeIds;
  }

  /**
   * Maps a memory node identifier to the respective input of a LambdaExitMemoryStateMergeOperation
   * node.
   *
   * @param node A LambdaExitMemoryStateMergeOperation node.
   * @param memoryNodeId A memory node identifier.
   * @return The respective input if the memory node identifier maps to one, otherwise nullptr.
   */
  [[nodiscard]] static rvsdg::Input *
  tryMapMemoryNodeIdToInput(const rvsdg::SimpleNode & node, MemoryNodeId memoryNodeId);

  /**
   * Maps the input a \ref LambdaExitMemoryStateMergeOperation node to the respective \ref
   * MemoryNodeId.
   *
   * @param input A input of \ref LambdaExitMemoryStateMergeOperation node.
   * @return The \ref MemoryNodeId.
   *
   * \pre The input is assumed to belong to a \ref LambdaExitMemoryStateMergeOperation node.
   */
  [[nodiscard]] static MemoryNodeId
  mapInputToMemoryNodeId(const rvsdg::Input & input);

  /**
   * Performs the following transformation:
   *
   * a, s1 = AllocaOperation ...
   * v, s2 = LoadOperation a s1
   * ... = LambdaExitMemoryStateMergeOperation s2 ... sn
   * =>
   * a, s1 = AllocaOperation ...
   * v, s2 = LoadOperation a s1
   * ... = LambdaExitMemoryStateMergeOperation s1 ... sn
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeLoadFromAlloca(
      const LambdaExitMemoryStateMergeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following transformation:
   *
   * a, s1 = AllocaOperation ...
   * s2 = StoreOperation a v s1
   * ... = LambdaExitMemoryStateMergeOperation s2 ... sn
   * =>
   * a, s1 = AllocaOperation ...
   * s2 = StoreOperation a v s1
   * ... = LambdaExitMemoryStateMergeOperation s1 ... sn
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeStoreToAlloca(
      const LambdaExitMemoryStateMergeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following transformation:
   *
   * a, s1 = AllocaOperation ...
   * ... = LambdaExitMemoryStateMergeOperation s1 ... sn
   * =>
   * a, s1 = AllocaOperation ...
   * s2 = UndefValueOperation
   * ... = LambdaExitMemoryStateMergeOperation s2 ... sn
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeAlloca(
      const LambdaExitMemoryStateMergeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  static rvsdg::Node &
  CreateNode(
      rvsdg::Region & region,
      const std::vector<rvsdg::Output *> & operands,
      const std::vector<MemoryNodeId> & memoryNodeIds)
  {
    return operands.empty()
             ? rvsdg::CreateOpNode<LambdaExitMemoryStateMergeOperation>(region, memoryNodeIds)
             : rvsdg::CreateOpNode<LambdaExitMemoryStateMergeOperation>(operands, memoryNodeIds);
  }

private:
  util::BijectiveMap<MemoryNodeId, size_t> MemoryNodeIdToIndex_{};
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

  explicit CallEntryMemoryStateMergeOperation(const std::vector<MemoryNodeId> & memoryNodeIds);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /**
   * @return The \ref MemoryNodeId for each of the operation's results.
   */
  [[nodiscard]] std::vector<MemoryNodeId>
  getMemoryNodeIds() const noexcept
  {
    std::vector<MemoryNodeId> memoryNodeIds(narguments());
    for (auto [memoryNodeId, index] : MemoryNodeIdToIndex_)
    {
      JLM_ASSERT(index < narguments());
      memoryNodeIds[index] = memoryNodeId;
    }

    return memoryNodeIds;
  }

  /**
   * Maps a memory node identifier to the respective input of a \ref
   * CallEntryMemoryStateMergeOperation node.
   *
   * @param node A \ref CallEntryMemoryStateMergeOperation node.
   * @param memoryNodeId A memory node identifier.
   * @return The respective input if the memory node identifier maps to one, otherwise nullptr.
   */
  [[nodiscard]] static rvsdg::Input *
  tryMapMemoryNodeIdToInput(const rvsdg::SimpleNode & node, MemoryNodeId memoryNodeId);

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Region & region,
      const std::vector<rvsdg::Output *> & operands,
      std::vector<MemoryNodeId> memoryNodeIds)
  {
    return operands.empty() ? rvsdg::CreateOpNode<CallEntryMemoryStateMergeOperation>(
                                  region,
                                  std::move(memoryNodeIds))
                            : rvsdg::CreateOpNode<CallEntryMemoryStateMergeOperation>(
                                  operands,
                                  std::move(memoryNodeIds));
  }

private:
  util::BijectiveMap<MemoryNodeId, size_t> MemoryNodeIdToIndex_{};
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

  explicit CallExitMemoryStateSplitOperation(const std::vector<MemoryNodeId> & memoryNodeIds);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /**
   * @return The \ref MemoryNodeId for each of the operation's results.
   */
  [[nodiscard]] std::vector<MemoryNodeId>
  getMemoryNodeIds() const noexcept
  {
    std::vector<MemoryNodeId> memoryNodeIds(nresults());
    for (auto [memoryNodeId, index] : memoryNodeIdToIndexMap_)
    {
      JLM_ASSERT(index < nresults());
      memoryNodeIds[index] = memoryNodeId;
    }

    return memoryNodeIds;
  }

  /**
   * Maps a memory node identifier to the respective output of a \ref
   * CallExitMemoryStateSplitOperation node.
   *
   * @param node A \ref CallExitMemoryStateSplitOperation node.
   * @param memoryNodeId A memory node identifier.
   * @return The respective output if the memory node identifier maps to one, otherwise nullptr.
   */
  [[nodiscard]] static rvsdg::Output *
  tryMapMemoryNodeIdToOutput(const rvsdg::SimpleNode & node, MemoryNodeId memoryNodeId);

  /**
   * Maps the output a \ref CallExitMemoryStateSplitOperation node to the respective \ref
   * MemoryNodeId.
   *
   * @param output A output of \ref CallExitMemoryStateSplitOperation node.
   * @return The \ref MemoryNodeId.
   *
   * \pre The output is assumed to belong to a \ref CallExitMemoryStateSplitOperation node.
   */
  [[nodiscard]] static MemoryNodeId
  mapOutputToMemoryNodeId(const rvsdg::Output & output);

  /**
   * Perform the following transformation:
   *
   * oN = LambdaExitMemoryStateMergeOperation o0 ... oK
   * oX ... oZ = CallExitMemoryStateSplitOperation oN
   * ... = AnyOp oX ... oZ
   * =>
   * ... = AnyOp o0 ... oK
   *
   * This transformation can occur after function inlining, i.e., a \ref CallOperation has been
   * replaced with the body of its respective \ref rvsdg::LambdaNode.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeLambdaExitMemoryStateMerge(
      const CallExitMemoryStateSplitOperation & callExitSplitOperation,
      const std::vector<rvsdg::Output *> & operands);

  static rvsdg::SimpleNode &
  CreateNode(rvsdg::Output & operand, std::vector<MemoryNodeId> memoryNodeIds)
  {
    return rvsdg::CreateOpNode<CallExitMemoryStateSplitOperation>(
        { &operand },
        std::move(memoryNodeIds));
  }

private:
  util::BijectiveMap<MemoryNodeId, size_t> memoryNodeIdToIndexMap_{};
};

/**
 * Checks if the given node takes any inputs, or produces any outputs,
 * that are of the memory state type.
 * @param node the node in question
 * @return true if any input or output is a memory state, otherwise false.
 */
[[nodiscard]] bool
hasMemoryState(const rvsdg::Node & node);

}

#endif
