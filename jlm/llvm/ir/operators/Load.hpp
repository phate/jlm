/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_LOAD_HPP
#define JLM_LLVM_IR_OPERATORS_LOAD_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>

#include <optional>

namespace jlm::llvm
{

/**
 * Abstract base class for load operations.
 *
 * @see LoadVolatileOperation
 * @see LoadNonVolatileOperation
 */
class LoadOperation : public rvsdg::SimpleOperation
{
public:
  class MemoryStateInputIterator final : public rvsdg::Input::iterator<rvsdg::SimpleInput>
  {
  public:
    virtual ~MemoryStateInputIterator() = default;

    constexpr explicit MemoryStateInputIterator(rvsdg::SimpleInput * input)
        : rvsdg::Input::iterator<rvsdg::SimpleInput>(input)
    {}

    [[nodiscard]] rvsdg::SimpleInput *
    next() const override
    {
      const auto index = value()->index();
      const auto node = value()->node();

      return index + 1 < node->ninputs() ? node->input(index + 1) : nullptr;
    }
  };

  class MemoryStateOutputIterator final : public rvsdg::Output::iterator<rvsdg::SimpleOutput>
  {
  public:
    virtual ~MemoryStateOutputIterator() = default;

    constexpr explicit MemoryStateOutputIterator(rvsdg::SimpleOutput * output)
        : rvsdg::Output::iterator<rvsdg::SimpleOutput>(output)
    {}

    [[nodiscard]] rvsdg::SimpleOutput *
    next() const override
    {
      const auto index = value()->index();
      const auto node = value()->node();

      return index + 1 < node->noutputs() ? node->output(index + 1) : nullptr;
    }
  };

  using MemoryStateInputRange = util::IteratorRange<MemoryStateInputIterator>;
  using MemoryStateOutputRange = util::IteratorRange<MemoryStateOutputIterator>;

protected:
  LoadOperation(
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes,
      const size_t numMemoryStates,
      const size_t alignment)
      : SimpleOperation(operandTypes, resultTypes),
        NumMemoryStates_(numMemoryStates),
        Alignment_(alignment)
  {
    JLM_ASSERT(!operandTypes.empty() && !resultTypes.empty());

    auto & addressType = *operandTypes[0];
    JLM_ASSERT(is<PointerType>(addressType));

    auto & loadedType = *resultTypes[0];
    JLM_ASSERT(is<rvsdg::ValueType>(loadedType));

    JLM_ASSERT(operandTypes.size() == resultTypes.size());
    for (size_t n = 1; n < operandTypes.size(); n++)
    {
      auto & operandType = *operandTypes[n];
      auto & resultType = *resultTypes[n];
      JLM_ASSERT(operandType == resultType);
      JLM_ASSERT(is<rvsdg::StateType>(operandType));
    }
  }

public:
  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return Alignment_;
  }

  [[nodiscard]] std::shared_ptr<const rvsdg::ValueType>
  GetLoadedType() const noexcept
  {
    auto type = std::dynamic_pointer_cast<const rvsdg::ValueType>(result(0));
    JLM_ASSERT(type);
    return type;
  }

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept
  {
    return NumMemoryStates_;
  }

  [[nodiscard]] static rvsdg::Input &
  AddressInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<LoadOperation>(&node));
    const auto input = node.input(0);
    JLM_ASSERT(is<PointerType>(input->Type()));
    return *input;
  }

  [[nodiscard]] static rvsdg::Output &
  LoadedValueOutput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<LoadOperation>(&node));
    const auto output = node.output(0);
    JLM_ASSERT(is<rvsdg::ValueType>(output->Type()));
    return *output;
  }

  [[nodiscard]] static MemoryStateOutputRange
  MemoryStateOutputs(const rvsdg::SimpleNode & node) noexcept
  {
    const auto loadOperation = util::AssertedCast<const LoadOperation>(&node.GetOperation());
    if (loadOperation->NumMemoryStates_ == 0)
    {
      return { MemoryStateOutputIterator(nullptr), MemoryStateOutputIterator(nullptr) };
    }

    const auto firstMemoryStateOutput =
        node.output(loadOperation->nresults() - loadOperation->NumMemoryStates_);
    JLM_ASSERT(is<MemoryStateType>(firstMemoryStateOutput->Type()));
    return { MemoryStateOutputIterator(firstMemoryStateOutput),
             MemoryStateOutputIterator(nullptr) };
  }

  /**
   * Maps a memory state output of a load operation to its corresponding memory state input.
   */
  [[nodiscard]] static rvsdg::Input &
  MapMemoryStateOutputToInput(const rvsdg::Output & output)
  {
    JLM_ASSERT(is<MemoryStateType>(output.Type()));
    auto [loadNode, loadOperation] = rvsdg::TryGetSimpleNodeAndOp<LoadOperation>(output);
    JLM_ASSERT(loadOperation);
    JLM_ASSERT(loadNode->ninputs() == loadNode->noutputs());
    const auto input = loadNode->input(output.index());
    JLM_ASSERT(is<MemoryStateType>(input->Type()));
    return *input;
  }

private:
  size_t NumMemoryStates_;
  size_t Alignment_;
};

/**
 * Represents a volatile LLVM load instruction.
 *
 * In contrast to LLVM, a volatile load requires in an RVSDG setting an I/O state as it incorporates
 * externally visible side-effects. This I/O state allows the volatile load operation to be
 * sequentialized with respect to other volatile memory accesses and I/O operations. This additional
 * I/O state is the main reason why volatile loads are modeled as its own operation and volatile is
 * not just a flag at the normal \ref LoadNonVolatileOperation.
 *
 * @see StoreVolatileOperation
 */
class LoadVolatileOperation final : public LoadOperation
{
public:
  ~LoadVolatileOperation() noexcept override;

  LoadVolatileOperation(
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
            CreateOperandTypes(numMemoryStates),
            CreateResultTypes(std::move(loadedType), numMemoryStates),
            numMemoryStates,
            alignment)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] static rvsdg::Input &
  IOStateInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<LoadVolatileOperation>(&node));
    const auto input = node.input(1);
    JLM_ASSERT(is<IOStateType>(input->Type()));
    return *input;
  }

  [[nodiscard]] static rvsdg::Output &
  IOStateOutput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<LoadVolatileOperation>(&node));
    const auto output = node.output(1);
    JLM_ASSERT(is<IOStateType>(output->Type()));
    return *output;
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  Create(
      const Variable * address,
      const Variable * iOState,
      const Variable * memoryState,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    LoadVolatileOperation operation(std::move(loadedType), 1, alignment);
    return ThreeAddressCode::create(operation, { address, iOState, memoryState });
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadVolatileOperation> loadOperation,
      const std::vector<rvsdg::Output *> & operands);

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Output & address,
      rvsdg::Output & iOState,
      const std::vector<rvsdg::Output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    std::vector operands({ &address, &iOState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<LoadVolatileOperation>(
        std::move(loadedType),
        memoryStates.size(),
        alignment);
    return CreateNode(*address.region(), std::move(operation), operands);
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(
        { PointerType::Create(), IOStateType::Create() });
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateResultTypes(std::shared_ptr<const rvsdg::ValueType> loadedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(
        { std::move(loadedType), IOStateType::Create() });
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }
};

/**
 * Represents an LLVM load instruction.
 *
 * @see LoadVolatileOperation
 */
class LoadNonVolatileOperation final : public LoadOperation
{
public:
  ~LoadNonVolatileOperation() noexcept override;

  LoadNonVolatileOperation(
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
            CreateOperandTypes(numMemoryStates),
            CreateResultTypes(std::move(loadedType), numMemoryStates),
            numMemoryStates,
            alignment)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  /**
   * \brief Swaps a memory state merge operation and a load operation.
   *
   * sx1 = MemStateMerge si1 ... siM
   * v sl1 = load_op a sx1
   * =>
   * v sl1 ... slM = load_op a si1 ... siM
   * sx1 = MemStateMerge sl1 ... slM
   *
   * FIXME: The reduction can be generalized: A load node can have multiple operands from different
   * merge nodes.
   *
   * @return If the normalization could be applied, then the results of the load operation after
   * the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeLoadMux(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * \brief If the producer of a load's address is an alloca operation, then we can remove all
   * state edges originating from other alloca operations.
   *
   * a1 s1 = AllocaOperation ...
   * a2 s2 = AllocaOperation ...
   * s3 = MuxOperation s1
   * v sl1 sl2 sl3 = LoadNonVolatileOperation a1 s1 s2 s3
   * =>
   * ...
   * v sl1 sl3 = LoadNonVolatileOperation a1 s1 s3
   *
   * @param operation The load operation on which the transformation is performed.
   * @param operands The operands of the load node.
   *
   * @return If the normalization could be applied, then the results of the load operation after
   * the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeLoadAlloca(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * \brief Forwards the value from a store operation.
   *
   * s2 = StoreNonVolatileOperation a v1 s1
   * v2 s3 = LoadNonVolatileOperation a s2
   * ... = AnyOperation v2
   * =>
   * s2 = StoreNonVolatileOperation a v1 s1
   * ... = AnyOperation v1
   *
   * @param operation The load operation on which the transformation is performed.
   * @param operands The operands of the load node.
   *
   * @return If the normalization could be applied, then the results of the load operation after
   * the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeLoadStore(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * \brief If the producer of a load's address is an alloca operation, then we can remove all
   * state edges originating from other alloca operations coming through store operations.
   *
   * a1 sa1 = AllocaOperation ...
   * a2 sa2 = AllocaOperation ...
   * ss1 = StoreNonVolatileOperation a1 ... sa1
   * ss2 = StoreNonVolatileOperation a2 ... sa2
   * ... = LoadNonVolatileOperation a1 ss1 ss2
   * =>
   * ...
   * ... = LoadNonVolatileOperation a1 ss1
   *
   * @param operation The load operation on which the transformation is performed.
   * @param operands The operands of the load node.
   *
   * @return If the normalization could be applied, then the results of the load operation after
   * the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeLoadStoreState(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * \brief Remove duplicated state operands
   *
   * v so1 so2 so3 = LoadNonVolatileOperation a si1 si1 si1
   * =>
   * v so1 = LoadNonVolatileOperation a si1
   *
   * @param operation The load operation on which the transformation is performed.
   * @param operands The operands of the load node.
   *
   * @return If the normalization could be applied, then the results of the load operation after
   * the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeDuplicateStates(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * \brief Avoid sequentialization of load operations.
   *
   * _ so1 = LoadNonVolatileOperation _ si1
   * _ so2 = LoadNonVolatileOperation _ so1
   * _ so3 = LoadNonVolatileOperation _ so2
   * =>
   * _ so1 = LoadNonVolatileOperation _ si1
   * _ so2 = LoadNonVolatileOperation _ si1
   * _ so3 = LoadNonVolatileOperation _ si1
   *
   * @param operation The load operation on which the transformation is performed.
   * @param operands The operands of the load node.
   *
   * @return If the normalization could be applied, then the results of the load operation after
   * the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  NormalizeLoadLoadState(
      const LoadNonVolatileOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  static std::unique_ptr<llvm::ThreeAddressCode>
  Create(
      const Variable * address,
      const Variable * state,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    LoadNonVolatileOperation operation(std::move(loadedType), 1, alignment);
    return ThreeAddressCode::create(operation, { address, state });
  }

  static std::vector<rvsdg::Output *>
  Create(
      rvsdg::Output * address,
      const std::vector<rvsdg::Output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      const size_t alignment)
  {
    return rvsdg::outputs(&CreateNode(*address, memoryStates, std::move(loadedType), alignment));
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadNonVolatileOperation> loadOperation,
      const std::vector<rvsdg::Output *> & operands)
  {
    return rvsdg::SimpleNode::Create(region, std::move(loadOperation), operands);
  }

  static std::vector<rvsdg::Output *>
  Create(
      rvsdg::Region & region,
      std::unique_ptr<LoadNonVolatileOperation> loadOperation,
      const std::vector<rvsdg::Output *> & operands)
  {
    return outputs(&CreateNode(region, std::move(loadOperation), operands));
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Output & address,
      const std::vector<rvsdg::Output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    std::vector operands({ &address });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<LoadNonVolatileOperation>(
        std::move(loadedType),
        memoryStates.size(),
        alignment);
    return CreateNode(*address.region(), std::move(operation), operands);
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(1, PointerType::Create());
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateResultTypes(std::shared_ptr<const rvsdg::ValueType> loadedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(1, std::move(loadedType));
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }
};

}

#endif
