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
protected:
  LoadOperation(
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes,
      size_t alignment)
      : SimpleOperation(operandTypes, resultTypes),
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

  [[nodiscard]] virtual size_t
  NumMemoryStates() const noexcept = 0;

private:
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
            alignment)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept override;

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * address,
      const variable * iOState,
      const variable * memoryState,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    LoadVolatileOperation operation(std::move(loadedType), 1, alignment);
    return tac::create(operation, { address, iOState, memoryState });
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
 * Abstract base class for load nodes.
 *
 * @see LoadVolatileNode
 * @see LoadNonVolatileNode
 */
class LoadNode : public rvsdg::SimpleNode
{
protected:
  LoadNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadOperation> operation,
      const std::vector<rvsdg::output *> & operands)
      : SimpleNode(region, std::move(operation), operands)
  {}

public:
  class MemoryStateInputIterator final : public rvsdg::input::iterator<rvsdg::simple_input>
  {
  public:
    constexpr explicit MemoryStateInputIterator(rvsdg::simple_input * input)
        : rvsdg::input::iterator<rvsdg::simple_input>(input)
    {}

    [[nodiscard]] rvsdg::simple_input *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->ninputs() > index + 1 ? node->input(index + 1) : nullptr;
    }
  };

  class MemoryStateOutputIterator final : public rvsdg::output::iterator<rvsdg::simple_output>
  {
  public:
    constexpr explicit MemoryStateOutputIterator(rvsdg::simple_output * output)
        : rvsdg::output::iterator<rvsdg::simple_output>(output)
    {}

    [[nodiscard]] rvsdg::simple_output *
    next() const override
    {
      auto index = value()->index();
      auto node = value()->node();

      return node->noutputs() > index + 1 ? node->output(index + 1) : nullptr;
    }
  };

  using MemoryStateInputRange = util::IteratorRange<MemoryStateInputIterator>;
  using MemoryStateOutputRange = util::IteratorRange<MemoryStateOutputIterator>;

  [[nodiscard]] const LoadOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept
  {
    return GetOperation().NumMemoryStates();
  }

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return GetOperation().GetAlignment();
  }

  [[nodiscard]] rvsdg::input &
  GetAddressInput() const noexcept
  {
    auto addressInput = input(0);
    JLM_ASSERT(is<PointerType>(addressInput->type()));
    return *addressInput;
  }

  [[nodiscard]] rvsdg::output &
  GetLoadedValueOutput() const noexcept
  {
    auto valueOutput = output(0);
    JLM_ASSERT(is<rvsdg::ValueType>(valueOutput->type()));
    return *valueOutput;
  }

  [[nodiscard]] virtual MemoryStateInputRange
  MemoryStateInputs() const noexcept = 0;

  [[nodiscard]] virtual MemoryStateOutputRange
  MemoryStateOutputs() const noexcept = 0;

  /**
   * Create a new copy of this LoadNode that consumes the provided \p memoryStates.
   *
   * @param memoryStates The memory states the newly created copy should consume.
   * @return A newly created LoadNode.
   */
  [[nodiscard]] virtual LoadNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const = 0;
};

/**
 * Represents a LoadVolatileOperation in an RVSDG.
 */
class LoadVolatileNode final : public LoadNode
{
private:
  LoadVolatileNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadVolatileOperation> operation,
      const std::vector<rvsdg::output *> & operands)
      : LoadNode(region, std::move(operation), operands)
  {}

public:
  Node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  [[nodiscard]] const LoadVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] rvsdg::input &
  GetIoStateInput() const noexcept
  {
    auto ioInput = input(1);
    JLM_ASSERT(is<IOStateType>(ioInput->type()));
    return *ioInput;
  }

  [[nodiscard]] rvsdg::output &
  GetIoStateOutput() const noexcept
  {
    auto ioOutput = output(1);
    JLM_ASSERT(is<IOStateType>(ioOutput->type()));
    return *ioOutput;
  }

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  [[nodiscard]] LoadVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

  static LoadVolatileNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadVolatileOperation> loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new LoadVolatileNode(region, std::move(loadOperation), operands));
  }

  static LoadVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & iOState,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    std::vector<rvsdg::output *> operands({ &address, &iOState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<LoadVolatileOperation>(
        std::move(loadedType),
        memoryStates.size(),
        alignment);
    return CreateNode(*address.region(), std::move(operation), operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::Region & region,
      std::unique_ptr<LoadVolatileOperation> loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, std::move(loadOperation), operands));
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
            alignment)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept override;

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * address,
      const variable * state,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    LoadNonVolatileOperation operation(std::move(loadedType), 1, alignment);
    return tac::create(operation, { address, state });
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

/**
 * Represents a LoadNonVolatileOperation in an RVSDG.
 */
class LoadNonVolatileNode final : public LoadNode
{
private:
  LoadNonVolatileNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadNonVolatileOperation> operation,
      const std::vector<rvsdg::output *> & operands)
      : LoadNode(region, std::move(operation), operands)
  {}

public:
  [[nodiscard]] const LoadNonVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  [[nodiscard]] LoadNonVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

  Node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::output * address,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    return rvsdg::outputs(&CreateNode(*address, memoryStates, std::move(loadedType), alignment));
  }

  static LoadNonVolatileNode &
  CreateNode(
      rvsdg::output & address,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    std::vector<rvsdg::output *> operands({ &address });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<LoadNonVolatileOperation>(
        std::move(loadedType),
        memoryStates.size(),
        alignment);
    return CreateNode(*address.region(), std::move(operation), operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::Region & region,
      std::unique_ptr<LoadNonVolatileOperation> loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, std::move(loadOperation), operands));
  }

  static LoadNonVolatileNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadNonVolatileOperation> loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new LoadNonVolatileNode(region, std::move(loadOperation), operands));
  }
};

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
std::optional<std::vector<rvsdg::output *>>
NormalizeLoadMux(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief If the producer of a load's address is an alloca operation, then we can remove all
 * state edges originating from other alloca operations.
 *
 * a1 s1 = alloca_op ...
 * a2 s2 = alloca_op ...
 * s3 = mux_op s1
 * v sl1 sl2 sl3 = load_op a1 s1 s2 s3
 * =>
 * ...
 * v sl1 sl3 = load_op a1 s1 s3
 *
 * @param operation The load operation on which the transformation is performed.
 * @param operands The operands of the load node.
 *
 * @return If the normalization could be applied, then the results of the load operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeLoadAlloca(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief Forwards the value from a store operation.
 *
 * s2 = store_op a v1 s1
 * v2 s3 = load_op a s2
 * ... = any_op v2
 * =>
 * s2 = store_op a v1 s1
 * ... = any_op v1
 *
 * @param operation The load operation on which the transformation is performed.
 * @param operands The operands of the load node.
 *
 * @return If the normalization could be applied, then the results of the load operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeLoadStore(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief If the producer of a load's address is an alloca operation, then we can remove all
 * state edges originating from other alloca operations coming through store operations.
 *
 * a1 sa1 = alloca_op ...
 * a2 sa2 = alloca_op ...
 * ss1 = store_op a1 ... sa1
 * ss2 = store_op a2 ... sa2
 * ... = load_op a1 ss1 ss2
 * =>
 * ...
 * ... = load_op a1 ss1
 *
 * @param operation The load operation on which the transformation is performed.
 * @param operands The operands of the load node.
 *
 * @return If the normalization could be applied, then the results of the load operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeLoadStoreState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief Remove duplicated state operands
 *
 * v so1 so2 so3 = load_op a si1 si1 si1
 * =>
 * v so1 = load_op a si1
 *
 * @param operation The load operation on which the transformation is performed.
 * @param operands The operands of the load node.
 *
 * @return If the normalization could be applied, then the results of the load operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeLoadDuplicateState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief Avoid sequentialization of load operations.
 *
 * _ so1 = load_op _ si1
 * _ so2 = load_op _ so1
 * _ so3 = load_op _ so2
 * =>
 * _ so1 = load_op _ si1
 * _ so2 = load_op _ si1
 * _ so3 = load_op _ si1
 *
 * @param operation The load operation on which the transformation is performed.
 * @param operands The operands of the load node.
 *
 * @return If the normalization could be applied, then the results of the load operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeLoadLoadState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

}

#endif
