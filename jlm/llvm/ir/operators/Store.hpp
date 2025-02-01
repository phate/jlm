/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_STORE_HPP
#define JLM_LLVM_IR_OPERATORS_STORE_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>

#include <optional>

namespace jlm::llvm
{

/**
 * Abstract base class for store operations.
 *
 * @see StoreVolatileOperation
 * @see StoreNonVolatileOperation
 */
class StoreOperation : public rvsdg::SimpleOperation
{
protected:
  StoreOperation(
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes,
      size_t alignment)
      : SimpleOperation(operandTypes, resultTypes),
        Alignment_(alignment)
  {
    JLM_ASSERT(operandTypes.size() >= 2);

    auto & addressType = *operandTypes[0];
    JLM_ASSERT(is<PointerType>(addressType));

    auto & storedType = *operandTypes[1];
    JLM_ASSERT(is<rvsdg::ValueType>(storedType));

    JLM_ASSERT(operandTypes.size() == resultTypes.size() + 2);
    for (size_t n = 0; n < resultTypes.size(); n++)
    {
      auto & operandType = *operandTypes[n + 2];
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

  [[nodiscard]] const rvsdg::ValueType &
  GetStoredType() const noexcept
  {
    return *util::AssertedCast<const rvsdg::ValueType>(argument(1).get());
  }

  [[nodiscard]] virtual size_t
  NumMemoryStates() const noexcept = 0;

private:
  size_t Alignment_;
};

/**
 * Represents an LLVM store instruction.
 *
 * @see StoreVolatileOperation
 */
class StoreNonVolatileOperation final : public StoreOperation
{
public:
  ~StoreNonVolatileOperation() noexcept override;

  StoreNonVolatileOperation(
      std::shared_ptr<const rvsdg::ValueType> storedType,
      size_t numMemoryStates,
      size_t alignment)
      : StoreOperation(
            CreateOperandTypes(std::move(storedType), numMemoryStates),
            { numMemoryStates, MemoryStateType::Create() },
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
  Create(const variable * address, const variable * value, const variable * state, size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value->Type());

    StoreNonVolatileOperation op(storedType, 1, alignment);
    return tac::create(op, { address, value, state });
  }

private:
  static const std::shared_ptr<const jlm::rvsdg::ValueType>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::ValueType>(type))
    {
      return storedType;
    }

    throw util::error("Expected value type");
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(std::shared_ptr<const rvsdg::ValueType> storedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(
        { PointerType::Create(), std::move(storedType) });
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }
};

/**
 * Abstract base class for store nodes
 *
 * @see StoreVolatileNode
 * @see StoreNonVolatileNode
 */
class StoreNode : public rvsdg::SimpleNode
{
protected:
  StoreNode(
      rvsdg::Region & region,
      std::unique_ptr<StoreOperation> operation,
      const std::vector<rvsdg::output *> & operands)
      : SimpleNode(region, std::move(operation), operands)
  {}

public:
  class MemoryStateInputIterator final : public rvsdg::input::iterator<rvsdg::simple_input>
  {
  public:
    constexpr explicit MemoryStateInputIterator(rvsdg::simple_input * input)
        : rvsdg::input::iterator<jlm::rvsdg::simple_input>(input)
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

  [[nodiscard]] const StoreOperation &
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

  [[nodiscard]] rvsdg::input &
  GetStoredValueInput() const noexcept
  {
    auto valueInput = input(1);
    JLM_ASSERT(is<rvsdg::ValueType>(valueInput->type()));
    return *valueInput;
  }

  [[nodiscard]] virtual MemoryStateInputRange
  MemoryStateInputs() const noexcept = 0;

  [[nodiscard]] virtual MemoryStateOutputRange
  MemoryStateOutputs() const noexcept = 0;

  /**
   * Create a new copy of this StoreNode that consumes the provided \p memoryStates.
   *
   * @param memoryStates The memory states the newly created copy should consume.
   * @return A newly created StoreNode.
   */
  [[nodiscard]] virtual StoreNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const = 0;
};

/**
 * Represents a StoreNonVolatileOperation in an RVSDG.
 */
class StoreNonVolatileNode final : public StoreNode
{
private:
  StoreNonVolatileNode(
      rvsdg::Region & region,
      std::unique_ptr<StoreNonVolatileOperation> operation,
      const std::vector<jlm::rvsdg::output *> & operands)
      : StoreNode(region, std::move(operation), operands)
  {}

public:
  [[nodiscard]] const StoreNonVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  [[nodiscard]] StoreNonVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

  Node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::output * address,
      rvsdg::output * value,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    return rvsdg::outputs(&CreateNode(*address, *value, memoryStates, alignment));
  }

  static StoreNonVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & value,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value.Type());

    std::vector<rvsdg::output *> operands({ &address, &value });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<StoreNonVolatileOperation>(
        std::move(storedType),
        memoryStates.size(),
        alignment);
    return CreateNode(*address.region(), std::move(operation), operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::Region & region,
      std::unique_ptr<StoreNonVolatileOperation> storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, std::move(storeOperation), operands));
  }

  static StoreNonVolatileNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<StoreNonVolatileOperation> storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new StoreNonVolatileNode(region, std::move(storeOperation), operands));
  }

private:
  static std::shared_ptr<const rvsdg::ValueType>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::ValueType>(type))
    {
      return storedType;
    }

    throw util::error("Expected value type.");
  }
};

/**
 * Represents a volatile LLVM store instruction.
 *
 * In contrast to LLVM, a volatile store requires in an RVSDG setting an I/O state as it
 * incorporates externally visible side-effects. This I/O state allows the volatile store operation
 * to be sequentialized with respect to other volatile memory accesses and I/O operations. This
 * additional I/O state is the main reason why volatile stores are modeled as its own operation and
 * volatile is not just a flag at the normal \ref StoreNonVolatileOperation.
 *
 * @see StoreNonVolatileOperation
 */
class StoreVolatileOperation final : public StoreOperation
{
public:
  ~StoreVolatileOperation() noexcept override;

  StoreVolatileOperation(
      std::shared_ptr<const rvsdg::ValueType> storedType,
      size_t numMemoryStates,
      size_t alignment)
      : StoreOperation(
            CreateOperandTypes(std::move(storedType), numMemoryStates),
            CreateResultTypes(numMemoryStates),
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
      const variable * value,
      const variable * ioState,
      const variable * memoryState,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value->Type());

    StoreVolatileOperation op(storedType, 1, alignment);
    return tac::create(op, { address, value, ioState, memoryState });
  }

private:
  static std::shared_ptr<const rvsdg::ValueType>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::ValueType>(type))
      return storedType;

    throw jlm::util::error("Expected value type");
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(std::shared_ptr<const rvsdg::ValueType> storedType, size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types(
        { PointerType::Create(), std::move(storedType), IOStateType::Create() });
    std::vector<std::shared_ptr<const rvsdg::Type>> states(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), states.begin(), states.end());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateResultTypes(size_t numMemoryStates)
  {
    std::vector<std::shared_ptr<const rvsdg::Type>> types({ IOStateType::Create() });
    std::vector<std::shared_ptr<const rvsdg::Type>> memoryStates(
        numMemoryStates,
        MemoryStateType::Create());
    types.insert(types.end(), memoryStates.begin(), memoryStates.end());
    return types;
  }
};

/**
 * Represents a StoreVolatileOperation in an RVSDG.
 */
class StoreVolatileNode final : public StoreNode
{
  StoreVolatileNode(
      rvsdg::Region & region,
      std::unique_ptr<StoreVolatileOperation> operation,
      const std::vector<rvsdg::output *> & operands)
      : StoreNode(region, std::move(operation), operands)
  {}

public:
  [[nodiscard]] const StoreVolatileOperation &
  GetOperation() const noexcept override;

  [[nodiscard]] MemoryStateInputRange
  MemoryStateInputs() const noexcept override;

  [[nodiscard]] MemoryStateOutputRange
  MemoryStateOutputs() const noexcept override;

  [[nodiscard]] StoreVolatileNode &
  CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const override;

  [[nodiscard]] rvsdg::input &
  GetIoStateInput() const noexcept
  {
    auto ioStateInput = input(2);
    JLM_ASSERT(is<IOStateType>(ioStateInput->type()));
    return *ioStateInput;
  }

  [[nodiscard]] rvsdg::output &
  GetIoStateOutput() const noexcept
  {
    auto ioStateOutput = output(0);
    JLM_ASSERT(is<IOStateType>(ioStateOutput->type()));
    return *ioStateOutput;
  }

  Node *
  copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const override;

  static StoreVolatileNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<StoreVolatileOperation> storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new StoreVolatileNode(region, std::move(storeOperation), operands));
  }

  static StoreVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & value,
      rvsdg::output & ioState,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value.Type());

    std::vector<rvsdg::output *> operands({ &address, &value, &ioState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation =
        std::make_unique<StoreVolatileOperation>(storedType, memoryStates.size(), alignment);
    return CreateNode(*address.region(), std::move(operation), operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::Region & region,
      std::unique_ptr<StoreVolatileOperation> storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::outputs(&CreateNode(region, std::move(storeOperation), operands));
  }

private:
  static std::shared_ptr<const rvsdg::ValueType>
  CheckAndExtractStoredType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto storedType = std::dynamic_pointer_cast<const rvsdg::ValueType>(type))
      return storedType;

    throw jlm::util::error("Expected value type.");
  }
};

/**
 * \brief Swaps a memory state merge operation and a store operation.
 *
 * sx1 = MemStateMerge si1 ... siM
 * sl1 = StoreNonVolatile a v sx1
 * =>
 * sl1 ... slM = StoreNonVolatile a v si1 ... siM
 * sx1 = MemStateMerge sl1 ... slM
 *
 * FIXME: The reduction can be generalized: A store node can have multiple operands from different
 * merge nodes.
 *
 * @param operation The operation of the StoreNonVolatile node.
 * @param operands The operands of the StoreNonVolatile node.
 *
 * @return If the normalization could be applied, then the results of the store operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeStoreMux(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief Removes a duplicated store to the same address.
 *
 * so1 so2 = StoreNonVolatile a v1 si1 si2
 * sx1 sx2 = StoreNonVolatile a v2 so1 so2
 * =>
 * sx1 sx2 = StoreNonVolatile a v2 si1 si2
 *
 * @param operation The operation of the StoreNonVolatile node.
 * @param operands The operands of the StoreNonVolatile node.
 *
 * @return If the normalization could be applied, then the results of the store operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeStoreStore(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief Removes unnecessary state from a store node when its address originates directly from an
 * alloca node.
 *
 * a s = Alloca b
 * so1 so2 = StoreNonVolatile a v s si1 si2
 * ... = AnyOp so1 so2
 * =>
 * a s = Alloca b
 * so1 = StoreNonVolatile a v s
 * ... = AnyOp so1 so1
 *
 * @param operation The operation of the StoreNonVolatile node.
 * @param operands The operands of the StoreNonVolatile node.
 *
 * @return If the normalization could be applied, then the results of the store operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeStoreAlloca(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

/**
 * \brief Remove duplicated state operands
 *
 * so1 so2 so3 = StoreNonVolatile a v si1 si1 si1
 * =>
 * so1 = StoreNonVolatile a v si1
 *
 * @param operation The load operation on which the transformation is performed.
 * @param operands The operands of the load node.
 *
 * @return If the normalization could be applied, then the results of the load operation after
 * the transformation. Otherwise, std::nullopt.
 */
std::optional<std::vector<rvsdg::output *>>
NormalizeStoreDuplicateState(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands);

}

#endif
