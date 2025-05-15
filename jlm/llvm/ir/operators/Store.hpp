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

#include <llvm/IR/Instructions.h>

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
public:
  class MemoryStateOutputIterator final : public rvsdg::output::iterator<rvsdg::SimpleOutput>
  {
  public:
    constexpr explicit MemoryStateOutputIterator(rvsdg::SimpleOutput * output)
        : rvsdg::output::iterator<rvsdg::SimpleOutput>(output)
    {}

    [[nodiscard]] rvsdg::SimpleOutput *
    next() const override
    {
      const auto index = value()->index();
      const auto nextIndex = index + 1;
      const auto node = value()->node();

      return nextIndex < node->noutputs() ? node->output(nextIndex) : nullptr;
    }
  };

  using MemoryStateOutputRange = util::IteratorRange<MemoryStateOutputIterator>;

protected:
  StoreOperation(
      ::llvm::StoreInst * llvmStore,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes,
      size_t numMemoryStates,
      size_t alignment)
      : SimpleOperation(operandTypes, resultTypes),
        LlvmStore_(llvmStore),
        NumMemoryStates_(numMemoryStates),
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
  [[nodiscard]] ::llvm::StoreInst *
  GetLlvmStore() const noexcept
  {
    return LlvmStore_;
  }

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

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept
  {
    return NumMemoryStates_;
  }

  [[nodiscard]] static rvsdg::input &
  AddressInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<StoreOperation>(&node));
    auto & input = *node.input(0);
    JLM_ASSERT(is<PointerType>(input.Type()));
    return input;
  }

  [[nodiscard]] static rvsdg::input &
  StoredValueInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<StoreOperation>(&node));
    auto & input = *node.input(1);
    JLM_ASSERT(is<rvsdg::ValueType>(input.Type()));
    return input;
  }

  [[nodiscard]] static MemoryStateOutputRange
  MemoryStateOutputs(const rvsdg::SimpleNode & node) noexcept
  {
    const auto storeOperation = util::AssertedCast<const StoreOperation>(&node.GetOperation());
    if (storeOperation->NumMemoryStates_ == 0)
    {
      return { MemoryStateOutputIterator(nullptr), MemoryStateOutputIterator(nullptr) };
    }

    const auto firstMemoryStateOutput =
        node.output(storeOperation->nresults() - storeOperation->NumMemoryStates_);
    JLM_ASSERT(is<MemoryStateType>(firstMemoryStateOutput->Type()));
    return { MemoryStateOutputIterator(firstMemoryStateOutput),
             MemoryStateOutputIterator(nullptr) };
  }

private:
  ::llvm::StoreInst * LlvmStore_;
  size_t NumMemoryStates_;
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
      ::llvm::StoreInst * llvmStore,
      std::shared_ptr<const rvsdg::ValueType> storedType,
      const size_t numMemoryStates,
      const size_t alignment)
      : StoreOperation(
            llvmStore,
            CreateOperandTypes(std::move(storedType), numMemoryStates),
            { numMemoryStates, MemoryStateType::Create() },
            numMemoryStates,
            alignment)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  Create(::llvm::StoreInst * llvmStore, const variable * address, const variable * value, const variable * state, size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value->Type());

    StoreNonVolatileOperation op(llvmStore, storedType, 1, alignment);
    return tac::create(op, { address, value, state });
  }

  static std::vector<rvsdg::output *>
  Create(
      ::llvm::StoreInst * llvmStore,
      rvsdg::output * address,
      rvsdg::output * value,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    return outputs(&CreateNode(llvmStore, *address, *value, memoryStates, alignment));
  }

  static rvsdg::SimpleNode &
  CreateNode(
      ::llvm::StoreInst * llvmStore,
      rvsdg::output & address,
      rvsdg::output & value,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value.Type());

    std::vector operands({ &address, &value });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<StoreNonVolatileOperation>(
        llvmStore,
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
    return outputs(&CreateNode(region, std::move(storeOperation), operands));
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<StoreNonVolatileOperation> storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::SimpleNode::Create(region, std::move(storeOperation), operands);
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
      ::llvm::StoreInst * llvmStore,
      std::shared_ptr<const rvsdg::ValueType> storedType,
      const size_t numMemoryStates,
      const size_t alignment)
      : StoreOperation(
            llvmStore,
            CreateOperandTypes(std::move(storedType), numMemoryStates),
            CreateResultTypes(numMemoryStates),
            numMemoryStates,
            alignment)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] static rvsdg::input &
  IOStateInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<StoreOperation>(&node));
    auto & input = *node.input(2);
    JLM_ASSERT(is<IOStateType>(input.Type()));
    return input;
  }

  [[nodiscard]] static rvsdg::output &
  IOStateOutput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<StoreOperation>(&node));
    auto & output = *node.output(0);
    JLM_ASSERT(is<IOStateType>(output.Type()));
    return output;
  }

  static std::unique_ptr<llvm::tac>
  Create(
      ::llvm::StoreInst * llvmStore,
      const variable * address,
      const variable * value,
      const variable * ioState,
      const variable * memoryState,
      size_t alignment)
  {
    auto storedType = CheckAndExtractStoredType(value->Type());

    StoreVolatileOperation op(llvmStore, storedType, 1, alignment);
    return tac::create(op, { address, value, ioState, memoryState });
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<StoreVolatileOperation> storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::SimpleNode::Create(region, std::move(storeOperation), operands);
  }

  static rvsdg::SimpleNode &
  CreateNode(
      ::llvm::StoreInst * llvmStore,
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
        std::make_unique<StoreVolatileOperation>(llvmStore, storedType, memoryStates.size(), alignment);
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
