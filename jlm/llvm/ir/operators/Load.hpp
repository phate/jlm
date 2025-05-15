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

#include <llvm/IR/Instructions.h>

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
  class MemoryStateInputIterator final : public rvsdg::input::iterator<rvsdg::SimpleInput>
  {
  public:
    virtual ~MemoryStateInputIterator() = default;

    constexpr explicit MemoryStateInputIterator(rvsdg::SimpleInput * input)
        : rvsdg::input::iterator<rvsdg::SimpleInput>(input)
    {}

    [[nodiscard]] rvsdg::SimpleInput *
    next() const override
    {
      const auto index = value()->index();
      const auto node = value()->node();

      return index + 1 < node->ninputs() ? node->input(index + 1) : nullptr;
    }
  };

  class MemoryStateOutputIterator final : public rvsdg::output::iterator<rvsdg::SimpleOutput>
  {
  public:
    virtual ~MemoryStateOutputIterator() = default;

    constexpr explicit MemoryStateOutputIterator(rvsdg::SimpleOutput * output)
        : rvsdg::output::iterator<rvsdg::SimpleOutput>(output)
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
      ::llvm::LoadInst * llvmLoad,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes,
      const size_t numMemoryStates,
      const size_t alignment)
      : SimpleOperation(operandTypes, resultTypes),
        LlvmLoad_(llvmLoad),
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
  [[nodiscard]] ::llvm::LoadInst *
    GetLlvmLoad() const noexcept
  {
    return LlvmLoad_;
  }

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

  [[nodiscard]] static rvsdg::input &
  AddressInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<LoadOperation>(&node));
    const auto input = node.input(0);
    JLM_ASSERT(is<PointerType>(input->Type()));
    return *input;
  }

  [[nodiscard]] static rvsdg::output &
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

private:
  ::llvm::LoadInst * LlvmLoad_;
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
      ::llvm::LoadInst * llvmLoad,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
            llvmLoad,
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

  [[nodiscard]] static rvsdg::input &
  IOStateInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<LoadVolatileOperation>(&node));
    const auto input = node.input(1);
    JLM_ASSERT(is<IOStateType>(input->Type()));
    return *input;
  }

  [[nodiscard]] static rvsdg::output &
  IOStateOutput(const rvsdg::Node & node)
  {
    JLM_ASSERT(is<LoadVolatileOperation>(&node));
    const auto output = node.output(1);
    JLM_ASSERT(is<IOStateType>(output->Type()));
    return *output;
  }

  static std::unique_ptr<llvm::tac>
  Create(
      ::llvm::LoadInst * llvmLoad,
      const variable * address,
      const variable * iOState,
      const variable * memoryState,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    LoadVolatileOperation operation(llvmLoad, std::move(loadedType), 1, alignment);
    return tac::create(operation, { address, iOState, memoryState });
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadVolatileOperation> loadOperation,
      const std::vector<rvsdg::output *> & operands);

  static rvsdg::SimpleNode &
  CreateNode(
      ::llvm::LoadInst * llvmLoad,
      rvsdg::output & address,
      rvsdg::output & iOState,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    std::vector operands({ &address, &iOState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<LoadVolatileOperation>(
        llvmLoad,
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
      ::llvm::LoadInst * llvmLoad,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : LoadOperation(
            llvmLoad,
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

  static std::unique_ptr<llvm::tac>
  Create(
      ::llvm::LoadInst * llvmLoad,
      const variable * address,
      const variable * state,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    LoadNonVolatileOperation operation(llvmLoad, std::move(loadedType), 1, alignment);
    return tac::create(operation, { address, state });
  }

  static std::vector<rvsdg::output *>
  Create(
      ::llvm::LoadInst * llvmLoad,
      rvsdg::output * address,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      const size_t alignment)
  {
    return rvsdg::outputs(&CreateNode(llvmLoad, *address, memoryStates, std::move(loadedType), alignment));
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Region & region,
      std::unique_ptr<LoadNonVolatileOperation> loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return rvsdg::SimpleNode::Create(region, std::move(loadOperation), operands);
  }

  static std::vector<rvsdg::output *>
  Create(
      rvsdg::Region & region,
      std::unique_ptr<LoadNonVolatileOperation> loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return outputs(&CreateNode(region, std::move(loadOperation), operands));
  }

  static rvsdg::SimpleNode &
  CreateNode(
      ::llvm::LoadInst * llvmLoad,
      rvsdg::output & address,
      const std::vector<rvsdg::output *> & memoryStates,
      std::shared_ptr<const rvsdg::ValueType> loadedType,
      size_t alignment)
  {
    std::vector operands({ &address });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<LoadNonVolatileOperation>(
        llvmLoad,
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
