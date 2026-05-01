/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_STDLIBINTRINSICOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_STDLIBINTRINSICOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Abstract base class for memcpy operations.
 *
 * @see MemCpyNonVolatileOperation
 * @see MemCpyVolatileOperation
 */
class MemCpyOperation : public rvsdg::SimpleOperation
{
protected:
  MemCpyOperation(
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes)
      : SimpleOperation(operandTypes, resultTypes)
  {
    JLM_ASSERT(operandTypes.size() >= 4);

    auto & dstAddressType = *operandTypes[0];
    JLM_ASSERT(is<PointerType>(dstAddressType));

    auto & srcAddressType = *operandTypes[1];
    JLM_ASSERT(is<PointerType>(srcAddressType));

    auto & lengthType = *operandTypes[2];
    if (lengthType != *rvsdg::BitType::Create(32) && lengthType != *rvsdg::BitType::Create(64))
    {
      throw util::Error("Expected 32 bit or 64 bit integer type.");
    }

    auto & memoryStateType = *operandTypes.back();
    if (!is<MemoryStateType>(memoryStateType))
    {
      throw util::Error("Number of memory states cannot be zero.");
    }
  }

public:
  [[nodiscard]] const rvsdg::BitType &
  LengthType() const noexcept
  {
    auto type = std::dynamic_pointer_cast<const rvsdg::BitType>(argument(2));
    JLM_ASSERT(type != nullptr);
    return *type;
  }

  [[nodiscard]] virtual size_t
  NumMemoryStates() const noexcept = 0;

  /**
   * @param node a SimpleNode containing a MemCpyOperation
   * @return the input of \p node that takes the pointer to store bytes to.
   */
  [[nodiscard]] static rvsdg::Input &
  destinationInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<MemCpyOperation>(&node));
    const auto input = node.input(0);
    JLM_ASSERT(is<PointerType>(input->Type()));
    return *input;
  }

  /**
   * @param node a SimpleNode containing a MemCpyOperation
   * @return the input of \p node that takes the pointer to load bytes from.
   */
  [[nodiscard]] static rvsdg::Input &
  sourceInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<MemCpyOperation>(&node));
    const auto input = node.input(1);
    JLM_ASSERT(is<PointerType>(input->Type()));
    return *input;
  }

  /**
   * @param node a SimpleNode containing a MemCpyOperation
   * @return the input of \p node that takes the number of bytes to copy.
   */
  [[nodiscard]] static rvsdg::Input &
  countInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<MemCpyOperation>(&node));
    const auto input = node.input(2);
    JLM_ASSERT(is<rvsdg::BitType>(input->Type()));
    return *input;
  }

  /**
   * Maps a memory state output to its corresponding memory state input.
   */
  [[nodiscard]] static rvsdg::Input &
  mapMemoryStateOutputToInput(const rvsdg::Output & output)
  {
    JLM_ASSERT(is<MemoryStateType>(output.Type()));
    auto [memCpyNode, memCpyOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<MemCpyOperation>(output);
    JLM_ASSERT(memCpyOperation);
    const auto numNonMemoryStateOutputs =
        memCpyNode->noutputs() - memCpyOperation->NumMemoryStates();
    JLM_ASSERT(output.index() >= numNonMemoryStateOutputs);
    const auto numNonMemoryStateInputs = memCpyNode->ninputs() - memCpyOperation->NumMemoryStates();
    const auto inputIndex = numNonMemoryStateInputs + (output.index() - numNonMemoryStateOutputs);
    const auto input = memCpyNode->input(inputIndex);
    JLM_ASSERT(is<MemoryStateType>(input->Type()));
    return *input;
  }

  /**
   * Maps a memory state input to its corresponding memory state output.
   */
  [[nodiscard]] static rvsdg::Output &
  mapMemoryStateInputToOutput(const rvsdg::Input & input)
  {
    JLM_ASSERT(is<MemoryStateType>(input.Type()));
    auto [memCpyNode, memCpyOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<MemCpyOperation>(input);
    JLM_ASSERT(memCpyOperation);
    const auto numNonMemoryStateInputs = memCpyNode->ninputs() - memCpyOperation->NumMemoryStates();
    JLM_ASSERT(input.index() >= numNonMemoryStateInputs);
    const auto numNonMemoryStateOutputs =
        memCpyNode->noutputs() - memCpyOperation->NumMemoryStates();
    const auto outputIndex = numNonMemoryStateOutputs + (input.index() - numNonMemoryStateInputs);
    const auto output = memCpyNode->output(outputIndex);
    JLM_ASSERT(is<MemoryStateType>(output->Type()));
    return *output;
  }
};

/**
 * Represents a non-volatile LLVM memcpy intrinsic.
 *
 * @see MemCpyVolatileOperation
 */
class MemCpyNonVolatileOperation final : public MemCpyOperation
{
public:
  ~MemCpyNonVolatileOperation() override;

  MemCpyNonVolatileOperation(std::shared_ptr<const rvsdg::Type> lengthType, size_t numMemoryStates)
      : MemCpyOperation(
            CreateOperandTypes(std::move(lengthType), numMemoryStates),
            CreateResultTypes(numMemoryStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(
      const Variable * destination,
      const Variable * source,
      const Variable * length,
      const std::vector<const Variable *> & memoryStates)
  {
    std::vector<const Variable *> operands = { destination, source, length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation =
        std::make_unique<MemCpyNonVolatileOperation>(length->Type(), memoryStates.size());
    return ThreeAddressCode::create(std::move(operation), operands);
  }

  static rvsdg::SimpleNode &
  createNode(
      rvsdg::Output & destination,
      rvsdg::Output & source,
      rvsdg::Output & length,
      const std::vector<rvsdg::Output *> & memoryStates)
  {
    std::vector operands = { &destination, &source, &length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    return rvsdg::CreateOpNode<MemCpyNonVolatileOperation>(
        operands,
        length.Type(),
        memoryStates.size());
  }

  static std::vector<rvsdg::Output *>
  create(
      rvsdg::Output * destination,
      rvsdg::Output * source,
      rvsdg::Output * length,
      const std::vector<rvsdg::Output *> & memoryStates)
  {
    std::vector operands = { destination, source, length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    return outputs(&rvsdg::CreateOpNode<MemCpyNonVolatileOperation>(
        operands,
        length->Type(),
        memoryStates.size()));
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(std::shared_ptr<const rvsdg::Type> length, size_t numMemoryStates)
  {
    auto pointerType = PointerType::Create();
    std::vector<std::shared_ptr<const rvsdg::Type>> types = { pointerType, pointerType, length };
    types.insert(types.end(), numMemoryStates, MemoryStateType::Create());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateResultTypes(size_t numMemoryStates)
  {
    // The memcpy() standard C library call has as return type void*, but LLVM models the
    // function call nevertheless as:
    // call void @llvm.memcpy.p0.p0.i64(ptr align 4 %7, ptr align 4 %8, i64 %9, i1 false)
    //
    // LLVM simply hands in register %7 (dest pointer) to all users of the return type of memcpy().
    // Thus, we only need state types as result types of the operation here.
    return { numMemoryStates, MemoryStateType::Create() };
  }
};

/**
 * Represents a volatile LLVM memcpy intrinsic
 *
 * In contrast to LLVM, a volatile memcpy requires in an RVSDG setting an I/O state as it
 * incorporates externally visible side-effects. This I/O state allows the volatile memcpy operation
 * to be sequentialized with respect to other volatile memory accesses and I/O operations. This
 * additional I/O state is the main reason why volatile memcpys are modeled as its own operation and
 * volatile is not just a flag at the normal \ref MemCpyNonVolatileOperation.
 *
 * @see MemCpyNonVolatileOperation
 */
class MemCpyVolatileOperation final : public MemCpyOperation
{
public:
  ~MemCpyVolatileOperation() noexcept override;

  MemCpyVolatileOperation(std::shared_ptr<const rvsdg::Type> lengthType, size_t numMemoryStates)
      : MemCpyOperation(
            CreateOperandTypes(std::move(lengthType), numMemoryStates),
            CreateResultTypes(numMemoryStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  CreateThreeAddressCode(
      const Variable & destination,
      const Variable & source,
      const Variable & length,
      const Variable & ioState,
      const std::vector<const Variable *> & memoryStates)
  {
    std::vector<const Variable *> operands = { &destination, &source, &length, &ioState };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation = std::make_unique<MemCpyVolatileOperation>(length.Type(), memoryStates.size());
    return ThreeAddressCode::create(std::move(operation), operands);
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Output & destination,
      rvsdg::Output & source,
      rvsdg::Output & length,
      rvsdg::Output & ioState,
      const std::vector<rvsdg::Output *> & memoryStates)
  {
    std::vector operands = { &destination, &source, &length, &ioState };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    return rvsdg::CreateOpNode<MemCpyVolatileOperation>(
        operands,
        length.Type(),
        memoryStates.size());
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateOperandTypes(std::shared_ptr<const rvsdg::Type> lengthType, size_t numMemoryStates)
  {
    auto pointerType = PointerType::Create();
    std::vector<std::shared_ptr<const rvsdg::Type>> types = { pointerType,
                                                              pointerType,
                                                              std::move(lengthType),
                                                              IOStateType::Create() };
    types.insert(types.end(), numMemoryStates, MemoryStateType::Create());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  CreateResultTypes(size_t numMemoryStates)
  {
    // The memcpy() standard C library call has as return type void*, but LLVM models the
    // function call nevertheless as:
    // call void @llvm.memcpy.p0.p0.i64(ptr align 4 %7, ptr align 4 %8, i64 %9, i1 false)
    //
    // LLVM simply hands in register %7 (dest pointer) to all users of the return type of memcpy().
    // Thus, we only need state types as result types of the operation here.
    std::vector<std::shared_ptr<const rvsdg::Type>> types(1, IOStateType::Create());
    types.insert(types.end(), numMemoryStates, MemoryStateType::Create());
    return types;
  }
};

/**
 * Abstract base class for memset operations.
 *
 * @see MemSetNonVolatileOperation
 */
class MemSetOperation : public rvsdg::SimpleOperation
{
protected:
  MemSetOperation(
      const std::vector<std::shared_ptr<const rvsdg::Type>> & operandTypes,
      const std::vector<std::shared_ptr<const rvsdg::Type>> & resultTypes)
      : SimpleOperation(operandTypes, resultTypes)
  {
    JLM_ASSERT(operandTypes.size() >= 4);

    auto & dstAddressType = *operandTypes[0];
    JLM_ASSERT(is<PointerType>(dstAddressType));

    if (auto & valueType = *operandTypes[1]; valueType != *rvsdg::BitType::Create(8))
    {
      throw std::runtime_error("Expected 8 bit integer type.");
    }

    if (auto & lengthType = *operandTypes[2];
        lengthType != *rvsdg::BitType::Create(32) && lengthType != *rvsdg::BitType::Create(64))
    {
      throw util::Error("Expected 32 bit or 64 bit integer type.");
    }

    if (auto & memoryStateType = *operandTypes.back(); !is<MemoryStateType>(memoryStateType))
    {
      throw util::Error("Number of memory states cannot be zero.");
    }
  }

public:
  /**
   * @return The type of the length argument
   */
  [[nodiscard]] const rvsdg::BitType &
  lengthType() const noexcept
  {
    const auto type = std::dynamic_pointer_cast<const rvsdg::BitType>(argument(2));
    JLM_ASSERT(type != nullptr);
    JLM_ASSERT(type->nbits() == 32 || type->nbits() == 64);
    return *type;
  }

  /**
   * @return Number of memory states
   */
  [[nodiscard]] virtual size_t
  numMemoryStates() const noexcept = 0;

  /**
   * @param node a SimpleNode containing a MemSetOperation
   * @return the input of \p node that points to the destination to fill.
   */
  [[nodiscard]] static rvsdg::Input &
  destinationInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<MemSetOperation>(&node));
    const auto input = node.input(0);
    JLM_ASSERT(is<PointerType>(input->Type()));
    return *input;
  }

  /**
   * @param node a SimpleNode containing a MemSetOperation
   * @return the input of \p node that points to the byte value with which to fill the memory
   */
  [[nodiscard]] static rvsdg::Input &
  valueInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<MemSetOperation>(&node));
    const auto input = node.input(1);
    JLM_ASSERT(*input->Type() == *rvsdg::BitType::Create(8));
    return *input;
  }

  /**
   * @param node a SimpleNode containing a MemSetOperation
   * @return the input of \p node that points to the number of bytes that need to be filled.
   */
  [[nodiscard]] static rvsdg::Input &
  lengthInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(is<MemSetOperation>(&node));
    const auto input = node.input(2);
    JLM_ASSERT(
        *input->Type() == *rvsdg::BitType::Create(32)
        || *input->Type() == *rvsdg::BitType::Create(64));
    return *input;
  }
};

/**
 * Represents a non-volatile LLVM memset intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-memset-inline-intrinsic) for more details.
 */
class MemSetNonVolatileOperation final : public MemSetOperation
{
public:
  ~MemSetNonVolatileOperation() noexcept override;

  MemSetNonVolatileOperation(std::shared_ptr<const rvsdg::Type> lengthType, size_t numMemoryStates)
      : MemSetOperation(
            createOperandTypes(std::move(lengthType), numMemoryStates),
            createResultTypes(numMemoryStates))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] size_t
  numMemoryStates() const noexcept override;

  static std::unique_ptr<ThreeAddressCode>
  createTac(
      const Variable & destination,
      const Variable & value,
      const Variable & length,
      const std::vector<const Variable *> & memoryStates)
  {
    std::vector operands = { &destination, &value, &length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    auto operation =
        std::make_unique<MemCpyNonVolatileOperation>(length.Type(), memoryStates.size());
    return ThreeAddressCode::create(std::move(operation), operands);
  }

  static rvsdg::SimpleNode &
  createNode(
      rvsdg::Output & destination,
      rvsdg::Output & value,
      rvsdg::Output & length,
      const std::vector<rvsdg::Output *> & memoryStates)
  {
    std::vector operands = { &destination, &value, &length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    return rvsdg::CreateOpNode<MemSetNonVolatileOperation>(
        operands,
        length.Type(),
        memoryStates.size());
  }

private:
  static std::vector<std::shared_ptr<const rvsdg::Type>>
  createOperandTypes(
      const std::shared_ptr<const rvsdg::Type> & lengthType,
      const size_t numMemoryStates)
  {
    const auto pointerType = PointerType::Create();
    const auto & valueType = rvsdg::BitType::Create(8);
    std::vector<std::shared_ptr<const rvsdg::Type>> types = { pointerType, valueType, lengthType };
    types.insert(types.end(), numMemoryStates, MemoryStateType::Create());
    return types;
  }

  static std::vector<std::shared_ptr<const rvsdg::Type>>
  createResultTypes(size_t numMemoryStates)
  {
    // The memset() standard C library call has as return type void*, but LLVM models the
    // function call nevertheless as:
    // call void @llvm.memset.p0.p0.i64(ptr align 4 %7, ptr align 4 %8, i64 %9, i1 false)
    //
    // LLVM simply hands in register %7 (dest pointer) to all users of the return type of memset().
    // Thus, we only need state types as result types of the operation here.
    return { numMemoryStates, MemoryStateType::Create() };
  }
};

}

#endif
