/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_MEMCPY_HPP
#define JLM_LLVM_IR_OPERATORS_MEMCPY_HPP

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
    std::vector<std::shared_ptr<const rvsdg::Type>> types(1, IOStateType::Create());
    types.insert(types.end(), numMemoryStates, MemoryStateType::Create());
    return types;
  }
};

}

#endif // JLM_LLVM_IR_OPERATORS_MEMCPY_HPP
