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
    if (lengthType != *rvsdg::bittype::Create(32) && lengthType != *rvsdg::bittype::Create(64))
    {
      throw util::error("Expected 32 bit or 64 bit integer type.");
    }

    auto & memoryStateType = *operandTypes.back();
    if (!is<MemoryStateType>(memoryStateType))
    {
      throw util::error("Number of memory states cannot be zero.");
    }
  }

public:
  [[nodiscard]] const rvsdg::bittype &
  LengthType() const noexcept
  {
    auto type = std::dynamic_pointer_cast<const rvsdg::bittype>(argument(2));
    JLM_ASSERT(type != nullptr);
    return *type;
  }

  [[nodiscard]] virtual size_t
  NumMemoryStates() const noexcept = 0;
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
      const variable * destination,
      const variable * source,
      const variable * length,
      const std::vector<const variable *> & memoryStates)
  {
    std::vector<const variable *> operands = { destination, source, length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyNonVolatileOperation operation(length->Type(), memoryStates.size());
    return ThreeAddressCode::create(operation, operands);
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
      const variable & destination,
      const variable & source,
      const variable & length,
      const variable & ioState,
      const std::vector<const variable *> & memoryStates)
  {
    std::vector<const variable *> operands = { &destination, &source, &length, &ioState };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyVolatileOperation operation(length.Type(), memoryStates.size());
    return ThreeAddressCode::create(operation, operands);
  }

  static rvsdg::SimpleNode &
  CreateNode(
      rvsdg::Output & destination,
      rvsdg::Output & source,
      rvsdg::Output & length,
      rvsdg::Output & ioState,
      const std::vector<rvsdg::Output *> & memoryStates)
  {
    std::vector<rvsdg::Output *> operands = { &destination, &source, &length, &ioState };
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
