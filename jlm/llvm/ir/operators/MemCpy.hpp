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
 * Represents an LLVM memcpy intrinsic.
 *
 * @see MemCpyVolatileOperation
 */
class MemCpyOperation final : public rvsdg::simple_op
{
public:
  ~MemCpyOperation() override;

  MemCpyOperation(const rvsdg::type & lengthType, size_t numMemoryStates)
      : simple_op(
          CheckAndCreateOperandPorts(lengthType, numMemoryStates),
          CreateResultPorts(numMemoryStates))
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  [[nodiscard]] const rvsdg::bittype &
  LengthType() const noexcept
  {
    auto type = dynamic_cast<const rvsdg::bittype *>(&argument(2).type());
    JLM_ASSERT(type != nullptr);
    return *type;
  }

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept
  {
    return nresults();
  }

  static std::unique_ptr<llvm::tac>
  create(
      const variable * destination,
      const variable * source,
      const variable * length,
      const std::vector<const variable *> & memoryStates)
  {
    std::vector<const variable *> operands = { destination, source, length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyOperation operation(length->type(), memoryStates.size());
    return tac::create(operation, operands);
  }

  static std::vector<rvsdg::output *>
  create(
      rvsdg::output * destination,
      rvsdg::output * source,
      rvsdg::output * length,
      const std::vector<rvsdg::output *> & memoryStates)
  {
    std::vector<rvsdg::output *> operands = { destination, source, length };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyOperation operation(length->type(), memoryStates.size());
    return rvsdg::simple_node::create_normalized(destination->region(), operation, operands);
  }

private:
  static std::vector<rvsdg::port>
  CheckAndCreateOperandPorts(const rvsdg::type & length, size_t numMemoryStates)
  {
    if (length != rvsdg::bit32 && length != rvsdg::bit64)
      throw util::error("Expected 32 bit or 64 bit integer type.");

    if (numMemoryStates == 0)
      throw util::error("Number of memory states cannot be zero.");

    PointerType pointerType;
    std::vector<rvsdg::port> ports = { pointerType, pointerType, length };
    ports.insert(ports.end(), numMemoryStates, { MemoryStateType::Create() });

    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(size_t nMemoryStates)
  {
    return std::vector<rvsdg::port>(nMemoryStates, { MemoryStateType::Create() });
  }
};

/**
 * Represents a volatile LLVM memcpy intrinsic
 *
 * In contrast to LLVM, a volatile memcpy requires in an RVSDG setting an I/O state as it
 * incorporates externally visible side-effects. This I/O state allows the volatile memcpy operation
 * to be sequentialized with respect to other volatile memory accesses and I/O operations. This
 * additional I/O state is the main reason why volatile memcpys are modeled as its own operation and
 * volatile is not just a flag at the normal \ref MemCpyOperation.
 *
 * @see MemCpyOperation
 */
class MemCpyVolatileOperation final : public rvsdg::simple_op
{
public:
  ~MemCpyVolatileOperation() noexcept override;

  MemCpyVolatileOperation(const rvsdg::type & lengthType, size_t numMemoryStates)
      : rvsdg::simple_op(
          CheckAndCreateOperandPorts(lengthType, numMemoryStates),
          CreateResultPorts(numMemoryStates))
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  CreateThreeAddressCode(
      const variable & destination,
      const variable & source,
      const variable & length,
      const variable & ioState,
      const std::vector<const variable *> & memoryStates)
  {
    std::vector<const variable *> operands = { &destination, &source, &length, &ioState };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyVolatileOperation operation(length.type(), memoryStates.size());
    return tac::create(operation, operands);
  }

  static rvsdg::simple_node &
  CreateNode(
      rvsdg::output & destination,
      rvsdg::output & source,
      rvsdg::output & length,
      rvsdg::output & ioState,
      const std::vector<rvsdg::output *> & memoryStates)
  {
    std::vector<rvsdg::output *> operands = { &destination, &source, &length, &ioState };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyVolatileOperation operation(length.type(), memoryStates.size());
    return *rvsdg::simple_node::create(destination.region(), operation, operands);
  }

private:
  static std::vector<rvsdg::port>
  CheckAndCreateOperandPorts(const rvsdg::type & lengthType, size_t numMemoryStates)
  {
    if (lengthType != rvsdg::bit32 && lengthType != rvsdg::bit64)
      throw util::error("Expected 32 bit or 64 bit integer type.");

    if (numMemoryStates == 0)
      throw util::error("Number of memory states cannot be zero.");

    PointerType pointerType;
    std::vector<rvsdg::port> ports = { pointerType, pointerType, lengthType, iostatetype() };
    ports.insert(ports.end(), numMemoryStates, { MemoryStateType() });
    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(size_t numMemoryStates)
  {
    std::vector<rvsdg::port> ports(1, iostatetype());
    ports.insert(ports.end(), numMemoryStates, { MemoryStateType() });
    return ports;
  }
};

}

#endif // JLM_LLVM_IR_OPERATORS_MEMCPY_HPP
