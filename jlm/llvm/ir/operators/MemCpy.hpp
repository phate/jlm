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
 * Represents the LLVM memcpy intrinsic.
 */
class MemCpyOperation final : public rvsdg::simple_op
{
public:
  ~MemCpyOperation() override;

  MemCpyOperation(
      const std::vector<rvsdg::port> & operandPorts,
      const std::vector<rvsdg::port> & resultPorts)
      : simple_op(operandPorts, resultPorts)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  static std::unique_ptr<llvm::tac>
  create(
      const variable * destination,
      const variable * source,
      const variable * length,
      const variable * isVolatile,
      const std::vector<const variable *> & memoryStates)
  {
    auto operandPorts = CheckAndCreateOperandPorts(length->type(), memoryStates.size());
    auto resultPorts = CreateResultPorts(memoryStates.size());

    std::vector<const variable *> operands = { destination, source, length, isVolatile };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyOperation op(operandPorts, resultPorts);
    return tac::create(op, operands);
  }

  static std::vector<rvsdg::output *>
  create(
      rvsdg::output * destination,
      rvsdg::output * source,
      rvsdg::output * length,
      rvsdg::output * isVolatile,
      const std::vector<rvsdg::output *> & memoryStates)
  {
    auto operandPorts = CheckAndCreateOperandPorts(length->type(), memoryStates.size());
    auto resultPorts = CreateResultPorts(memoryStates.size());

    std::vector<rvsdg::output *> operands = { destination, source, length, isVolatile };
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    MemCpyOperation op(operandPorts, resultPorts);
    return rvsdg::simple_node::create_normalized(destination->region(), op, operands);
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
    std::vector<rvsdg::port> ports = { pointerType, pointerType, length, rvsdg::bit1 };
    ports.insert(ports.end(), numMemoryStates, { MemoryStateType::Create() });

    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(size_t nMemoryStates)
  {
    return std::vector<rvsdg::port>(nMemoryStates, { MemoryStateType::Create() });
  }
};

}

#endif // JLM_LLVM_IR_OPERATORS_MEMCPY_HPP
