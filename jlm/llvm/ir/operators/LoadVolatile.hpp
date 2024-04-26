/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_LOADVOLATILE_HPP
#define JLM_LLVM_IR_OPERATORS_LOADVOLATILE_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents a volatile LLVM load instruction.
 *
 * In contrast to LLVM, a volatile load requires in an RVSDG setting an I/O state as it incorporates
 * externally visible side-effects. This I/O state allows the volatile load operation to be
 * sequentialized with respect to other volatile memory accesses and I/O operations. This additional
 * I/O state is the main reason why volatile loads are modeled as its own operation and volatile is
 * not just a flag at the normal \ref LoadOperation.
 */
class LoadVolatileOperation final : public rvsdg::simple_op
{
public:
  ~LoadVolatileOperation() noexcept override;

  LoadVolatileOperation(
      const rvsdg::valuetype & loadedType,
      size_t numMemoryStates,
      size_t alignment)
      : simple_op(
          CreateOperandPorts(numMemoryStates),
          CreateResultPorts(loadedType, numMemoryStates)),
        Alignment_(alignment)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return Alignment_;
  }

  [[nodiscard]] const rvsdg::valuetype &
  GetLoadedType() const noexcept
  {
    return *util::AssertedCast<const rvsdg::valuetype>(&result(0).type());
  }

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept
  {
    return narguments() - 2;
  }

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * address,
      const variable * iOState,
      const variable * memoryState,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    LoadVolatileOperation operation(loadedType, 1, alignment);
    return tac::create(operation, { address, iOState, memoryState });
  }

private:
  static std::vector<rvsdg::port>
  CreateOperandPorts(size_t numStates)
  {
    std::vector<rvsdg::port> ports({ PointerType(), iostatetype() });
    std::vector<rvsdg::port> states(numStates, { MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(const rvsdg::valuetype & loadedType, size_t numStates)
  {
    std::vector<rvsdg::port> ports({ loadedType, iostatetype() });
    std::vector<rvsdg::port> states(numStates, { MemoryStateType::Create() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  size_t Alignment_;
};

/**
 * Represents a LoadVolatileOperation in an RVSDG.
 */
class LoadVolatileNode final : public rvsdg::simple_node
{
private:
  LoadVolatileNode(
      rvsdg::region & region,
      const LoadVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

public:
  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  [[nodiscard]] const LoadVolatileOperation &
  GetOperation() const noexcept
  {
    return *util::AssertedCast<const LoadVolatileOperation>(&operation());
  }

  [[nodiscard]] rvsdg::input &
  GetAddressInput() const noexcept
  {
    auto addressInput = input(0);
    JLM_ASSERT(is<PointerType>(addressInput->type()));
    return *addressInput;
  }

  [[nodiscard]] rvsdg::input &
  GetIoStateInput() const noexcept
  {
    auto ioInput = input(1);
    JLM_ASSERT(is<iostatetype>(ioInput->type()));
    return *ioInput;
  }

  [[nodiscard]] rvsdg::output &
  GetLoadedValueOutput() const noexcept
  {
    auto valueOutput = output(0);
    JLM_ASSERT(is<rvsdg::valuetype>(valueOutput->type()));
    return *valueOutput;
  }

  static LoadVolatileNode &
  CreateNode(
      rvsdg::region & region,
      const LoadVolatileOperation & loadOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new LoadVolatileNode(region, loadOperation, operands));
  }

  static LoadVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & iOState,
      const std::vector<rvsdg::output *> & memoryStates,
      const rvsdg::valuetype & loadedType,
      size_t alignment)
  {
    std::vector<rvsdg::output *> operands({ &address, &iOState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    LoadVolatileOperation operation(loadedType, memoryStates.size(), alignment);
    return CreateNode(*address.region(), operation, operands);
  }
};

}

#endif // JLM_LLVM_IR_OPERATORS_LOADVOLATILE_HPP
