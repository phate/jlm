/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_STOREVOLATILE_HPP
#define JLM_LLVM_IR_OPERATORS_STOREVOLATILE_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents a volatile LLVM store instruction.
 *
 * In contrast to LLVM, a volatile store requires in an RVSDG setting an I/O state as it
 * incorporates externally visible side-effects. This I/O state allows the volatile store operation
 * to be sequentialized with respect to other volatile memory accesses and I/O operations. This
 * additional I/O state is the main reason why volatile stores are modeled as its own operation and
 * volatile is not just a flag at the normal \ref StoreOperation.
 *
 * @see LoadVolatileOperation
 */
class StoreVolatileOperation final : public rvsdg::simple_op
{
public:
  ~StoreVolatileOperation() noexcept override;

  StoreVolatileOperation(
      const rvsdg::valuetype & storedType,
      size_t numMemoryStates,
      size_t alignment)
      : simple_op(
            CreateOperandPorts(storedType, numMemoryStates),
            CreateResultPorts(numMemoryStates)),
        Alignment_(alignment)
  {}

  bool
  operator==(const operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<rvsdg::operation>
  copy() const override;

  [[nodiscard]] const jlm::rvsdg::valuetype &
  GetStoredType() const noexcept
  {
    return *util::AssertedCast<const jlm::rvsdg::valuetype>(&argument(1).type());
  }

  [[nodiscard]] size_t
  NumMemoryStates() const noexcept
  {
    return nresults() - 1;
  }

  [[nodiscard]] size_t
  GetAlignment() const noexcept
  {
    return Alignment_;
  }

  static std::unique_ptr<llvm::tac>
  Create(
      const variable * address,
      const variable * value,
      const variable * ioState,
      const variable * memoryState,
      size_t alignment)
  {
    auto & storedType = CheckAndExtractStoredType(value->type());

    StoreVolatileOperation op(storedType, 1, alignment);
    return tac::create(op, { address, value, ioState, memoryState });
  }

private:
  static const rvsdg::valuetype &
  CheckAndExtractStoredType(const rvsdg::type & type)
  {
    if (auto storedType = dynamic_cast<const rvsdg::valuetype *>(&type))
      return *storedType;

    throw jlm::util::error("Expected value type");
  }

  static std::vector<rvsdg::port>
  CreateOperandPorts(const rvsdg::valuetype & storedType, size_t numStates)
  {
    std::vector<rvsdg::port> ports({ PointerType(), storedType, iostatetype() });
    std::vector<rvsdg::port> states(numStates, { MemoryStateType() });
    ports.insert(ports.end(), states.begin(), states.end());
    return ports;
  }

  static std::vector<rvsdg::port>
  CreateResultPorts(size_t numMemoryStates)
  {
    std::vector<rvsdg::port> ports({ iostatetype() });
    std::vector<rvsdg::port> memoryStates(numMemoryStates, { MemoryStateType() });
    ports.insert(ports.end(), memoryStates.begin(), memoryStates.end());
    return ports;
  }

  size_t Alignment_;
};

/**
 * Represents a StoreVolatileOperation in an RVSDG.
 */
class StoreVolatileNode final : public rvsdg::simple_node
{
  StoreVolatileNode(
      rvsdg::region & region,
      const StoreVolatileOperation & operation,
      const std::vector<rvsdg::output *> & operands)
      : simple_node(&region, operation, operands)
  {}

public:
  [[nodiscard]] const StoreVolatileOperation &
  GetOperation() const noexcept
  {
    return *util::AssertedCast<const StoreVolatileOperation>(&operation());
  }

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
  GetValueInput() const noexcept
  {
    auto valueInput = input(1);
    JLM_ASSERT(is<rvsdg::valuetype>(valueInput->type()));
    return *valueInput;
  }

  [[nodiscard]] rvsdg::input &
  GetIoStateInput() const noexcept
  {
    auto ioStateInput = input(2);
    JLM_ASSERT(is<iostatetype>(ioStateInput->type()));
    return *ioStateInput;
  }

  [[nodiscard]] rvsdg::output &
  GetIoStateOutput() const noexcept
  {
    auto ioStateOutput = output(0);
    JLM_ASSERT(is<iostatetype>(ioStateOutput->type()));
    return *ioStateOutput;
  }

  rvsdg::node *
  copy(rvsdg::region * region, const std::vector<rvsdg::output *> & operands) const override;

  static StoreVolatileNode &
  CreateNode(
      rvsdg::region & region,
      const StoreVolatileOperation & storeOperation,
      const std::vector<rvsdg::output *> & operands)
  {
    return *(new StoreVolatileNode(region, storeOperation, operands));
  }

  static StoreVolatileNode &
  CreateNode(
      rvsdg::output & address,
      rvsdg::output & value,
      rvsdg::output & ioState,
      const std::vector<rvsdg::output *> & memoryStates,
      size_t alignment)
  {
    auto & storedType = CheckAndExtractStoredType(value.type());

    std::vector<rvsdg::output *> operands({ &address, &value, &ioState });
    operands.insert(operands.end(), memoryStates.begin(), memoryStates.end());

    StoreVolatileOperation storeOperation(storedType, memoryStates.size(), alignment);
    return CreateNode(*address.region(), storeOperation, operands);
  }

private:
  static const rvsdg::valuetype &
  CheckAndExtractStoredType(const rvsdg::type & type)
  {
    if (auto storedType = dynamic_cast<const rvsdg::valuetype *>(&type))
      return *storedType;

    throw jlm::util::error("Expected value type.");
  }
};

}

#endif // JLM_LLVM_IR_OPERATORS_STOREVOLATILE_HPP
