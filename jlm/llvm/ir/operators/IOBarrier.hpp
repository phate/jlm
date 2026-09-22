/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_IOBARRIER_HPP
#define JLM_LLVM_IR_OPERATORS_IOBARRIER_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * An IOBarrier operation is used to sequentialize other operations after other IO state operations.
 * It has no equivalent in LLVM.
 *
 * Example:
 *
 * \code{.c}
 * int f(int x)
 * {
 *   opaque(); //calls internally exit(0)
 *   return x / 0;
 * }
 * \endcode
 *
 * The above code is valid C code and not undefined even though there is a division by zero present.
 * The reason for this is that the function opaque() invokes exit(0), and the division by zero is
 * never performed at runtime. In the RVSDG, the division operation has no dependency on the
 * function call to opaque() and therefore it can happen that it is sequentialized before the call
 * operation, transforming the valid program to an undefined program.
 *
 * The IOBarrier operation ensures a sequentialization of these two operations by routing one of the
 * division operands through it along with an I/O state as additional operand. The division
 * operation consumes then the result value of the IOBarrier operation, effectively seuqentializing
 * the division after the barrier and with that after the call operation:
 *
 * ... io = Call opaque ....
 * xo = IOBarrier x io
 * ... = ISDiv xo 0
 */
class IOBarrierOperation final : public rvsdg::SimpleOperation
{
public:
  ~IOBarrierOperation() noexcept override;

  explicit IOBarrierOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : SimpleOperation({ type, IOStateType::Create() }, { type })
  {}

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  Type() const noexcept
  {
    return result(0);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] static rvsdg::Input &
  BarredInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(rvsdg::is<IOBarrierOperation>(&node));
    const auto input = node.input(0);
    return *input;
  }

  [[nodiscard]] static rvsdg::Input &
  getIOStateInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(rvsdg::is<IOBarrierOperation>(&node));
    const auto input = node.input(1);
    JLM_ASSERT(rvsdg::is<IOStateType>(input->Type()));
    return *input;
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & value, rvsdg::Output & ioState)
  {
    return rvsdg::CreateOpNode<IOBarrierOperation>({ &value, &ioState }, value.Type());
  }
};

/**
 * A \ref MemoryHoistBarrierOperation is used to sequentialize memory operations, such as
 * \ref LoadNonVolatileOperation or \ref StoreNonVolatileOperation, after other IO state operations.
 * It has no equivalent in LLVM.
 *
 * Example:
 *
 * \code{.c}
 * int f(int * x)
 * {
 *   opaque(); //calls internally exit(0)
 *   return *x;
 * }
 * \endcode
 *
 * The above code is valid C code and not undefined even if x is null.
 * The reason for this is that the function opaque() invokes exit(0), and the load operation is
 * never performed at runtime. In the RVSDG, the load operation might have no dependency on the
 * function call to opaque() and therefore it can happen that it is sequentialized before the call
 * operation, transforming the valid program to an undefined program.
 *
 * The \ref MemoryHoistBarrierOperation ensures a sequentialization of these two operations by
 * routing the address operand through it along with an I/O state as additional operand. The
 * load operation consumes then the result value of the \ref MemoryHoistBarrierOperation,
 * effectively sequentializing the load after the barrier and with that after the call operation:
 *
 * ... io = Call opaque ....
 * ptr2 = MemoryHoistBarrierOperation ptr io
 * ... = LoadNonVolatileOperation ptr2 ...
 *
 * The \ref MemoryHoistBarrierOperation has a \ref MemoryHoistBarrierOperation::dereferenceableSize_
 * attribute, which determines the number of bytes that its input address is known to be
 * dereferenceable.
 */
class MemoryHoistBarrierOperation final : public rvsdg::SimpleOperation
{
public:
  ~MemoryHoistBarrierOperation() noexcept override;

  explicit MemoryHoistBarrierOperation(const std::size_t dereferenceableSize)
      : SimpleOperation(
            { PointerType::Create(), IOStateType::Create() },
            { PointerType::Create() }),
        dereferenceableSize_(dereferenceableSize)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::size_t
  getDereferenceableSize() const noexcept
  {
    return dereferenceableSize_;
  }

  [[nodiscard]] static rvsdg::Input &
  getAddressInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(rvsdg::is<MemoryHoistBarrierOperation>(&node));
    const auto input = node.input(0);
    JLM_ASSERT(rvsdg::is<PointerType>(input->Type()));
    return *input;
  }

  [[nodiscard]] static rvsdg::Output &
  getAddressOutput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(rvsdg::is<MemoryHoistBarrierOperation>(&node));
    const auto output = node.output(0);
    JLM_ASSERT(rvsdg::is<PointerType>(output->Type()));
    return *output;
  }

  [[nodiscard]] static rvsdg::Input &
  getIOStateInput(const rvsdg::Node & node) noexcept
  {
    JLM_ASSERT(rvsdg::is<MemoryHoistBarrierOperation>(&node));
    const auto input = node.input(1);
    JLM_ASSERT(rvsdg::is<IOStateType>(input->Type()));
    return *input;
  }

  static rvsdg::SimpleNode &
  createNode(
      rvsdg::Output & address,
      rvsdg::Output & ioState,
      const std::size_t dereferenceableSize)
  {
    return rvsdg::CreateOpNode<MemoryHoistBarrierOperation>(
        { &address, &ioState },
        dereferenceableSize);
  }

private:
  /**
   * Dereferenceable size of the memory input in bytes.
   */
  std::size_t dereferenceableSize_;
};

}

#endif // JLM_LLVM_IR_OPERATORS_IOBARRIER_HPP
