/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_IOBARRIER_HPP
#define JLM_LLVM_IR_OPERATORS_IOBARRIER_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/operation.hpp>

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
};

}

#endif // JLM_LLVM_IR_OPERATORS_IOBARRIER_HPP
