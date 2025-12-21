/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_ALLOCA_HPP
#define JLM_LLVM_IR_OPERATORS_ALLOCA_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

class AllocaOperation final : public rvsdg::SimpleOperation
{
public:
  ~AllocaOperation() noexcept override;

  AllocaOperation(
      std::shared_ptr<const rvsdg::Type> allocatedType,
      std::shared_ptr<const rvsdg::BitType> btype,
      size_t alignment)
      : SimpleOperation({ btype }, { { PointerType::Create() }, { MemoryStateType::Create() } }),
        alignment_(alignment),
        AllocatedType_(std::move(allocatedType))
  {}

  AllocaOperation(const AllocaOperation & other) = default;

  AllocaOperation(AllocaOperation && other) noexcept = default;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline const rvsdg::BitType &
  size_type() const noexcept
  {
    return *std::static_pointer_cast<const rvsdg::BitType>(argument(0));
  }

  [[nodiscard]] const rvsdg::Type &
  value_type() const noexcept
  {
    return *AllocatedType_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  ValueType() const noexcept
  {
    return AllocatedType_;
  }

  inline size_t
  alignment() const noexcept
  {
    return alignment_;
  }

  static rvsdg::Input &
  getCountInput(rvsdg::Node & node)
  {
    JLM_ASSERT(is<AllocaOperation>(&node));
    return *node.input(0);
  }

  static rvsdg::Output &
  getPointerOutput(rvsdg::Node & node)
  {
    JLM_ASSERT(is<AllocaOperation>(&node));
    return *node.output(0);
  }

  static rvsdg::Output &
  getMemoryStateOutput(rvsdg::Node & node)
  {
    JLM_ASSERT(is<AllocaOperation>(&node));
    return *node.output(1);
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(std::shared_ptr<const rvsdg::Type> allocatedType, const Variable * size, size_t alignment)
  {
    auto bt = std::dynamic_pointer_cast<const rvsdg::BitType>(size->Type());
    if (!bt)
      throw util::Error("expected bits type.");

    AllocaOperation op(std::move(allocatedType), std::move(bt), alignment);
    return ThreeAddressCode::create(op, { size });
  }

  /**
   * Creates a SimpleNode containing an AllocaOperation.
   * @param allocatedType the type being allocated
   * @param size the number of elements of the given type to allocate. Should almost always be 1.
   * @param alignment the minimum alignment of the allocation
   * @return the outputs of the created SimpleNode
   */
  static std::vector<rvsdg::Output *>
  create(std::shared_ptr<const rvsdg::Type> allocatedType, rvsdg::Output * size, size_t alignment)
  {
    auto bt = std::dynamic_pointer_cast<const rvsdg::BitType>(size->Type());
    if (!bt)
      throw util::Error("expected bits type.");

    return outputs(&rvsdg::CreateOpNode<AllocaOperation>(
        { size },
        std::move(allocatedType),
        std::move(bt),
        alignment));
  }

private:
  size_t alignment_;
  std::shared_ptr<const rvsdg::Type> AllocatedType_;
};

}

#endif
