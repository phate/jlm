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
      std::shared_ptr<const rvsdg::BitType> countType,
      const size_t alignment)
      : SimpleOperation(
            { countType },
            { { PointerType::Create() }, { MemoryStateType::Create() } }),
        alignment_(alignment),
        allocatedType_(std::move(allocatedType))
  {}

  AllocaOperation(const AllocaOperation & other) = default;

  AllocaOperation(AllocaOperation && other) noexcept = default;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  const rvsdg::BitType &
  countType() const noexcept
  {
    const auto type = argument(0);
    JLM_ASSERT(is<rvsdg::BitType>(type));
    return *std::static_pointer_cast<const rvsdg::BitType>(type);
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  allocatedType() const noexcept
  {
    return allocatedType_;
  }

  size_t
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

  static std::unique_ptr<ThreeAddressCode>
  createTac(
      std::shared_ptr<const rvsdg::Type> allocatedType,
      const Variable * count,
      size_t alignment)
  {
    auto bitType = checkOperandType(count->Type());

    auto op =
        std::make_unique<AllocaOperation>(std::move(allocatedType), std::move(bitType), alignment);
    return ThreeAddressCode::create(std::move(op), { count });
  }

  /**
   * Creates a SimpleNode containing an AllocaOperation.
   *
   * @param allocatedType the type being allocated
   * @param count the number of elements of the given type to allocate.
   * @param alignment the minimum alignment of the allocation
   * @return the created SimpleNode
   */
  static rvsdg::SimpleNode &
  createNode(
      std::shared_ptr<const rvsdg::Type> allocatedType,
      rvsdg::Output & count,
      const size_t alignment)
  {
    auto bitType = checkOperandType(count.Type());

    return rvsdg::CreateOpNode<AllocaOperation>(
        { &count },
        std::move(allocatedType),
        std::move(bitType),
        alignment);
  }

  /**
   * Creates a SimpleNode containing an AllocaOperation.
   * @param allocatedType the type being allocated
   * @param count the number of elements of the given type to allocate. Should almost always be 1.
   * @param alignment the minimum alignment of the allocation
   * @return the outputs of the created SimpleNode
   */
  static std::vector<rvsdg::Output *>
  create(
      std::shared_ptr<const rvsdg::Type> allocatedType,
      rvsdg::Output * count,
      const size_t alignment)
  {
    return rvsdg::outputs(&createNode(std::move(allocatedType), *count, alignment));
  }

private:
  static std::shared_ptr<const rvsdg::BitType>
  checkOperandType(const std::shared_ptr<const rvsdg::Type> & countType)
  {
    if (auto bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(countType))
      return bitType;

    throw util::Error("Expected bits type.");
  }

  size_t alignment_;
  std::shared_ptr<const rvsdg::Type> allocatedType_;
};

}

#endif
