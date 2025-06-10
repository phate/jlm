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

/* alloca operator */

class alloca_op final : public rvsdg::SimpleOperation
{
public:
  virtual ~alloca_op() noexcept;

  inline alloca_op(
      std::shared_ptr<const rvsdg::ValueType> allocatedType,
      std::shared_ptr<const rvsdg::bittype> btype,
      size_t alignment)
      : SimpleOperation({ btype }, { { PointerType::Create() }, { MemoryStateType::Create() } }),
        alignment_(alignment),
        AllocatedType_(std::move(allocatedType))
  {}

  alloca_op(const alloca_op & other) = default;

  alloca_op(alloca_op && other) noexcept = default;

  virtual bool
  operator==(const Operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline const rvsdg::bittype &
  size_type() const noexcept
  {
    return *std::static_pointer_cast<const rvsdg::bittype>(argument(0));
  }

  [[nodiscard]] const rvsdg::ValueType &
  value_type() const noexcept
  {
    return *AllocatedType_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  ValueType() const noexcept
  {
    return AllocatedType_;
  }

  inline size_t
  alignment() const noexcept
  {
    return alignment_;
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(
      std::shared_ptr<const rvsdg::ValueType> allocatedType,
      const variable * size,
      size_t alignment)
  {
    auto bt = std::dynamic_pointer_cast<const rvsdg::bittype>(size->Type());
    if (!bt)
      throw jlm::util::error("expected bits type.");

    alloca_op op(std::move(allocatedType), std::move(bt), alignment);
    return ThreeAddressCode::create(op, { size });
  }

  static std::vector<rvsdg::Output *>
  create(
      std::shared_ptr<const rvsdg::ValueType> allocatedType,
      rvsdg::Output * size,
      size_t alignment)
  {
    auto bt = std::dynamic_pointer_cast<const rvsdg::bittype>(size->Type());
    if (!bt)
      throw util::error("expected bits type.");

    return outputs(&rvsdg::CreateOpNode<alloca_op>(
        { size },
        std::move(allocatedType),
        std::move(bt),
        alignment));
  }

private:
  size_t alignment_;
  std::shared_ptr<const rvsdg::ValueType> AllocatedType_;
};

}

#endif
