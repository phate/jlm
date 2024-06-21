/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_ALLOCA_HPP
#define JLM_LLVM_IR_OPERATORS_ALLOCA_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/simple-normal-form.hpp>

namespace jlm::llvm
{

/* alloca operator */

class alloca_op final : public rvsdg::simple_op
{
public:
  virtual ~alloca_op() noexcept;

  inline alloca_op(
      std::shared_ptr<const rvsdg::valuetype> allocatedType,
      std::shared_ptr<const rvsdg::bittype> btype,
      size_t alignment)
      : simple_op({ btype }, { { PointerType::Create() }, { MemoryStateType::Create() } }),
        alignment_(alignment),
        AllocatedType_(std::move(allocatedType))
  {}

  alloca_op(const alloca_op & other) = default;

  alloca_op(alloca_op && other) noexcept = default;

  virtual bool
  operator==(const operation & other) const noexcept override;

  virtual std::string
  debug_string() const override;

  virtual std::unique_ptr<rvsdg::operation>
  copy() const override;

  inline const rvsdg::bittype &
  size_type() const noexcept
  {
    return *static_cast<const rvsdg::bittype *>(&argument(0).type());
  }

  inline const rvsdg::valuetype &
  value_type() const noexcept
  {
    return *AllocatedType_;
  }

  inline const std::shared_ptr<const rvsdg::valuetype> &
  ValueType() const noexcept
  {
    return AllocatedType_;
  }

  inline size_t
  alignment() const noexcept
  {
    return alignment_;
  }

  static std::unique_ptr<llvm::tac>
  create(
      std::shared_ptr<const rvsdg::valuetype> allocatedType,
      const variable * size,
      size_t alignment)
  {
    auto bt = std::dynamic_pointer_cast<const rvsdg::bittype>(size->Type());
    if (!bt)
      throw jlm::util::error("expected bits type.");

    alloca_op op(std::move(allocatedType), std::move(bt), alignment);
    return tac::create(op, { size });
  }

  static std::vector<rvsdg::output *>
  create(
      std::shared_ptr<const rvsdg::valuetype> allocatedType,
      rvsdg::output * size,
      size_t alignment)
  {
    auto bt = std::dynamic_pointer_cast<const rvsdg::bittype>(size->Type());
    if (!bt)
      throw jlm::util::error("expected bits type.");

    alloca_op op(std::move(allocatedType), std::move(bt), alignment);
    return rvsdg::simple_node::create_normalized(size->region(), op, { size });
  }

private:
  size_t alignment_;
  std::shared_ptr<const rvsdg::valuetype> AllocatedType_;
};

}

#endif
