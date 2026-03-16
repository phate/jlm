/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_DELTA_HPP
#define JLM_LLVM_IR_OPERATORS_DELTA_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/util/iterator_range.hpp>

namespace jlm::llvm
{

/** \brief Delta operation
 */
class DeltaOperation final : public rvsdg::DeltaOperation
{
public:
  ~DeltaOperation() noexcept override;

  DeltaOperation(
      const std::shared_ptr<const rvsdg::Type> & type,
      const std::string & name,
      const Linkage & linkage,
      std::string section,
      const bool constant,
      const size_t alignment)
      : rvsdg::DeltaOperation(type, constant, PointerType::Create()),
        name_(name),
        Section_(std::move(section)),
        linkage_(linkage),
        alignment_(alignment)
  {}

  DeltaOperation(const DeltaOperation & other) = default;

  DeltaOperation(DeltaOperation && other) noexcept = default;

  DeltaOperation &
  operator=(const DeltaOperation &) = delete;

  DeltaOperation &
  operator=(DeltaOperation &&) = delete;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] bool
  operator==(const Operation & other) const noexcept override;

  const std::string &
  name() const noexcept
  {
    return name_;
  }

  [[nodiscard]] const std::string &
  Section() const noexcept
  {
    return Section_;
  }

  const llvm::Linkage &
  linkage() const noexcept
  {
    return linkage_;
  }

  [[nodiscard]] size_t
  getAlignment() const noexcept
  {
    return alignment_;
  }

  static std::unique_ptr<DeltaOperation>
  Create(
      std::shared_ptr<const rvsdg::Type> type,
      const std::string & name,
      const Linkage & linkage,
      std::string section,
      bool constant,
      const size_t alignment)
  {
    return std::make_unique<DeltaOperation>(
        std::move(type),
        name,
        linkage,
        std::move(section),
        constant,
        alignment);
  }

private:
  std::string name_;
  std::string Section_;
  Linkage linkage_;
  size_t alignment_;
};

}

#endif
