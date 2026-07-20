/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::llvm
{

LlvmDeltaOperation::~LlvmDeltaOperation() noexcept = default;

std::string
LlvmDeltaOperation::debug_string() const
{
  return util::strfmt("DELTA[", name(), "]");
}

std::unique_ptr<rvsdg::Operation>
LlvmDeltaOperation::copy() const
{
  return std::make_unique<LlvmDeltaOperation>(*this);
}

bool
LlvmDeltaOperation::operator==(const Operation & other) const noexcept
{
  const auto op = dynamic_cast<const LlvmDeltaOperation *>(&other);
  return op && op->name_ == name_ && op->linkage_ == linkage_ && op->constant() == constant()
      && op->Section_ == Section_ && op->alignment_ == alignment_ && *op->Type() == *Type();
}

}
