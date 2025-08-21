/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::llvm
{

DeltaOperation::~DeltaOperation() noexcept = default;

std::string
DeltaOperation::debug_string() const
{
  return util::strfmt("DELTA[", name(), "]");
}

std::unique_ptr<rvsdg::Operation>
DeltaOperation::copy() const
{
  return std::make_unique<DeltaOperation>(*this);
}

bool
DeltaOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const DeltaOperation *>(&other);
  return op && op->name_ == name_ && op->linkage_ == linkage_ && op->constant() == constant()
      && op->Section_ == Section_ && *op->Type() == *Type();
}

}
