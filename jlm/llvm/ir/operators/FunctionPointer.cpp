/*
 * Copyright 2024 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/FunctionPointer.hpp>

namespace jlm::llvm
{

FunctionToPointerOperation::~FunctionToPointerOperation() noexcept
{}

FunctionToPointerOperation::FunctionToPointerOperation(
    std::shared_ptr<const llvm::FunctionType> fn,
    std::shared_ptr<const llvm::PointerType> ptr)
    : SimpleOperation({ fn }, { ptr }),
      FunctionType_(std::move(fn)),
      PointerType_(std::move(ptr))
{}

bool
FunctionToPointerOperation::operator==(const Operation & other) const noexcept
{
  if (auto o = dynamic_cast<const FunctionToPointerOperation *>(&other))
  {
    return *FunctionType() == *o->FunctionType() && *PointerType() == *o->PointerType();
  }
  else
  {
    return false;
  }
}

[[nodiscard]] std::string
FunctionToPointerOperation::debug_string() const
{
  return "FunPtr(" + FunctionType()->debug_string() + "," + PointerType()->debug_string() + ")";
}

[[nodiscard]] std::unique_ptr<rvsdg::Operation>
FunctionToPointerOperation::copy() const
{
  return Create(FunctionType(), PointerType());
}

std::unique_ptr<FunctionToPointerOperation>
FunctionToPointerOperation::Create(
    std::shared_ptr<const llvm::FunctionType> fn,
    std::shared_ptr<const llvm::PointerType> ptr)
{
  return std::make_unique<FunctionToPointerOperation>(std::move(fn), std::move(ptr));
}

PointerToFunctionOperation::~PointerToFunctionOperation() noexcept
{}

PointerToFunctionOperation::PointerToFunctionOperation(
    std::shared_ptr<const llvm::PointerType> ptr,
    std::shared_ptr<const llvm::FunctionType> fn)
    : SimpleOperation({ ptr }, { fn }),
      PointerType_(std::move(ptr)),
      FunctionType_(std::move(fn))
{}

bool
PointerToFunctionOperation::operator==(const Operation & other) const noexcept
{
  if (auto o = dynamic_cast<const PointerToFunctionOperation *>(&other))
  {
    return *PointerType() == *o->PointerType() && *FunctionType() == *o->FunctionType();
  }
  else
  {
    return false;
  }
}

[[nodiscard]] std::string
PointerToFunctionOperation::debug_string() const
{
  return "PtrFun(" + PointerType()->debug_string() + "," + FunctionType()->debug_string() + ")";
}

[[nodiscard]] std::unique_ptr<rvsdg::Operation>
PointerToFunctionOperation::copy() const
{
  return Create(PointerType(), FunctionType());
}

std::unique_ptr<PointerToFunctionOperation>
PointerToFunctionOperation::Create(
    std::shared_ptr<const llvm::PointerType> ptr,
    std::shared_ptr<const llvm::FunctionType> fn)
{
  return std::make_unique<PointerToFunctionOperation>(std::move(ptr), std::move(fn));
}

}
