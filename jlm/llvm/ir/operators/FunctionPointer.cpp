/*
 * Copyright 2024 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

FunctionToPointerOperation::~FunctionToPointerOperation() noexcept
{}

FunctionToPointerOperation::FunctionToPointerOperation(
    std::shared_ptr<const rvsdg::FunctionType> fn)
    : UnaryOperation(fn, PointerType::Create()),
      FunctionType_(std::move(fn))
{}

bool
FunctionToPointerOperation::operator==(const Operation & other) const noexcept
{
  if (auto o = dynamic_cast<const FunctionToPointerOperation *>(&other))
  {
    return *FunctionType() == *o->FunctionType();
  }
  else
  {
    return false;
  }
}

[[nodiscard]] std::string
FunctionToPointerOperation::debug_string() const
{
  return "FunPtr(" + FunctionType()->debug_string() + ")";
}

[[nodiscard]] std::unique_ptr<rvsdg::Operation>
FunctionToPointerOperation::copy() const
{
  return Create(FunctionType());
}

rvsdg::unop_reduction_path_t
FunctionToPointerOperation::can_reduce_operand(const jlm::rvsdg::output * arg) const noexcept
{
  if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*arg))
  {
    if (auto op = dynamic_cast<const PointerToFunctionOperation *>(&node->GetOperation()))
    {
      if (*op->FunctionType() == *FunctionType())
      {
        return rvsdg::unop_reduction_inverse;
      }
    }
  }
  return rvsdg::unop_reduction_none;
}

jlm::rvsdg::output *
FunctionToPointerOperation::reduce_operand(
    rvsdg::unop_reduction_path_t path,
    jlm::rvsdg::output * arg) const
{
  if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*arg))
  {
    if (auto op = dynamic_cast<const PointerToFunctionOperation *>(&node->GetOperation()))
    {
      if (*op->FunctionType() == *FunctionType() && path == rvsdg::unop_reduction_inverse)
      {
        return node->input(0)->origin();
      }
    }
  }
  return arg;
}

std::unique_ptr<FunctionToPointerOperation>
FunctionToPointerOperation::Create(std::shared_ptr<const rvsdg::FunctionType> fn)
{
  return std::make_unique<FunctionToPointerOperation>(std::move(fn));
}

PointerToFunctionOperation::~PointerToFunctionOperation() noexcept
{}

PointerToFunctionOperation::PointerToFunctionOperation(
    std::shared_ptr<const rvsdg::FunctionType> fn)
    : UnaryOperation(PointerType::Create(), fn),
      FunctionType_(std::move(fn))
{}

bool
PointerToFunctionOperation::operator==(const Operation & other) const noexcept
{
  if (auto o = dynamic_cast<const PointerToFunctionOperation *>(&other))
  {
    return *FunctionType() == *o->FunctionType();
  }
  else
  {
    return false;
  }
}

[[nodiscard]] std::string
PointerToFunctionOperation::debug_string() const
{
  return "PtrFun(" + FunctionType()->debug_string() + ")";
}

[[nodiscard]] std::unique_ptr<rvsdg::Operation>
PointerToFunctionOperation::copy() const
{
  return Create(FunctionType());
}

rvsdg::unop_reduction_path_t
PointerToFunctionOperation::can_reduce_operand(const jlm::rvsdg::output * arg) const noexcept
{
  if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*arg))
  {
    if (auto op = dynamic_cast<const FunctionToPointerOperation *>(&node->GetOperation()))
    {
      if (*op->FunctionType() == *FunctionType())
      {
        return rvsdg::unop_reduction_inverse;
      }
    }
  }
  return rvsdg::unop_reduction_none;
}

jlm::rvsdg::output *
PointerToFunctionOperation::reduce_operand(
    rvsdg::unop_reduction_path_t path,
    jlm::rvsdg::output * arg) const
{
  if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*arg))
  {
    if (auto op = dynamic_cast<const FunctionToPointerOperation *>(&node->GetOperation()))
    {
      if (*op->FunctionType() == *FunctionType() && path == rvsdg::unop_reduction_inverse)
      {
        return node->input(0)->origin();
      }
    }
  }
  return arg;
}

std::unique_ptr<PointerToFunctionOperation>
PointerToFunctionOperation::Create(std::shared_ptr<const rvsdg::FunctionType> fn)
{
  return std::make_unique<PointerToFunctionOperation>(std::move(fn));
}

}
