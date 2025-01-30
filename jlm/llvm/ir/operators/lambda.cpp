/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm
{

LlvmLambdaOperation::~LlvmLambdaOperation() = default;

LlvmLambdaOperation::LlvmLambdaOperation(
    std::shared_ptr<const jlm::rvsdg::FunctionType> type,
    std::string name,
    const jlm::llvm::linkage & linkage,
    jlm::llvm::attributeset attributes)
    : rvsdg::LambdaOperation(std::move(type)),
      name_(std::move(name)),
      linkage_(linkage),
      attributes_(std::move(attributes))
{
  ArgumentAttributes_.resize(Type()->NumArguments());
}

std::string
LlvmLambdaOperation::debug_string() const
{
  return util::strfmt("LAMBDA[", name(), "]");
}

bool
LlvmLambdaOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const LlvmLambdaOperation *>(&other);
  return op && op->type() == type() && op->name() == name() && op->linkage() == linkage()
      && op->attributes() == attributes();
}

std::unique_ptr<rvsdg::Operation>
LlvmLambdaOperation::copy() const
{
  return std::make_unique<LlvmLambdaOperation>(*this);
}

[[nodiscard]] const jlm::llvm::attributeset &
LlvmLambdaOperation::GetArgumentAttributes(std::size_t index) const noexcept
{
  JLM_ASSERT(index < ArgumentAttributes_.size());
  return ArgumentAttributes_[index];
}

void
LlvmLambdaOperation::SetArgumentAttributes(
    std::size_t index,
    const jlm::llvm::attributeset & attributes)
{
  JLM_ASSERT(index < ArgumentAttributes_.size());
  ArgumentAttributes_[index] = attributes;
}

}
