/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_LAMBDA_HPP
#define JLM_LLVM_IR_OPERATORS_LAMBDA_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/linkage.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/iterator_range.hpp>

#include <optional>
#include <utility>

namespace jlm::llvm
{

/** \brief Lambda operation
 *
 * A lambda operation determines a lambda's name and \ref rvsdg::FunctionType "function type".
 */
class LlvmLambdaOperation final : public rvsdg::LambdaOperation
{
public:
  ~LlvmLambdaOperation() override;

  LlvmLambdaOperation(
      std::shared_ptr<const jlm::rvsdg::FunctionType> type,
      std::string name,
      const jlm::llvm::linkage & linkage,
      jlm::llvm::attributeset attributes);

  [[nodiscard]] const std::string &
  name() const noexcept
  {
    return name_;
  }

  [[nodiscard]] const jlm::llvm::linkage &
  linkage() const noexcept
  {
    return linkage_;
  }

  [[nodiscard]] const jlm::llvm::attributeset &
  attributes() const noexcept
  {
    return attributes_;
  }

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] const jlm::llvm::attributeset &
  GetArgumentAttributes(std::size_t index) const noexcept;

  void
  SetArgumentAttributes(std::size_t index, const jlm::llvm::attributeset & attributes);

  static std::unique_ptr<LlvmLambdaOperation>
  Create(
      std::shared_ptr<const jlm::rvsdg::FunctionType> type,
      std::string name,
      const jlm::llvm::linkage & linkage,
      jlm::llvm::attributeset attributes)
  {
    return std::make_unique<LlvmLambdaOperation>(
        std::move(type),
        std::move(name),
        linkage,
        std::move(attributes));
  }

  static std::unique_ptr<LlvmLambdaOperation>
  Create(
      std::shared_ptr<const jlm::rvsdg::FunctionType> type,
      std::string name,
      const jlm::llvm::linkage & linkage)
  {
    return std::make_unique<LlvmLambdaOperation>(
        std::move(type),
        std::move(name),
        linkage,
        jlm::llvm::attributeset{});
  }

private:
  std::string name_;
  jlm::llvm::linkage linkage_;
  jlm::llvm::attributeset attributes_;
  std::vector<jlm::llvm::attributeset> ArgumentAttributes_;
};

}

#endif
