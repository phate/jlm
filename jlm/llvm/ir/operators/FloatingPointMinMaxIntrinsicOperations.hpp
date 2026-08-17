/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_FLOATINGPOINTMINMAXINTRINSICOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_FLOATINGPOINTMINMAXINTRINSICOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Represents LLVM's llvm.floor.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-floor-intrinsic) for more details.
 */
class FloorOperation final : public rvsdg::UnaryOperation
{
public:
  ~FloorOperation() noexcept override;

  explicit FloorOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : UnaryOperation(type, type)
  {
    checkType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand)
  {
    auto operation = std::make_unique<FloorOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<FloorOperation>({ &operand }, operand.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);
};

/**
 * Represents LLVM's llvm.ceil.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-ceil-intrinsic) for more details.
 */
class CeilOperation final : public rvsdg::UnaryOperation
{
public:
  ~CeilOperation() noexcept override;

  explicit CeilOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : UnaryOperation(type, type)
  {
    checkType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand)
  {
    auto operation = std::make_unique<CeilOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<CeilOperation>({ &operand }, operand.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);
};

/**
 * Represents LLVM's llvm.round.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-round-intrinsic) for more details.
 */
class RoundOperation final : public rvsdg::UnaryOperation
{
public:
  ~RoundOperation() noexcept override;

  explicit RoundOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : UnaryOperation(type, type)
  {
    checkType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand)
  {
    auto operation = std::make_unique<RoundOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<RoundOperation>({ &operand }, operand.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);
};

/**
 * Represents LLVM's llvm.trunc.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-trunc-intrinsic) for more details.
 */
class TruncIntrinsicOperation final : public rvsdg::UnaryOperation
{
public:
  ~TruncIntrinsicOperation() noexcept override;

  explicit TruncIntrinsicOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : UnaryOperation(type, type)
  {
    checkType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand)
  {
    auto operation = std::make_unique<TruncIntrinsicOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<TruncIntrinsicOperation>({ &operand }, operand.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);
};

/**
 * Represents LLVM's llvm.rint.* intrinsic
 *
 * See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#llvm-rint-intrinsic) for more details.
 */
class RIntOperation final : public rvsdg::UnaryOperation
{
public:
  ~RIntOperation() noexcept override;

  explicit RIntOperation(const std::shared_ptr<const rvsdg::Type> & type)
      : UnaryOperation(type, type)
  {
    checkType(type);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  [[nodiscard]] std::shared_ptr<const rvsdg::Type>
  getType() const noexcept
  {
    return result(0);
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable & operand)
  {
    auto operation = std::make_unique<RIntOperation>(operand.Type());
    return ThreeAddressCode::create(std::move(operation), { &operand });
  }

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<RIntOperation>({ &operand }, operand.Type());
  }

private:
  static void
  checkType(const std::shared_ptr<const rvsdg::Type> & type);
};

}

#endif
