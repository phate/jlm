/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_CONVERSIONOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_CONVERSIONOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::llvm
{

class BitCastOperation final : public rvsdg::UnaryOperation
{
public:
  ~BitCastOperation() noexcept override;

  BitCastOperation(
      std::shared_ptr<const jlm::rvsdg::Type> srctype,
      std::shared_ptr<const jlm::rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    check_types(srctype, dsttype);
  }

  BitCastOperation(const BitCastOperation &) = default;

  explicit BitCastOperation(Operation &&) = delete;

  BitCastOperation &
  operator=(const Operation &) = delete;

  BitCastOperation &
  operator=(Operation &&) = delete;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * output) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * output)
      const override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  createTac(const Variable * operand, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto pair = check_types(operand->Type(), type);

    auto op = std::make_unique<BitCastOperation>(pair.first, pair.second);
    return ThreeAddressCode::create(std::move(op), { operand });
  }

  static jlm::rvsdg::Output *
  create(jlm::rvsdg::Output * operand, std::shared_ptr<const jlm::rvsdg::Type> rtype)
  {
    auto pair = check_types(operand->Type(), rtype);
    return rvsdg::CreateOpNode<BitCastOperation>({ operand }, pair.first, pair.second).output(0);
  }

private:
  static std::pair<std::shared_ptr<const jlm::rvsdg::Type>, std::shared_ptr<const jlm::rvsdg::Type>>
  check_types(
      const std::shared_ptr<const jlm::rvsdg::Type> & otype,
      const std::shared_ptr<const jlm::rvsdg::Type> & rtype)
  {
    if (otype->Kind() != rvsdg::TypeKind::Value)
      throw util::Error("expected value type.");

    if (rtype->Kind() != rvsdg::TypeKind::Value)
      throw util::Error("expected value type.");

    return std::make_pair(otype, rtype);
  }
};

class SExtOperation final : public rvsdg::UnaryOperation
{
public:
  ~SExtOperation() noexcept override;

  SExtOperation(
      std::shared_ptr<const rvsdg::Type> srctype,
      std::shared_ptr<const rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    auto ot = std::dynamic_pointer_cast<const rvsdg::BitType>(srctype);
    if (!ot)
      throw util::Error("expected bits type.");

    auto rt = std::dynamic_pointer_cast<const rvsdg::BitType>(dsttype);
    if (!rt)
      throw util::Error("expected bits type.");

    if (ot->nbits() >= rt->nbits())
      throw util::Error("expected operand's #bits to be smaller than results' #bits.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::unop_reduction_path_t
  can_reduce_operand(const rvsdg::Output * operand) const noexcept override;

  rvsdg::Output *
  reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::Output * operand) const override;

  inline size_t
  nsrcbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(argument(0))->nbits();
  }

  inline size_t
  ndstbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(result(0))->nbits();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  createTac(const Variable * operand, const std::shared_ptr<const rvsdg::Type> & type)
  {
    auto operation = std::make_unique<SExtOperation>(operand->Type(), std::move(type));
    return ThreeAddressCode::create(std::move(operation), { operand });
  }

  static rvsdg::Output &
  create(size_t ndstbits, rvsdg::Output & operand)
  {
    return *rvsdg::CreateOpNode<SExtOperation>(
                { &operand },
                operand.Type(),
                rvsdg::BitType::Create(ndstbits))
                .output(0);
  }
};

class ZExtOperation final : public rvsdg::UnaryOperation
{
public:
  ~ZExtOperation() noexcept override;

  ZExtOperation(
      std::shared_ptr<const jlm::rvsdg::Type> srctype,
      std::shared_ptr<const jlm::rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    auto st = dynamic_cast<const jlm::rvsdg::BitType *>(srctype.get());
    if (!st)
      throw util::Error("expected bitstring type.");

    auto dt = dynamic_cast<const jlm::rvsdg::BitType *>(dsttype.get());
    if (!dt)
      throw util::Error("expected bitstring type.");

    if (dt->nbits() < st->nbits())
      throw util::Error("# destination bits must be greater than # source bits.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * operand) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * operand)
      const override;

  inline size_t
  nsrcbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(argument(0))->nbits();
  }

  inline size_t
  ndstbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(result(0))->nbits();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  createTac(const Variable * operand, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto operation = std::make_unique<ZExtOperation>(operand->Type(), std::move(type));
    return ThreeAddressCode::create(std::move(operation), { operand });
  }

  static rvsdg::Output &
  create(size_t ndstbits, rvsdg::Output & operand)
  {
    return *rvsdg::CreateOpNode<ZExtOperation>(
                { &operand },
                operand.Type(),
                rvsdg::BitType::Create(ndstbits))
                .output(0);
  }
};

class TruncOperation final : public rvsdg::UnaryOperation
{
public:
  ~TruncOperation() noexcept override;

  TruncOperation(
      const std::shared_ptr<const rvsdg::BitType> & operandType,
      const std::shared_ptr<const rvsdg::BitType> & resultType)
      : UnaryOperation(operandType, resultType)
  {
    if (operandType->nbits() < resultType->nbits())
      throw util::Error("expected operand's #bits to be larger than results' #bits.");
  }

  TruncOperation(
      std::shared_ptr<const rvsdg::Type> operandType,
      std::shared_ptr<const rvsdg::Type> resultType)
      : TruncOperation(checkBitType(std::move(operandType)), checkBitType(std::move(resultType)))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  jlm::rvsdg::unop_reduction_path_t
  can_reduce_operand(const jlm::rvsdg::Output * operand) const noexcept override;

  jlm::rvsdg::Output *
  reduce_operand(jlm::rvsdg::unop_reduction_path_t path, jlm::rvsdg::Output * operand)
      const override;

  inline size_t
  nsrcbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(argument(0))->nbits();
  }

  inline size_t
  ndstbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(result(0))->nbits();
  }

  static std::unique_ptr<ThreeAddressCode>
  createTac(const Variable * operand, const std::shared_ptr<const rvsdg::Type> & resultType)
  {
    auto truncOperation = std::make_unique<TruncOperation>(operand->Type(), resultType);
    return ThreeAddressCode::create(std::move(truncOperation), { operand });
  }

  static rvsdg::Node &
  createNode(rvsdg::Output & operand, std::shared_ptr<const rvsdg::Type> resultType)
  {
    return rvsdg::CreateOpNode<TruncOperation>({ &operand }, operand.Type(), std::move(resultType));
  }

  static rvsdg::Output &
  create(size_t ndstbits, rvsdg::Output & operand)
  {
    return *createNode(operand, rvsdg::BitType::Create(ndstbits)).output(0);
  }

private:
  static std::shared_ptr<const rvsdg::BitType>
  checkBitType(const std::shared_ptr<const rvsdg::Type> & type)
  {
    if (auto bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(type))
    {
      return bitType;
    }

    throw std::logic_error("expected bits type.");
  }
};

}

#endif
