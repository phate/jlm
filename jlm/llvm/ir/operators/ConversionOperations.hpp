/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_CONVERSIONOPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_CONVERSIONOPERATIONS_HPP

#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/control.hpp>
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

  static rvsdg::SimpleNode &
  createNode(const size_t numResultBits, rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<SExtOperation>(
        { &operand },
        operand.Type(),
        rvsdg::BitType::Create(numResultBits));
  }

  static rvsdg::Output &
  create(size_t ndstbits, rvsdg::Output & operand)
  {
    return *createNode(ndstbits, operand).output(0);
  }

  /**
   * Performs constant folding by statically evaluating the constant operand and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref SExtOperation on which the transformation is performed.
   * @param operands The operands of the \ref SExtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref SExtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstant(const SExtOperation & operation, const std::vector<rvsdg::Output *> & operands);
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

  static rvsdg::SimpleNode &
  createNode(const size_t numResultBits, rvsdg::Output & operand)
  {
    return rvsdg::CreateOpNode<ZExtOperation>(
        { &operand },
        operand.Type(),
        rvsdg::BitType::Create(numResultBits));
  }

  static rvsdg::Output &
  create(size_t ndstbits, rvsdg::Output & operand)
  {
    return *createNode(ndstbits, operand).output(0);
  }

  /**
   * Performs constant folding by statically evaluating the constant operand and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref ZExtOperation on which the transformation is performed.
   * @param operands The operands of the \ref ZExtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref ZExtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstant(const ZExtOperation & operation, const std::vector<rvsdg::Output *> & operands);
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

  static rvsdg::SimpleNode &
  createNode(rvsdg::Output & operand, std::shared_ptr<const rvsdg::Type> resultType)
  {
    return rvsdg::CreateOpNode<TruncOperation>({ &operand }, operand.Type(), std::move(resultType));
  }

  static rvsdg::Output &
  create(size_t ndstbits, rvsdg::Output & operand)
  {
    return *createNode(operand, rvsdg::BitType::Create(ndstbits)).output(0);
  }

  /**
   * Performs constant folding by statically evaluating the constant operand and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref TruncOperation on which the transformation is performed.
   * @param operands The operands of the \ref TruncOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref TruncOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstant(const TruncOperation & operation, const std::vector<rvsdg::Output *> & operands);

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

class PtrToIntOperation final : public rvsdg::UnaryOperation
{
public:
  ~PtrToIntOperation() noexcept override;

  PtrToIntOperation(
      std::shared_ptr<const PointerType> ptype,
      std::shared_ptr<const jlm::rvsdg::BitType> btype)
      : UnaryOperation(std::move(ptype), std::move(btype))
  {}

  PtrToIntOperation(
      std::shared_ptr<const jlm::rvsdg::Type> srctype,
      std::shared_ptr<const jlm::rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    auto pt = dynamic_cast<const PointerType *>(srctype.get());
    if (!pt)
      throw util::Error("expected pointer type.");

    auto bt = dynamic_cast<const jlm::rvsdg::BitType *>(dsttype.get());
    if (!bt)
      throw util::Error("expected bitstring type.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline size_t
  nbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(result(0))->nbits();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * argument, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto pt = std::dynamic_pointer_cast<const PointerType>(argument->Type());
    if (!pt)
      throw util::Error("expected pointer type.");

    auto bt = std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(type);
    if (!bt)
      throw util::Error("expected bitstring type.");

    auto op = std::make_unique<PtrToIntOperation>(std::move(pt), std::move(bt));
    return ThreeAddressCode::create(std::move(op), { argument });
  }
};

class FPExtOperation final : public rvsdg::UnaryOperation
{
public:
  ~FPExtOperation() noexcept override;

  FPExtOperation(const fpsize & srcsize, const fpsize & dstsize)
      : UnaryOperation(FloatingPointType::Create(srcsize), FloatingPointType::Create(dstsize))
  {
    if (srcsize == fpsize::flt && dstsize == fpsize::half)
      throw util::Error("destination type size must be bigger than source type size.");
  }

  FPExtOperation(
      const std::shared_ptr<const FloatingPointType> & srctype,
      const std::shared_ptr<const FloatingPointType> & dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    if (srctype->size() == fpsize::flt && dsttype->size() == fpsize::half)
      throw util::Error("destination type size must be bigger than source type size.");
  }

  FPExtOperation(
      std::shared_ptr<const jlm::rvsdg::Type> srctype,
      std::shared_ptr<const jlm::rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    auto st = dynamic_cast<const FloatingPointType *>(srctype.get());
    if (!st)
      throw util::Error("expected floating point type.");

    auto dt = dynamic_cast<const FloatingPointType *>(dsttype.get());
    if (!dt)
      throw util::Error("expected floating point type.");

    if (st->size() == fpsize::flt && dt->size() == fpsize::half)
      throw util::Error("destination type size must be bigger than source type size.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline const fpsize &
  srcsize() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(argument(0))->size();
  }

  inline const fpsize &
  dstsize() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(result(0))->size();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * operand, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto st = std::dynamic_pointer_cast<const FloatingPointType>(operand->Type());
    if (!st)
      throw util::Error("expected floating point type.");

    auto dt = std::dynamic_pointer_cast<const FloatingPointType>(type);
    if (!dt)
      throw util::Error("expected floating point type.");

    auto op = std::make_unique<FPExtOperation>(std::move(st), std::move(dt));
    return ThreeAddressCode::create(std::move(op), { operand });
  }
};

class FPTruncOperation final : public rvsdg::UnaryOperation
{
public:
  ~FPTruncOperation() noexcept override;

  FPTruncOperation(const fpsize & srcsize, const fpsize & dstsize)
      : UnaryOperation(FloatingPointType::Create(srcsize), FloatingPointType::Create(dstsize))
  {
    if (srcsize == fpsize::half || (srcsize == fpsize::flt && dstsize != fpsize::half)
        || (srcsize == fpsize::dbl && dstsize == fpsize::dbl))
      throw util::Error("destination tpye size must be smaller than source size type.");
  }

  FPTruncOperation(
      const std::shared_ptr<const FloatingPointType> & srctype,
      const std::shared_ptr<const FloatingPointType> & dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    if (srctype->size() == fpsize::flt && dsttype->size() == fpsize::half)
      throw util::Error("destination type size must be bigger than source type size.");
  }

  FPTruncOperation(
      std::shared_ptr<const jlm::rvsdg::Type> srctype,
      std::shared_ptr<const jlm::rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    auto st = dynamic_cast<const FloatingPointType *>(srctype.get());
    if (!st)
      throw util::Error("expected floating point type.");

    auto dt = dynamic_cast<const FloatingPointType *>(dsttype.get());
    if (!dt)
      throw util::Error("expected floating point type.");

    if (st->size() == fpsize::half || (st->size() == fpsize::flt && dt->size() != fpsize::half)
        || (st->size() == fpsize::dbl && dt->size() == fpsize::dbl))
      throw util::Error("destination type size must be smaller than source size type.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  inline const fpsize &
  srcsize() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(argument(0))->size();
  }

  inline const fpsize &
  dstsize() const noexcept
  {
    return std::static_pointer_cast<const FloatingPointType>(result(0))->size();
  }

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * operand, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    auto st = std::dynamic_pointer_cast<const FloatingPointType>(operand->Type());
    if (!st)
      throw util::Error("expected floating point type.");

    auto dt = std::dynamic_pointer_cast<const FloatingPointType>(type);
    if (!dt)
      throw util::Error("expected floating point type.");

    auto op = std::make_unique<FPTruncOperation>(std::move(st), std::move(dt));
    return ThreeAddressCode::create(std::move(op), { operand });
  }
};

class UIToFPOperation final : public rvsdg::UnaryOperation
{
public:
  ~UIToFPOperation() noexcept override;

  UIToFPOperation(
      std::shared_ptr<const jlm::rvsdg::BitType> srctype,
      std::shared_ptr<const FloatingPointType> dsttype)
      : UnaryOperation(std::move(srctype), std::move(dsttype))
  {}

  UIToFPOperation(
      std::shared_ptr<const jlm::rvsdg::Type> optype,
      std::shared_ptr<const jlm::rvsdg::Type> restype)
      : UnaryOperation(optype, restype)
  {
    auto st = dynamic_cast<const jlm::rvsdg::BitType *>(optype.get());
    if (!st)
      throw util::Error("expected bits type.");

    auto rt = dynamic_cast<const FloatingPointType *>(restype.get());
    if (!rt)
      throw util::Error("expected floating point type.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * operand, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto st = std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(operand->Type());
    if (!st)
      throw util::Error("expected bits type.");

    auto rt = std::dynamic_pointer_cast<const FloatingPointType>(type);
    if (!rt)
      throw util::Error("expected floating point type.");

    auto op = std::make_unique<UIToFPOperation>(std::move(st), std::move(rt));
    return ThreeAddressCode::create(std::move(op), { operand });
  }
};

class SIToFPOperation final : public rvsdg::UnaryOperation
{
public:
  ~SIToFPOperation() noexcept override;

  SIToFPOperation(
      std::shared_ptr<const jlm::rvsdg::BitType> srctype,
      std::shared_ptr<const FloatingPointType> dsttype)
      : UnaryOperation(std::move(srctype), std::move(dsttype))
  {}

  SIToFPOperation(
      std::shared_ptr<const jlm::rvsdg::Type> srctype,
      std::shared_ptr<const jlm::rvsdg::Type> dsttype)
      : UnaryOperation(srctype, dsttype)
  {
    auto st = dynamic_cast<const jlm::rvsdg::BitType *>(srctype.get());
    if (!st)
      throw util::Error("expected bits type.");

    auto rt = dynamic_cast<const FloatingPointType *>(dsttype.get());
    if (!rt)
      throw util::Error("expected floating point type.");
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * operand, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto st = std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(operand->Type());
    if (!st)
      throw util::Error("expected bits type.");

    auto rt = std::dynamic_pointer_cast<const FloatingPointType>(type);
    if (!rt)
      throw util::Error("expected floating point type.");

    auto op = std::make_unique<SIToFPOperation>(std::move(st), std::move(rt));
    return ThreeAddressCode::create(std::move(op), { operand });
  }
};

class IntToPtrOperation final : public rvsdg::UnaryOperation
{
public:
  ~IntToPtrOperation() noexcept override;

  // FIXME: The second parameter is unused, but we need to entangled the users of the constructor
  // first before we can eliminate it.
  explicit IntToPtrOperation(
      std::shared_ptr<const rvsdg::BitType> operandType,
      std::shared_ptr<const PointerType>)
      : UnaryOperation(std::move(operandType), PointerType::Create())
  {}

  // FIXME: The second parameter is unused, but we need to entangled the users of the constructor
  // first before we can eliminate it.
  explicit IntToPtrOperation(
      const std::shared_ptr<const rvsdg::Type> & operandType,
      const std::shared_ptr<const rvsdg::Type> &)
      : UnaryOperation(operandType, PointerType::Create())
  {
    checkAndExtractOperandType(operandType);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  size_t
  nbits() const noexcept
  {
    return std::static_pointer_cast<const rvsdg::BitType>(argument(0))->nbits();
  }

  static std::unique_ptr<ThreeAddressCode>
  create(const Variable * argument)
  {
    const auto operandType = checkAndExtractOperandType(argument->Type());
    auto op = std::make_unique<IntToPtrOperation>(operandType, PointerType::Create());
    return ThreeAddressCode::create(std::move(op), { argument });
  }

  static rvsdg::Output *
  create(rvsdg::Output * operand)
  {
    const auto operandType = checkAndExtractOperandType(operand->Type());
    return rvsdg::CreateOpNode<IntToPtrOperation>({ operand }, operandType, PointerType::Create())
        .output(0);
  }

private:
  static std::shared_ptr<const rvsdg::BitType>
  checkAndExtractOperandType(const std::shared_ptr<const rvsdg::Type> & operandType)
  {
    if (auto bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(operandType))
    {
      return bitType;
    }

    throw std::logic_error("expected bitstring type.");
  }
};

class FPToUIOperation final : public rvsdg::UnaryOperation
{
public:
  ~FPToUIOperation() noexcept override;

  FPToUIOperation(const fpsize size, std::shared_ptr<const rvsdg::BitType> resultType)
      : UnaryOperation(FloatingPointType::Create(size), std::move(resultType))
  {}

  FPToUIOperation(
      std::shared_ptr<const FloatingPointType> operandType,
      std::shared_ptr<const rvsdg::BitType> resultType)
      : UnaryOperation(std::move(operandType), std::move(resultType))
  {}

  FPToUIOperation(
      const std::shared_ptr<const rvsdg::Type> & operandType,
      const std::shared_ptr<const rvsdg::Type> & resultType)
      : UnaryOperation(operandType, resultType)
  {
    checkAndExtractOperandType(operandType);
    checkAndExtractResultType(resultType);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<ThreeAddressCode>
  create(const Variable * operand, const std::shared_ptr<const rvsdg::Type> & type)
  {
    auto operandType = checkAndExtractOperandType(operand->Type());
    auto bitType = checkAndExtractResultType(type);
    auto operation = std::make_unique<FPToUIOperation>(std::move(operandType), std::move(bitType));
    return ThreeAddressCode::create(std::move(operation), { operand });
  }

private:
  static std::shared_ptr<const FloatingPointType>
  checkAndExtractOperandType(const std::shared_ptr<const rvsdg::Type> & operandType)
  {
    if (auto fpType = std::dynamic_pointer_cast<const FloatingPointType>(operandType))
    {
      return fpType;
    }

    throw std::logic_error("Expected floating point type.");
  }

  static std::shared_ptr<const rvsdg::BitType>
  checkAndExtractResultType(const std::shared_ptr<const rvsdg::Type> & resultType)
  {
    if (auto bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(resultType))
    {
      return bitType;
    }

    throw std::logic_error("Expected bitstring type.");
  }
};

class FPToSIOperation final : public rvsdg::UnaryOperation
{
public:
  ~FPToSIOperation() noexcept override;

  FPToSIOperation(const fpsize size, std::shared_ptr<const rvsdg::BitType> resultType)
      : UnaryOperation(FloatingPointType::Create(size), std::move(resultType))
  {}

  FPToSIOperation(
      std::shared_ptr<const FloatingPointType> operandType,
      std::shared_ptr<const rvsdg::BitType> resultType)
      : UnaryOperation(std::move(operandType), std::move(resultType))
  {}

  FPToSIOperation(
      const std::shared_ptr<const rvsdg::Type> & operandType,
      const std::shared_ptr<const rvsdg::Type> & resultType)
      : UnaryOperation(operandType, resultType)
  {
    checkAndExtractOperandType(operandType);
    checkAndExtractResultType(resultType);
  }

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<ThreeAddressCode>
  create(const Variable * operand, const std::shared_ptr<const rvsdg::Type> & type)
  {
    auto operandType = checkAndExtractOperandType(operand->Type());
    auto bitType = checkAndExtractResultType(type);
    auto op = std::make_unique<FPToSIOperation>(std::move(operandType), std::move(bitType));
    return ThreeAddressCode::create(std::move(op), { operand });
  }

private:
  static std::shared_ptr<const FloatingPointType>
  checkAndExtractOperandType(const std::shared_ptr<const rvsdg::Type> & operandType)
  {
    if (auto fpType = std::dynamic_pointer_cast<const FloatingPointType>(operandType))
    {
      return fpType;
    }

    throw std::logic_error("Expected floating point type.");
  }

  static std::shared_ptr<const rvsdg::BitType>
  checkAndExtractResultType(const std::shared_ptr<const rvsdg::Type> & resultType)
  {
    if (auto bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(resultType))
    {
      return bitType;
    }

    throw std::logic_error("Expected bitstring type.");
  }
};

class ControlToIntOperation final : public rvsdg::SimpleOperation
{
public:
  ~ControlToIntOperation() noexcept override;

  ControlToIntOperation(
      std::shared_ptr<const rvsdg::ControlType> srctype,
      std::shared_ptr<const jlm::rvsdg::BitType> dsttype)
      : SimpleOperation({ std::move(srctype) }, { std::move(dsttype) })
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<llvm::ThreeAddressCode>
  create(const Variable * operand, const std::shared_ptr<const jlm::rvsdg::Type> & type)
  {
    auto st = std::dynamic_pointer_cast<const rvsdg::ControlType>(operand->Type());
    if (!st)
      throw util::Error("expected control type.");

    auto dt = std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(type);
    if (!dt)
      throw util::Error("expected bitstring type.");

    auto op = std::make_unique<ControlToIntOperation>(std::move(st), std::move(dt));
    return ThreeAddressCode::create(std::move(op), { operand });
  }
};

/**
  \brief Get address of compiled function object.
  */
class FunctionToPointerOperation final : public rvsdg::UnaryOperation
{
public:
  ~FunctionToPointerOperation() noexcept override;

  explicit FunctionToPointerOperation(std::shared_ptr<const rvsdg::FunctionType> fn);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<FunctionToPointerOperation>
  Create(std::shared_ptr<const rvsdg::FunctionType> fn);

  const std::shared_ptr<const rvsdg::FunctionType> &
  FunctionType() const noexcept
  {
    return FunctionType_;
  }

  /**
   * Performs the inversion of the \ref FunctionToPointerOperation by detecting that its operand is
   * a \ref PointerToFunctionOperation.
   *
   * f = PointerToFunctionOperation x
   * p = FunctionToPointerOperation f
   * =>
   * p = x
   *
   * @param operation The \ref FunctionToPointerOperation on which the transformation is performed.
   * @param operands The operands of the \ref FunctionToPointerOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref
   * FunctionToPointerOperation after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  invertFunctionToPointer(
      const FunctionToPointerOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

private:
  std::shared_ptr<const rvsdg::FunctionType> FunctionType_;
};

/**
  \brief Interpret pointer as callable function.
  */
class PointerToFunctionOperation final : public rvsdg::UnaryOperation
{
public:
  ~PointerToFunctionOperation() noexcept override;

  explicit PointerToFunctionOperation(std::shared_ptr<const rvsdg::FunctionType> fn);

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  static std::unique_ptr<PointerToFunctionOperation>
  Create(std::shared_ptr<const rvsdg::FunctionType> fn);

  const std::shared_ptr<const rvsdg::FunctionType> &
  FunctionType() const noexcept
  {
    return FunctionType_;
  }

  /**
   * Performs the inversion of the \ref PointerToFunctionOperation by detecting that its operand is
   * a \ref FunctionToPointerOperation.
   *
   * f = FunctionToPointerOperation x
   * p = PointerToFunctionOperation f
   * =>
   * p = x
   *
   * @param operation The \ref PointerToFunctionOperation on which the transformation is performed.
   * @param operands The operands of the \ref PointerToFunctionOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref
   * PointerToFunctionOperation after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  invertPointerToFunction(
      const PointerToFunctionOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

private:
  std::shared_ptr<const rvsdg::FunctionType> FunctionType_;
};

}

#endif
