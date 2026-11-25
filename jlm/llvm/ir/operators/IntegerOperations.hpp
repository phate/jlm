/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/nullary.hpp>

#include <cstring>
#include <limits>
#include <stdexcept>

namespace jlm::llvm
{

class IntegerValueRepresentation final
{
public:
  IntegerValueRepresentation(const size_t nbits, const int64_t value)
      : numBits_(nbits),
        bits_(nbits / 64 + (nbits % 64 != 0))
  {
    if (!nbits)
      throw util::Error("Number of bits is zero.");
    if (nbits < 64 && (value >> nbits) != 0 && (value >> nbits) != -1)
      throw util::Error("Value cannot be represented with the given number of bits.");
    const uint64_t bits = static_cast<uint64_t>(value);
    for (size_t i = 0; i < nbits; ++i)
      setBit(i, (i < 64 ? ((bits >> i) & 1) != 0 : value < 0) ? '1' : '0');
  }

  IntegerValueRepresentation(const char * bits)
      : IntegerValueRepresentation(std::strlen(bits), '0', 0)
  {
    for (size_t i = 0; i < numBits_; ++i)
      setBit(i, bits[i]);
  }

  IntegerValueRepresentation(const uint8_t bit)
      : IntegerValueRepresentation(1, bit, 0)
  {}

  size_t
  nbits() const noexcept
  {
    return numBits_;
  }

  size_t
  NumBits() const noexcept
  {
    return numBits_;
  }

  uint8_t
  sign() const noexcept
  {
    return getBit(numBits_ - 1);
  }

  bool
  is_negative() const noexcept
  {
    return sign() == '1';
  }

  bool
  isZero() const noexcept
  {
    for (size_t i = 0; i < numBits_; ++i)
      if (getBit(i) != '0')
        return false;
    return true;
  }

  std::string
  str() const
  {
    std::string result;
    result.reserve(numBits_);
    for (size_t i = 0; i < numBits_; ++i)
      result.push_back(getBit(i));
    return result;
  }

  bool
  operator==(const IntegerValueRepresentation & other) const noexcept
  {
    return numBits_ == other.numBits_ && bits_ == other.bits_;
  }

  bool
  operator!=(const IntegerValueRepresentation & other) const noexcept
  {
    return !(*this == other);
  }

  bool
  operator==(int64_t value) const
  {
    return *this == IntegerValueRepresentation(numBits_, value);
  }

  bool
  operator!=(int64_t value) const
  {
    return !(*this == value);
  }

  bool
  operator==(const std::string & value) const
  {
    return str() == value;
  }

  bool
  operator!=(const std::string & value) const
  {
    return !(*this == value);
  }

  static IntegerValueRepresentation
  repeat(size_t nbits, uint8_t bit)
  {
    return IntegerValueRepresentation(nbits, bit, 0);
  }

  static IntegerValueRepresentation
  create(const std::vector<IntegerValueRepresentation> & values)
  {
    JLM_ASSERT(!values.empty());
    auto result = values.front();
    for (size_t i = 1; i < values.size(); ++i)
      result.append(values[i]);
    return result;
  }

  void
  append(const IntegerValueRepresentation & other)
  {
    *this = concat(other);
  }

  uint64_t
  to_uint() const
  {
    uint64_t result = 0;
    for (size_t i = 0; i < numBits_; ++i)
    {
      const uint8_t bit = getBit(i);
      if (bit == '1')
      {
        if (i >= 64)
          throw std::range_error("Bit constant value exceeds uint64 range");
        result |= uint64_t(1) << i;
      }
    }
    return result;
  }

  int64_t
  to_int() const
  {
    const uint8_t s = sign();
    uint64_t result = s == '1' ? ~uint64_t(0) : 0;
    for (size_t i = 0; i < numBits_; ++i)
    {
      const uint8_t bit = getBit(i);
      if (i >= 63 && bit != s)
        throw std::range_error("Bit constant value exceeds int64 range");
      if (i < 64)
        result = (result & ~(uint64_t(1) << i)) | (uint64_t(bit == '1') << i);
    }
    if ((result >> 63) == 0)
      return static_cast<int64_t>(result);
    return -1 - static_cast<int64_t>(~result);
  }

  IntegerValueRepresentation
  concat(const IntegerValueRepresentation & other) const
  {
    if (other.numBits_ > std::numeric_limits<size_t>::max() - numBits_)
      throw util::Error("Width overflow.");
    auto result = repeat(numBits_ + other.numBits_, '0');
    for (size_t i = 0; i < numBits_; ++i)
      result.setBit(i, getBit(i));
    for (size_t i = 0; i < other.numBits_; ++i)
      result.setBit(numBits_ + i, other.getBit(i));
    return result;
  }

  IntegerValueRepresentation
  add(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    auto result = repeat(numBits_, '0');
    uint8_t carry = '0';
    for (size_t i = 0; i < numBits_; ++i)
    {
      uint8_t a = getBit(i), b = other.getBit(i);
      result.setBit(i, bitXor(bitXor(a, b), carry));
      carry = bitOr(bitOr(bitAnd(a, b), bitAnd(a, carry)), bitAnd(b, carry));
    }
    return result;
  }

  IntegerValueRepresentation
  sub(const IntegerValueRepresentation & other) const
  {
    return add(other.neg());
  }

  IntegerValueRepresentation
  land(const IntegerValueRepresentation & other) const
  {
    return bitwise(other, bitAnd);
  }

  IntegerValueRepresentation
  lor(const IntegerValueRepresentation & other) const
  {
    return bitwise(other, bitOr);
  }

  IntegerValueRepresentation
  lxor(const IntegerValueRepresentation & other) const
  {
    return bitwise(other, bitXor);
  }

  IntegerValueRepresentation
  lnot() const
  {
    return lxor(repeat(numBits_, '1'));
  }

  IntegerValueRepresentation
  neg() const
  {
    return lnot().add(IntegerValueRepresentation(numBits_, 1));
  }

  uint8_t
  ne(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    uint8_t result = '0';
    for (size_t i = 0; i < numBits_; ++i)
      result = bitOr(result, bitXor(getBit(i), other.getBit(i)));
    return result;
  }

  uint8_t
  eq(const IntegerValueRepresentation & other) const
  {
    return bitNot(ne(other));
  }

  uint8_t
  ult(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    uint8_t v = bitAnd(bitNot(getBit(0)), other.getBit(0));
    for (size_t i = 1; i < numBits_; ++i)
      v = bitAnd(
          bitOr(bitNot(getBit(i)), other.getBit(i)),
          bitOr(bitAnd(bitNot(getBit(i)), other.getBit(i)), v));
    return v;
  }

  uint8_t
  ule(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    uint8_t v = '1';
    for (size_t i = 0; i < numBits_; ++i)
      v = bitAnd(
          bitAnd(bitOr(bitNot(getBit(i)), other.getBit(i)), bitOr(bitNot(getBit(i)), v)),
          bitOr(v, other.getBit(i)));
    return v;
  }

  uint8_t
  slt(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    auto a = *this, b = other;
    a.setBit(numBits_ - 1, bitNot(sign()));
    b.setBit(numBits_ - 1, bitNot(other.sign()));
    return a.ult(b);
  }

  uint8_t
  sle(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    auto a = *this, b = other;
    a.setBit(numBits_ - 1, bitNot(sign()));
    b.setBit(numBits_ - 1, bitNot(other.sign()));
    return a.ule(b);
  }

  uint8_t
  uge(const IntegerValueRepresentation & other) const
  {
    return bitNot(ult(other));
  }

  uint8_t
  ugt(const IntegerValueRepresentation & other) const
  {
    return bitNot(ule(other));
  }

  uint8_t
  sge(const IntegerValueRepresentation & other) const
  {
    return bitNot(slt(other));
  }

  uint8_t
  sgt(const IntegerValueRepresentation & other) const
  {
    return bitNot(sle(other));
  }

  IntegerValueRepresentation
  slice(const size_t low, const size_t high) const
  {
    if (high <= low || high > numBits_)
      throw util::Error("Slice is out of bound.");
    auto result = repeat(high - low, '0');
    for (size_t i = low; i < high; ++i)
      result.setBit(i - low, getBit(i));
    return result;
  }

  IntegerValueRepresentation
  trunc(const size_t n) const
  {
    return slice(0, n);
  }

  IntegerValueRepresentation
  zext(const size_t n) const
  {
    return n ? concat(repeat(n, '0')) : *this;
  }

  IntegerValueRepresentation
  sext(const size_t n) const
  {
    return n ? concat(repeat(n, sign())) : *this;
  }

  IntegerValueRepresentation
  shl(const size_t n) const
  {
    auto result = repeat(numBits_, '0');
    if (n < numBits_)
      for (size_t i = n; i < numBits_; ++i)
        result.setBit(i, getBit(i - n));
    return result;
  }

  IntegerValueRepresentation
  shr(const size_t n) const
  {
    return shiftRight(n, '0');
  }

  IntegerValueRepresentation
  ashr(const size_t n) const
  {
    return shiftRight(n, sign());
  }

  IntegerValueRepresentation
  mul(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    auto result = repeat(numBits_, '0');
    for (size_t i = 0; i < numBits_; ++i)
      result = result.add(land(repeat(numBits_, other.getBit(i))).shl(i));
    return result;
  }

  IntegerValueRepresentation
  umulh(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    return zext(numBits_).mul(other.zext(numBits_)).slice(numBits_, 2 * numBits_);
  }

  IntegerValueRepresentation
  smulh(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    return sext(numBits_).mul(other.sext(numBits_)).slice(numBits_, 2 * numBits_);
  }

  IntegerValueRepresentation
  udiv(const IntegerValueRepresentation & other) const
  {
    return divide(other, false, false);
  }

  IntegerValueRepresentation
  umod(const IntegerValueRepresentation & other) const
  {
    return divide(other, false, true);
  }

  IntegerValueRepresentation
  sdiv(const IntegerValueRepresentation & other) const
  {
    return divide(other, true, false);
  }

  IntegerValueRepresentation
  smod(const IntegerValueRepresentation & other) const
  {
    return divide(other, true, true);
  }

private:
  IntegerValueRepresentation(const size_t nbits, const uint8_t bit, int)
      : numBits_(nbits),
        bits_(nbits / 64 + (nbits % 64 != 0))
  {
    if (!nbits)
      throw util::Error("Number of bits is zero.");
    for (size_t i = 0; i < nbits; ++i)
      setBit(i, bit);
  }

  static uint8_t
  bitAnd(const uint8_t a, const uint8_t b)
  {
    return a == '1' && b == '1' ? '1' : '0';
  }

  static uint8_t
  bitOr(const uint8_t a, const uint8_t b)
  {
    return a == '1' || b == '1' ? '1' : '0';
  }

  static uint8_t
  bitXor(const uint8_t a, const uint8_t b)
  {
    return a == b ? '0' : '1';
  }

  static uint8_t
  bitNot(const uint8_t a)
  {
    return bitXor(a, '1');
  }

  IntegerValueRepresentation
  bitwise(const IntegerValueRepresentation & other, uint8_t (*op)(uint8_t, uint8_t)) const
  {
    checkWidth(other);
    auto result = repeat(numBits_, '0');
    for (size_t i = 0; i < numBits_; ++i)
      result.setBit(i, op(getBit(i), other.getBit(i)));
    return result;
  }

  IntegerValueRepresentation
  shiftRight(const size_t n, const uint8_t fill) const
  {
    auto result = repeat(numBits_, fill);
    if (n < numBits_)
      for (size_t i = 0; i < numBits_ - n; ++i)
        result.setBit(i, getBit(i + n));
    return result;
  }

  IntegerValueRepresentation
  divide(
      const IntegerValueRepresentation & other,
      const bool signedDivision,
      const bool remainderResult) const
  {
    checkDivisor(other);
    const bool negativeDividend = signedDivision && is_negative();
    const bool negativeDivisor = signedDivision && other.is_negative();
    auto dividend = negativeDividend ? neg() : *this;
    auto divisor = negativeDivisor ? other.neg() : other;
    auto quotient = repeat(numBits_, '0');
    auto remainder = repeat(numBits_, '0');
    for (size_t i = numBits_; i-- > 0;)
    {
      const bool overflow = remainder.sign() == '1';
      remainder = remainder.shl(1);
      remainder.setBit(0, dividend.getBit(i));
      if (overflow || remainder.uge(divisor) == '1')
      {
        remainder = remainder.sub(divisor);
        quotient.setBit(i, '1');
      }
    }
    if (remainderResult)
      return negativeDividend ? remainder.neg() : remainder;
    return negativeDividend != negativeDivisor ? quotient.neg() : quotient;
  }

  void
  checkWidth(const IntegerValueRepresentation & other) const
  {
    if (numBits_ != other.numBits_)
      throw util::Error("Unequal number of bits.");
  }

  void
  checkDivisor(const IntegerValueRepresentation & other) const
  {
    checkWidth(other);
    if (other.isZero())
      throw util::Error("Division by zero.");
  }

  uint8_t
  getBit(const size_t index) const noexcept
  {
    JLM_ASSERT(index < numBits_);
    const uint64_t mask = static_cast<uint64_t>(1) << (index % 64);
    return (bits_[index / 64] & mask) != 0 ? '1' : '0';
  }

  void
  setBit(const size_t index, const uint8_t bit)
  {
    JLM_ASSERT(index < numBits_);
    JLM_ASSERT(bit == '0' || bit == '1');

    const uint64_t mask = static_cast<uint64_t>(1) << (index % 64);
    auto & value = bits_[index / 64];
    value = (value & ~mask) | (bit == '1' ? mask : 0);
  }

  size_t numBits_;

  /**
   * Bit 0 is the LSB and bit numBits_ - 1 is the MSB.
   */
  std::vector<uint64_t> bits_;
};

/**
 * Represents an LLVM integer constant
 */
class IntegerConstantOperation final : public rvsdg::NullaryOperation
{
public:
  ~IntegerConstantOperation() override;

  explicit IntegerConstantOperation(IntegerValueRepresentation representation)
      : NullaryOperation(rvsdg::BitType::Create(representation.nbits())),
        Representation_(std::move(representation))
  {}

  std::unique_ptr<Operation>
  copy() const override;

  std::string
  debug_string() const override;

  bool
  operator==(const Operation & other) const noexcept override;

  [[nodiscard]] const IntegerValueRepresentation &
  Representation() const noexcept
  {
    return Representation_;
  }

  static rvsdg::Node &
  Create(rvsdg::Region & region, IntegerValueRepresentation representation)
  {
    return rvsdg::CreateOpNode<IntegerConstantOperation>(region, std::move(representation));
  }

  static rvsdg::Node &
  Create(rvsdg::Region & region, std::size_t numBits, std::int64_t value)
  {
    return Create(region, { numBits, value });
  }

private:
  IntegerValueRepresentation Representation_;
};

/**
 * Represents an LLVM integer binary operation
 */
class IntegerBinaryOperation : public rvsdg::BinaryOperation
{
public:
  ~IntegerBinaryOperation() noexcept override;

  IntegerBinaryOperation(
      const std::size_t numArgumentBits,
      const std::size_t numResultBits) noexcept
      : BinaryOperation(
            { rvsdg::BitType::Create(numArgumentBits), rvsdg::BitType::Create(numArgumentBits) },
            rvsdg::BitType::Create(numResultBits))
  {}

  [[nodiscard]] const rvsdg::BitType &
  Type() const noexcept
  {
    return *util::assertedCast<const rvsdg::BitType>(argument(0).get());
  }
};

/**
 * This operation is equivalent to LLVM's 'add' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#add-instruction) for more
 * details.
 */
class IntegerAddOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAddOperation() noexcept override;

  explicit IntegerAddOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerAddOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerAddOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerAddOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerAddOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerAddOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   *
   * v = IntegerAddOperation x 0
   * =>
   * v = x
   *
   * @param operation The \ref IntegerAddOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerAddOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerAddOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdempotent(
      const IntegerAddOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'sub' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#sub-instruction) for more
 * details.
 */
class IntegerSubOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSubOperation() noexcept override;

  explicit IntegerSubOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSubOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Normalizes the additive inverse of a \ref IntegerSubOperation node:
   *
   * z = IntegerSubOperation x x
   * =>
   * z = 0
   *
   * @param operation The \ref IntegerSubOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSubOperation node.
   *
   * @return If the normalization could be applied, then 0. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeAdditiveInverse(
      const IntegerSubOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerSubOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSubOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSubOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSubOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'mul' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#mul-instruction) for more
 * details.
 */
class IntegerMulOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerMulOperation() noexcept override;

  explicit IntegerMulOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerMulOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerMulOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerMulOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerMulOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerMulOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'sdiv' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#sdiv-instruction) for
 * more details.
 */
class IntegerSDivOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSDivOperation() noexcept override;

  explicit IntegerSDivOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSDivOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerSDivOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSDivOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSDivOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSDivOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'udiv' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#udiv-instruction) for
 * more details.
 */
class IntegerUDivOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUDivOperation() noexcept override;

  explicit IntegerUDivOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUDivOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerUDivOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUDivOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUDivOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUDivOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'srem' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#srem-instruction) for
 * more details.
 */
class IntegerSRemOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSRemOperation() noexcept override;

  explicit IntegerSRemOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSRemOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerSRemOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSRemOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSRemOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSRemOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'urem' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#urem-instruction) for
 * more details.
 */
class IntegerURemOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerURemOperation() noexcept override;

  explicit IntegerURemOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerURemOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerURemOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerURemOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerURemOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerURemOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'ashr' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#ashr-instruction) for
 * more details.
 */
class IntegerAShrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAShrOperation() noexcept override;

  explicit IntegerAShrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerAShrOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerAShrOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerAShrOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerAShrOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerAShrOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'shl' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#shl-instruction) for more
 * details.
 */
class IntegerShlOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerShlOperation() noexcept override;

  explicit IntegerShlOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerShlOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerShlOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerShlOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerShlOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerShlOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'lshr' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#lshr-instruction) for
 * more details.
 */
class IntegerLShrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerLShrOperation() noexcept override;

  explicit IntegerLShrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerLShrOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operation's result with the resulting constant.
   *
   * @param operation The \ref IntegerLShrOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerLShrOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerLShrOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerLShrOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'and' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#and-instruction) for more
 * details.
 */
class IntegerAndOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerAndOperation() noexcept override;

  explicit IntegerAndOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerAndOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerAndOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerAndOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerAndOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerAndOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'or' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#or-instruction) for more
 * details.
 */
class IntegerOrOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerOrOperation() noexcept override;

  explicit IntegerOrOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerOrOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerOrOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerOrOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerOrOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerOrOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   *
   * v = IntegerOrOperation x 0
   * =>
   * v = x
   *
   * @param operation The \ref IntegerOrOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerOrOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerOrOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdempotent(
      const IntegerOrOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'xor' instruction for integer operands.
 * See [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html#xor-instruction) for more
 * details.
 */
class IntegerXorOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerXorOperation() noexcept override;

  explicit IntegerXorOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, numBits)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerXorOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerXorOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerXorOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerXorOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerXorOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'eq' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerEqOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerEqOperation() noexcept override;

  explicit IntegerEqOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerEqOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerEqOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerEqOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerEqOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerEqOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerEqOperation x x
   * =>
   * y = 1
   *
   * @param operation The \ref IntegerEqOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerEqOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerEqOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerEqOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ne' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerNeOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerNeOperation() noexcept override;

  explicit IntegerNeOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerNeOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerNeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerNeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerNeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerNeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerNeOperation x x
   * =>
   * y = 0
   *
   * @param operation The \ref IntegerNeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerNeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerNeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerNeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'sge' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSgeOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSgeOperation() noexcept override;

  explicit IntegerSgeOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSgeOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSgeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSgeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSgeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSgeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerSgeOperation x x
   * =>
   * y = 1
   *
   * @param operation The \ref IntegerSgeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSgeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSgeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerSgeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'sgt' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSgtOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSgtOperation() noexcept override;

  explicit IntegerSgtOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSgtOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSgtOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSgtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSgtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSgtOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerSgtOperation x x
   * =>
   * y = 0
   *
   * @param operation The \ref IntegerSgtOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSgtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSgtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerSgtOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'sle' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSleOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSleOperation() noexcept override;

  explicit IntegerSleOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSleOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSleOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSleOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSleOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSleOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerSleOperation x x
   * =>
   * y = 1
   *
   * @param operation The \ref IntegerSleOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSleOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSleOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerSleOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'slt' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerSltOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerSltOperation() noexcept override;

  explicit IntegerSltOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerSltOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerSltOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSltOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSltOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerSltOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerSltOperation x x
   * =>
   * y = 0
   *
   * @param operation The \ref IntegerSltOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerSltOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerSltOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerSltOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'uge' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUgeOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUgeOperation() noexcept override;

  explicit IntegerUgeOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUgeOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUgeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUgeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUgeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUgeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerUgeOperation x x
   * =>
   * y = 1
   *
   * @param operation The \ref IntegerUgeOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUgeOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUgeOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerUgeOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ugt' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUgtOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUgtOperation() noexcept override;

  explicit IntegerUgtOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUgtOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUgtOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUgtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUgtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUgtOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerUgtOperation x x
   * =>
   * y = 0
   *
   * @param operation The \ref IntegerUgtOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUgtOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUgtOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerUgtOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ule' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUleOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUleOperation() noexcept override;

  explicit IntegerUleOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUleOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUleOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUleOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUleOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUleOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerUleOperation x x
   * =>
   * y = 1
   *
   * @param operation The \ref IntegerUleOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUleOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUleOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerUleOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

/**
 * This operation is equivalent to LLVM's 'icmp' instruction with condition 'ult' for integer
 * operands. See [LLVM Language Reference
 * Manual](https://llvm.org/docs/LangRef.html#icmp-instruction) for more details.
 */
class IntegerUltOperation final : public IntegerBinaryOperation
{
public:
  ~IntegerUltOperation() noexcept override;

  explicit IntegerUltOperation(const std::size_t numBits)
      : IntegerBinaryOperation(numBits, 1)
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  rvsdg::binop_reduction_path_t
  can_reduce_operand_pair(const rvsdg::Output * op1, const rvsdg::Output * op2)
      const noexcept override;

  rvsdg::Output *
  reduce_operand_pair(rvsdg::binop_reduction_path_t path, rvsdg::Output * op1, rvsdg::Output * op2)
      const override;

  enum flags
  flags() const noexcept override;

  static rvsdg::Node &
  createNode(const size_t numBits, rvsdg::Output & operand1, rvsdg::Output & operand2)
  {
    return rvsdg::CreateOpNode<IntegerUltOperation>({ &operand1, &operand2 }, numBits);
  }

  /**
   * Performs constant folding by statically evaluating the two constant operands and replacing the
   * operations result with the resulting constant.
   *
   * @param operation The \ref IntegerUltOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUltOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUltOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  foldConstants(
      const IntegerUltOperation & operation,
      const std::vector<rvsdg::Output *> & operands);

  /**
   * Performs the following normalization:
   * y = IntegerUltOperation x x
   * =>
   * y = 0
   *
   * @param operation The \ref IntegerUltOperation on which the transformation is performed.
   * @param operands The operands of the \ref IntegerUltOperation node.
   *
   * @return If the normalization could be applied, then the result of the \ref IntegerUltOperation
   * after the transformation. Otherwise, std::nullopt.
   */
  static std::optional<std::vector<rvsdg::Output *>>
  normalizeIdenticalOperands(
      const IntegerUltOperation & operation,
      const std::vector<rvsdg::Output *> & operands);
};

}

#endif // JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
