/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
#define JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP

#include <jlm/rvsdg/binary.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/nullary.hpp>

namespace jlm::llvm
{

// FIXME: Implement our own value representation instead of re-using the bitstring value
// representation
using IntegerValueRepresentation = rvsdg::BitValueRepresentation;

class IntegerValueRepresentation final
{
public:
  IntegerValueRepresentation(const size_t numBits, int64_t value)
  {
    if (numBits == 0)
      throw std::runtime_error("Number of bits is zeo.");

    if (numBits < 64 && (value >> numBits) != 0 && (value >> numBits != -1))
      throw std::runtime_error("Value cannot be represented with the given number of bits.");

    InitializeToBitValue(numBits, 0);
    for (size_t n = 0; n < numBits; ++n)
    {
      SetBit(n, value & 1);
      value = value >> 1;
    }
  }

  size_t NumBits() const noexcept
  {
    return NumBits_;
  }

  size_t SignBit() const noexcept
  {
    return Bits_[NumBits() - 1];
  }

  IntegerValueRepresentation
  And(const IntegerValueRepresentation &other)
  {
    if (NumBits() != other.NumBits())
      throw std::runtime_error("Unequal number of bits.");

    IntegerValueRepresentation result(NumBits(), 0);
    for (size_t n = 0; n < NumBits(); n++)
      SetBit(n, GetBit(n) & other.GetBit(n));

    return result;
  }

  IntegerValueRepresentation
  Or(const IntegerValueRepresentation & other)
  {
    if (NumBits() != other.NumBits())
      throw std::runtime_error("Unequal number of bits.");

    IntegerValueRepresentation result(NumBits(), 0);
    for (size_t n = 0; n < NumBits(); n++)
      SetBit(n, GetBit(n) | other.GetBit(n));

    return result;
  }

  IntegerValueRepresentation
  XOr(const IntegerValueRepresentation & other)
  {
    if (NumBits() != other.NumBits())
      throw std::runtime_error("Unequal number of bits.");

    IntegerValueRepresentation result(NumBits(), 0);
    for (size_t n = 0; n < NumBits(); n++)
      SetBit(n, GetBit(n) ^ other.GetBit(n));

    return result;
  }

  IntegerValueRepresentation
  Add(const IntegerValueRepresentation & other) const
  {
    if (NumBits() != other.NumBits())
      throw std::runtime_error("Unequal number of bits.");

    uint64_t carry = 0;
    IntegerValueRepresentation sum(NumBits(), 0);
    for (size_t n = 0; n < NumBits(); n++)
    {
      sum.SetBit(n, Add(GetBit(n), other.GetBit(n), carry));
      carry = Carry(GetBit(n), other.GetBit(n), carry);
    }

    return sum;
  }

  IntegerValueRepresentation
  Negate() const
  {
    uint64_t c = 1;
    IntegerValueRepresentation result(NumBits(), 0);
    for (size_t n = 0; n < NumBits(); n++)
    {
      const uint64_t tmp = GetBit(n) ^ 1;
      result.SetBit(n, Add(tmp, 0, c));
      c = Carry(tmp, 0, c);
    }

    return result;
  }

  IntegerValueRepresentation
  Subtract(const IntegerValueRepresentation & other) const
  {
    return Add(other.Negate());
  }

  IntegerValueRepresentation
  Multiply(const IntegerValueRepresentation & other) const
  {
    if (NumBits() != other.NumBits())
      throw std::runtime_error("Unequal number of bits.");

    IntegerValueRepresentation product(2 * NumBits(), 0);
    mul(*this, other, product);
    return product.slice(0, nbits());
  }

private:
  static uint64_t Carry(const uint64_t x, const uint64_t y, const uint64_t z)
  {
    JLM_ASSERT(x == 0 || x == 1);
    JLM_ASSERT(y == 0 || y == 1);
    JLM_ASSERT(z == 0 || z == 1);

    return (x & y) | (x & z) | (y & z);
  }

  static uint64_t Add(const uint64_t x, const uint64_t y, const uint64_t z)
  {
    JLM_ASSERT(x == 0 || x == 1);
    JLM_ASSERT(y == 0 || y == 1);
    JLM_ASSERT(z == 0 || z == 1);

    return (x ^ y) ^ z;
  }

  void SetBit(const size_t bit, const uint64_t bitValue) noexcept
  {
    JLM_ASSERT(bit < NumBits());
    JLM_ASSERT(bitValue == 0 || bitValue == 1);

    const auto index = bit / 64;
    const auto remBits = bit % 64;
    Bits_[index] = Bits_[index] | (bitValue << remBits);
  }

  uint64_t GetBit(const size_t bit) const noexcept
  {
    JLM_ASSERT(bit < NumBits());

    const auto index = bit / 64;
    const auto remBits = bit % 64;
    return Bits_[index] & (1 << remBits);
  }

  void
  InitializeToBitValue(const size_t numBits, const uint64_t bitValue)
  {
    JLM_ASSERT(bitValue == 0 || bitValue == 1);

    size_t numIndices = numBits / 64;
    if (numBits % 64 != 0)
      numIndices++;

    Bits_ = std::vector(numIndices, bitValue);
  }

  // [LSB ... MSB]
  std::vector<uint64_t> Bits_{};
  size_t NumBits_{};
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
    return *util::AssertedCast<const rvsdg::BitType>(argument(0).get());
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
};

}

#endif // JLM_LLVM_IR_OPERATORS_INTEGEROPERATIONS_HPP
