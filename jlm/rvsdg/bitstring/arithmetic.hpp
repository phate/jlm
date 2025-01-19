/*
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_ARITHMETIC_HPP
#define JLM_RVSDG_BITSTRING_ARITHMETIC_HPP

#include <jlm/rvsdg/bitstring/bitoperation-classes.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::rvsdg
{

template<typename reduction, const char * name>
class MakeBitUnaryOperation final : public bitunary_op
{
public:
  ~MakeBitUnaryOperation() noexcept override;

  explicit MakeBitUnaryOperation(std::size_t nbits) noexcept
      : bitunary_op(bittype::Create(nbits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  bitvalue_repr
  reduce_constant(const bitvalue_repr & arg) const override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  std::unique_ptr<bitunary_op>
  create(size_t nbits) const override;

  static output *
  create(size_t nbits, output * op)
  {
    return CreateOpNode<MakeBitUnaryOperation>({ op }, nbits).output(0);
  }
};

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
class MakeBitBinaryOperation final : public bitbinary_op
{
public:
  ~MakeBitBinaryOperation() noexcept override;

  explicit MakeBitBinaryOperation(std::size_t nbits) noexcept
      : bitbinary_op(bittype::Create(nbits))
  {}

  bool
  operator==(const Operation & other) const noexcept override;

  enum BinaryOperation::flags
  flags() const noexcept override;

  bitvalue_repr
  reduce_constants(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const override;

  std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;

  std::unique_ptr<bitbinary_op>
  create(size_t nbits) const override;

  static output *
  create(size_t nbits, output * op1, output * op2)
  {
    return CreateOpNode<MakeBitBinaryOperation>({ op1, op2 }, nbits).output(0);
  }
};

struct reduce_neg;
extern const char BitNegateLabel[];
using bitneg_op = MakeBitUnaryOperation<reduce_neg, BitNegateLabel>;
extern template class MakeBitUnaryOperation<reduce_neg, BitNegateLabel>;

struct reduce_not;
extern const char BitNotLabel[];
using bitnot_op = MakeBitUnaryOperation<reduce_not, BitNotLabel>;
extern template class MakeBitUnaryOperation<reduce_not, BitNotLabel>;

struct reduce_add;
extern const char BitAddLabel[];
using bitadd_op = MakeBitBinaryOperation<
    reduce_add,
    BitAddLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;
extern template class MakeBitBinaryOperation<
    reduce_add,
    BitAddLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_and;
extern const char BitAndLabel[];
using bitand_op = MakeBitBinaryOperation<
    reduce_and,
    BitAndLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;
extern template class MakeBitBinaryOperation<
    reduce_and,
    BitAndLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_ashr;
extern const char BitAShrLabel[];
using bitashr_op = MakeBitBinaryOperation<reduce_ashr, BitAShrLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<
    reduce_ashr,
    BitAShrLabel,
    BinaryOperation::flags::none>;

struct reduce_mul;
extern const char BitMulLabel[];
using bitmul_op = MakeBitBinaryOperation<
    reduce_mul,
    BitMulLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;
extern template class MakeBitBinaryOperation<
    reduce_mul,
    BitMulLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_or;
extern const char BitOrLabel[];
using bitor_op = MakeBitBinaryOperation<
    reduce_or,
    BitOrLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;
extern template class MakeBitBinaryOperation<
    reduce_or,
    BitOrLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_sdiv;
extern const char BitSDivLabel[];
using bitsdiv_op = MakeBitBinaryOperation<reduce_sdiv, BitSDivLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<
    reduce_sdiv,
    BitSDivLabel,
    BinaryOperation::flags::none>;

struct reduce_shl;
extern const char BitShlLabel[];
using bitshl_op = MakeBitBinaryOperation<reduce_shl, BitShlLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<reduce_shl, BitShlLabel, BinaryOperation::flags::none>;

struct reduce_shr;
extern const char BitShrLabel[];
using bitshr_op = MakeBitBinaryOperation<reduce_shr, BitShrLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<reduce_shr, BitShrLabel, BinaryOperation::flags::none>;

struct reduce_smod;
extern const char BitSModLabel[];
using bitsmod_op = MakeBitBinaryOperation<reduce_smod, BitSModLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<
    reduce_smod,
    BitSModLabel,
    BinaryOperation::flags::none>;

struct reduce_smulh;
extern const char BitSMulHLabel[];
using bitsmulh_op =
    MakeBitBinaryOperation<reduce_smulh, BitSMulHLabel, BinaryOperation::flags::commutative>;
extern template class MakeBitBinaryOperation<
    reduce_smulh,
    BitSMulHLabel,
    BinaryOperation::flags::commutative>;

struct reduce_sub;
extern const char BitSubLabel[];
using bitsub_op = MakeBitBinaryOperation<reduce_sub, BitSubLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<reduce_sub, BitSubLabel, BinaryOperation::flags::none>;

struct reduce_udiv;
extern const char BitUDivLabel[];
using bitudiv_op = MakeBitBinaryOperation<reduce_udiv, BitUDivLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<
    reduce_udiv,
    BitUDivLabel,
    BinaryOperation::flags::none>;

struct reduce_umod;
extern const char BitUModLabel[];
using bitumod_op = MakeBitBinaryOperation<reduce_umod, BitUModLabel, BinaryOperation::flags::none>;
extern template class MakeBitBinaryOperation<
    reduce_umod,
    BitUModLabel,
    BinaryOperation::flags::none>;

struct reduce_umulh;
extern const char BitUMulHLabel[];
using bitumulh_op =
    MakeBitBinaryOperation<reduce_umulh, BitUMulHLabel, BinaryOperation::flags::commutative>;
extern template class MakeBitBinaryOperation<
    reduce_umulh,
    BitUMulHLabel,
    BinaryOperation::flags::commutative>;

struct reduce_xor;
extern const char BitXorLabel[];
using bitxor_op = MakeBitBinaryOperation<
    reduce_xor,
    BitXorLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;
extern template class MakeBitBinaryOperation<
    reduce_xor,
    BitXorLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

}

#endif
