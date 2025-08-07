/*
 * Copyright 2014 2015 2024 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/arithmetic-impl.hpp>
#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>

namespace jlm::rvsdg
{

// This provides the explicit template instantiations underlying
// all bitstring operation classes: This ensures that there is a
// single definition for all virtual functions, vmts etc in the
// rvsdg library (instead of the compiler template-instantiating
// them multiple times in each translation unit, and then relying
// on linker to do de-duplication).

struct reduce_neg
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg) const
  {
    return arg.neg();
  }
};

const char BitNegateLabel[] = "BitNegate";
template class MakeBitUnaryOperation<reduce_neg, BitNegateLabel>;

struct reduce_not
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg) const
  {
    return arg.lnot();
  }
};

const char BitNotLabel[] = "BitNot";
template class MakeBitUnaryOperation<reduce_not, BitNotLabel>;

struct reduce_add
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.add(arg2);
  }
};

const char BitAddLabel[] = "BitAdd";
template class MakeBitBinaryOperation<
    reduce_add,
    BitAddLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_and
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.land(arg2);
  }
};

const char BitAndLabel[] = "BitAnd";
template class MakeBitBinaryOperation<
    reduce_and,
    BitAndLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_ashr
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.ashr(arg2.to_uint());
  }
};

const char BitAShrLabel[] = "BitAShr";
template class MakeBitBinaryOperation<reduce_ashr, BitAShrLabel, BinaryOperation::flags::none>;

struct reduce_mul
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.mul(arg2);
  }
};

const char BitMulLabel[] = "BitMul";
template class MakeBitBinaryOperation<
    reduce_mul,
    BitMulLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_or
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.lor(arg2);
  }
};

const char BitOrLabel[] = "BitOr";
template class MakeBitBinaryOperation<
    reduce_or,
    BitOrLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

struct reduce_sdiv
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.sdiv(arg2);
  }
};

const char BitSDivLabel[] = "BitSDiv";
template class MakeBitBinaryOperation<reduce_sdiv, BitSDivLabel, BinaryOperation::flags::none>;

struct reduce_shl
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.shl(arg2.to_uint());
  }
};

const char BitShlLabel[] = "BitShl";
template class MakeBitBinaryOperation<reduce_shl, BitShlLabel, BinaryOperation::flags::none>;

struct reduce_shr
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.shr(arg2.to_uint());
  }
};

const char BitShrLabel[] = "BitShr";
template class MakeBitBinaryOperation<reduce_shr, BitShrLabel, BinaryOperation::flags::none>;

struct reduce_smod
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.smod(arg2);
  }
};

const char BitSModLabel[] = "BitSMod";
template class MakeBitBinaryOperation<reduce_smod, BitSModLabel, BinaryOperation::flags::none>;

struct reduce_smulh
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.smulh(arg2);
  }
};

const char BitSMulHLabel[] = "BitSMulH";
template class MakeBitBinaryOperation<
    reduce_smulh,
    BitSMulHLabel,
    BinaryOperation::flags::commutative>;

struct reduce_sub
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.sub(arg2);
  }
};

const char BitSubLabel[] = "BitSub";
template class MakeBitBinaryOperation<reduce_sub, BitSubLabel, BinaryOperation::flags::none>;

struct reduce_udiv
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.udiv(arg2);
  }
};

const char BitUDivLabel[] = "BitUDiv";
template class MakeBitBinaryOperation<reduce_udiv, BitUDivLabel, BinaryOperation::flags::none>;

struct reduce_umod
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.umod(arg2);
  }
};

const char BitUModLabel[] = "BitUMod";
template class MakeBitBinaryOperation<reduce_umod, BitUModLabel, BinaryOperation::flags::none>;

struct reduce_umulh
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.umulh(arg2);
  }
};

const char BitUMulHLabel[] = "BitUMulH";
template class MakeBitBinaryOperation<
    reduce_umulh,
    BitUMulHLabel,
    BinaryOperation::flags::commutative>;

struct reduce_xor
{
  BitValueRepresentation
  operator()(const BitValueRepresentation & arg1, const BitValueRepresentation & arg2) const
  {
    return arg1.lxor(arg2);
  }
};

const char BitXorLabel[] = "BitXor";
template class MakeBitBinaryOperation<
    reduce_xor,
    BitXorLabel,
    BinaryOperation::flags::associative | BinaryOperation::flags::commutative>;

}
