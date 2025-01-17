/*
 * Copyright 2014 2024 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/comparison-impl.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>

namespace jlm::rvsdg
{

struct reduce_eq
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.eq(arg2);
  }
};

const char BitEqLabel[] = "BitEq";
template class MakeBitComparisonOperation<
    reduce_eq,
    BitEqLabel,
    BinaryOperation::flags::commutative>;

struct reduce_ne
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.ne(arg2);
  }
};

const char BitNeLabel[] = "BitNe";
template class MakeBitComparisonOperation<
    reduce_ne,
    BitNeLabel,
    BinaryOperation::flags::commutative>;

struct reduce_sge
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.sge(arg2);
  }
};

const char BitSgeLabel[] = "BitSge";
template class MakeBitComparisonOperation<reduce_sge, BitSgeLabel, BinaryOperation::flags::none>;

struct reduce_sgt
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.sgt(arg2);
  }
};

const char BitSgtLabel[] = "BitSgt";
template class MakeBitComparisonOperation<reduce_sgt, BitSgtLabel, BinaryOperation::flags::none>;

struct reduce_sle
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.sle(arg2);
  }
};

const char BitSleLabel[] = "BitSle";
template class MakeBitComparisonOperation<reduce_sle, BitSleLabel, BinaryOperation::flags::none>;

struct reduce_slt
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.slt(arg2);
  }
};

const char BitSltLabel[] = "BitSlt";
template class MakeBitComparisonOperation<reduce_slt, BitSltLabel, BinaryOperation::flags::none>;

struct reduce_uge
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.uge(arg2);
  }
};

const char BitUgeLabel[] = "BitUge";
template class MakeBitComparisonOperation<reduce_uge, BitUgeLabel, BinaryOperation::flags::none>;

struct reduce_ugt
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.ugt(arg2);
  }
};

const char BitUgtLabel[] = "BitUgt";
template class MakeBitComparisonOperation<reduce_ugt, BitUgtLabel, BinaryOperation::flags::none>;

struct reduce_ule
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.ule(arg2);
  }
};

const char BitUleLabel[] = "BitUle";
template class MakeBitComparisonOperation<reduce_ule, BitUleLabel, BinaryOperation::flags::none>;

struct reduce_ult
{
  char
  operator()(const bitvalue_repr & arg1, const bitvalue_repr & arg2) const
  {
    return arg1.ult(arg2);
  }
};

const char BitUltLabel[] = "BitUlt";
template class MakeBitComparisonOperation<reduce_ult, BitUltLabel, BinaryOperation::flags::none>;

}
