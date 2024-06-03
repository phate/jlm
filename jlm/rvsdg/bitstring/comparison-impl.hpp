/*
 * Copyright 2024 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_COMPARISON_IMPL_HPP
#define JLM_RVSDG_BITSTRING_COMPARISON_IMPL_HPP

#include <jlm/rvsdg/bitstring/comparison.hpp>

namespace jlm::rvsdg
{

template<typename reduction, const char * name, enum binary_op::flags opflags>
MakeBitComparisonOperation<reduction, name, opflags>::~MakeBitComparisonOperation() noexcept
{}

template<typename reduction, const char * name, enum binary_op::flags opflags>
bool
MakeBitComparisonOperation<reduction, name, opflags>::operator==(
    const operation & other) const noexcept
{
  auto op = dynamic_cast<const MakeBitComparisonOperation<reduction, name, opflags> *>(&other);
  return op && op->type() == type();
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
compare_result
MakeBitComparisonOperation<reduction, name, opflags>::reduce_constants(
    const bitvalue_repr & arg1,
    const bitvalue_repr & arg2) const
{
  switch (reduction()(arg1, arg2))
  {
  case '0':
    return compare_result::static_false;
  case '1':
    return compare_result::static_true;
  default:
    return compare_result::undecidable;
  }
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
enum binary_op::flags
MakeBitComparisonOperation<reduction, name, opflags>::flags() const noexcept
{
  return opflags;
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
std::string
MakeBitComparisonOperation<reduction, name, opflags>::debug_string() const
{
  return jlm::util::strfmt(name, type().nbits());
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
std::unique_ptr<jlm::rvsdg::operation>
MakeBitComparisonOperation<reduction, name, opflags>::copy() const
{
  return std::make_unique<MakeBitComparisonOperation>(*this);
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
std::unique_ptr<bitcompare_op>
MakeBitComparisonOperation<reduction, name, opflags>::create(size_t nbits) const
{
  return std::make_unique<MakeBitComparisonOperation>(nbits);
}

}

#endif
