/*
 * Copyright 2024 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_ARITHMETIC_IMPL_HPP
#define JLM_RVSDG_BITSTRING_ARITHMETIC_IMPL_HPP

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::rvsdg
{

template<typename reduction, const char * name>
MakeBitUnaryOperation<reduction, name>::~MakeBitUnaryOperation() noexcept
{}

template<typename reduction, const char * name>
bool
MakeBitUnaryOperation<reduction, name>::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const MakeBitUnaryOperation *>(&other);
  return op && op->type() == type();
}

template<typename reduction, const char * name>
bitvalue_repr
MakeBitUnaryOperation<reduction, name>::reduce_constant(const bitvalue_repr & arg) const
{
  return reduction{}(arg);
}

template<typename reduction, const char * name>
std::string
MakeBitUnaryOperation<reduction, name>::debug_string() const
{
  return jlm::util::strfmt(name, type().nbits());
}

template<typename reduction, const char * name>
std::unique_ptr<operation>
MakeBitUnaryOperation<reduction, name>::copy() const
{
  return std::make_unique<MakeBitUnaryOperation>(*this);
}

template<typename reduction, const char * name>
std::unique_ptr<bitunary_op>
MakeBitUnaryOperation<reduction, name>::create(size_t nbits) const
{
  return std::make_unique<MakeBitUnaryOperation>(nbits);
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
MakeBitBinaryOperation<reduction, name, opflags>::~MakeBitBinaryOperation() noexcept
{}

template<typename reduction, const char * name, enum binary_op::flags opflags>
bool
MakeBitBinaryOperation<reduction, name, opflags>::operator==(const operation & other) const noexcept
{
  auto op = dynamic_cast<const MakeBitBinaryOperation *>(&other);
  return op && op->type() == type();
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
bitvalue_repr
MakeBitBinaryOperation<reduction, name, opflags>::reduce_constants(
    const bitvalue_repr & arg1,
    const bitvalue_repr & arg2) const
{
  return reduction{}(arg1, arg2);
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
enum binary_op::flags
MakeBitBinaryOperation<reduction, name, opflags>::flags() const noexcept
{
  return opflags;
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
std::string
MakeBitBinaryOperation<reduction, name, opflags>::debug_string() const
{
  return jlm::util::strfmt(name, type().nbits());
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
std::unique_ptr<operation>
MakeBitBinaryOperation<reduction, name, opflags>::copy() const
{
  return std::make_unique<MakeBitBinaryOperation>(*this);
}

template<typename reduction, const char * name, enum binary_op::flags opflags>
std::unique_ptr<bitbinary_op>
MakeBitBinaryOperation<reduction, name, opflags>::create(size_t nbits) const
{
  return std::make_unique<MakeBitBinaryOperation>(nbits);
}

}

#endif
