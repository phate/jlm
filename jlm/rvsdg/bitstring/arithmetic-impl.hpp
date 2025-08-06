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
MakeBitUnaryOperation<reduction, name>::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const MakeBitUnaryOperation *>(&other);
  return op && op->type() == type();
}

template<typename reduction, const char * name>
BitValueRepresentation
MakeBitUnaryOperation<reduction, name>::reduce_constant(const BitValueRepresentation & arg) const
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
std::unique_ptr<Operation>
MakeBitUnaryOperation<reduction, name>::copy() const
{
  return std::make_unique<MakeBitUnaryOperation>(*this);
}

template<typename reduction, const char * name>
std::unique_ptr<BitUnaryOperation>
MakeBitUnaryOperation<reduction, name>::create(size_t nbits) const
{
  return std::make_unique<MakeBitUnaryOperation>(nbits);
}

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
MakeBitBinaryOperation<reduction, name, opflags>::~MakeBitBinaryOperation() noexcept
{}

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
bool
MakeBitBinaryOperation<reduction, name, opflags>::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const MakeBitBinaryOperation *>(&other);
  return op && op->type() == type();
}

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
BitValueRepresentation
MakeBitBinaryOperation<reduction, name, opflags>::reduce_constants(
    const BitValueRepresentation & arg1,
    const BitValueRepresentation & arg2) const
{
  return reduction{}(arg1, arg2);
}

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
enum BinaryOperation::flags
MakeBitBinaryOperation<reduction, name, opflags>::flags() const noexcept
{
  return opflags;
}

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
std::string
MakeBitBinaryOperation<reduction, name, opflags>::debug_string() const
{
  return jlm::util::strfmt(name, type().nbits());
}

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
std::unique_ptr<Operation>
MakeBitBinaryOperation<reduction, name, opflags>::copy() const
{
  return std::make_unique<MakeBitBinaryOperation>(*this);
}

template<typename reduction, const char * name, enum BinaryOperation::flags opflags>
std::unique_ptr<BitBinaryOperation>
MakeBitBinaryOperation<reduction, name, opflags>::create(size_t nbits) const
{
  return std::make_unique<MakeBitBinaryOperation>(nbits);
}

}

#endif
