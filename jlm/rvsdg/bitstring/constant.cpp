/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/constant.hpp>

namespace jlm::rvsdg
{

BitConstantOperation::~BitConstantOperation() noexcept = default;

BitConstantOperation::BitConstantOperation(BitValueRepresentation value)
    : NullaryOperation(BitType::Create(value.nbits())),
      value_(std::move(value))
{}

bool
BitConstantOperation::operator==(const Operation & other) const noexcept
{
  // Compare with another BitConstantOperation by value
  if (auto * op = dynamic_cast<const BitConstantOperation *>(&other))
    return op->value_ == value_;

  return false;
}

std::string
BitConstantOperation::debug_string() const
{
  if (value_.is_known() && value_.nbits() <= 64)
    return jlm::util::strfmt("BITS", value_.nbits(), "(", value_.to_uint(), ")");

  return value_.str();
}

std::unique_ptr<Operation>
BitConstantOperation::copy() const
{
  return std::make_unique<BitConstantOperation>(value_);
}

}
