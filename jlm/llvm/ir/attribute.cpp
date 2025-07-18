/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/attribute.hpp>

namespace jlm::llvm
{

Attribute::~Attribute() noexcept = default;

StringAttribute::~StringAttribute() noexcept = default;

bool
StringAttribute::operator==(const Attribute & other) const
{
  auto sa = dynamic_cast<const StringAttribute *>(&other);
  return sa && sa->kind() == kind() && sa->value() == value();
}

EnumAttribute::~EnumAttribute() noexcept = default;

bool
EnumAttribute::operator==(const Attribute & other) const
{
  auto ea = dynamic_cast<const EnumAttribute *>(&other);
  return ea && ea->kind() == kind();
}

IntAttribute::~IntAttribute() noexcept = default;

bool
IntAttribute::operator==(const Attribute & other) const
{
  auto ia = dynamic_cast<const IntAttribute *>(&other);
  return ia && ia->kind() == kind() && ia->value() == value();
}

TypeAttribute::~TypeAttribute() noexcept = default;

bool
TypeAttribute::operator==(const Attribute & other) const
{
  auto ta = dynamic_cast<const TypeAttribute *>(&other);
  return ta && ta->kind() == kind() && ta->type() == type();
}

AttributeSet::EnumAttributeRange
AttributeSet::EnumAttributes() const
{
  return EnumAttributes_.Items();
}

AttributeSet::IntAttributeRange
AttributeSet::IntAttributes() const
{
  return IntAttributes_.Items();
}

AttributeSet::TypeAttributeRange
AttributeSet::TypeAttributes() const
{
  return TypeAttributes_.Items();
}

AttributeSet::StringAttributeRange
AttributeSet::StringAttributes() const
{
  return StringAttributes_.Items();
}

}
