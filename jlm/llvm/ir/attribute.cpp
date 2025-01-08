/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/attribute.hpp>

namespace jlm::llvm
{

attribute::~attribute() noexcept = default;

string_attribute::~string_attribute() noexcept = default;

bool
string_attribute::operator==(const attribute & other) const
{
  auto sa = dynamic_cast<const string_attribute *>(&other);
  return sa && sa->kind() == kind() && sa->value() == value();
}

enum_attribute::~enum_attribute() noexcept = default;

bool
enum_attribute::operator==(const attribute & other) const
{
  auto ea = dynamic_cast<const enum_attribute *>(&other);
  return ea && ea->kind() == kind();
}

int_attribute::~int_attribute() noexcept = default;

bool
int_attribute::operator==(const attribute & other) const
{
  auto ia = dynamic_cast<const int_attribute *>(&other);
  return ia && ia->kind() == kind() && ia->value() == value();
}

type_attribute::~type_attribute() noexcept = default;

bool
type_attribute::operator==(const attribute & other) const
{
  auto ta = dynamic_cast<const type_attribute *>(&other);
  return ta && ta->kind() == kind() && ta->type() == type();
}

attributeset::EnumAttributeRange
attributeset::EnumAttributes() const
{
  return EnumAttributes_.Items();
}

attributeset::IntAttributeRange
attributeset::IntAttributes() const
{
  return IntAttributes_.Items();
}

attributeset::TypeAttributeRange
attributeset::TypeAttributes() const
{
  return TypeAttributes_.Items();
}

attributeset::StringAttributeRange
attributeset::StringAttributes() const
{
  return StringAttributes_.Items();
}

}
