/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/TestType.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Hash.hpp>
#include <jlm/util/strfmt.hpp>

#include <unordered_map>

namespace jlm::rvsdg
{

static std::string
ToString(const TypeKind kind)
{
  switch (kind)
  {
  case TypeKind::Value:
    return "Value";
  case TypeKind::State:
    return "State";
  default:
    throw std::logic_error("Unhandled type kind!");
  }
}

TestType::~TestType() noexcept = default;

std::string
TestType::debug_string() const
{
  return util::strfmt("TestType[", ToString(kind_), "]");
}

bool
TestType::operator==(const Type & other) const noexcept
{
  const auto testType = dynamic_cast<const TestType *>(&other);
  return testType && kind_ == testType->kind_;
}

std::size_t
TestType::ComputeHash() const noexcept
{
  const auto typeHash = typeid(TestType).hash_code();
  const auto numAlternativesHash = std::hash<TypeKind>()(kind_);
  return util::CombineHashes(typeHash, numAlternativesHash);
}

TypeKind
TestType::Kind() const noexcept
{
  return kind_;
}

std::shared_ptr<const TestType>
TestType::createStateType()
{
  static const TestType stateType(TypeKind::State);
  return std::shared_ptr<const TestType>(std::shared_ptr<void>(), &stateType);
}

std::shared_ptr<const TestType>
TestType::createValueType()
{
  static const TestType valueType(TypeKind::Value);
  return std::shared_ptr<const TestType>(std::shared_ptr<void>(), &valueType);
}

}
