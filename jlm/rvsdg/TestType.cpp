/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/TestType.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Hash.hpp>

#include <unordered_map>

namespace jlm::rvsdg
{

TestType::~TestType() noexcept = default;

std::string
TestType::debug_string() const
{
  return "TestType";
}

bool
TestType::operator==(const Type & other) const noexcept
{
  return dynamic_cast<const TestType *>(&other) != nullptr;
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
TestType::Create(const TypeKind kind)
{
  static std::unordered_map<TypeKind, const TestType> instances(
      { { TypeKind::Value, TestType(TypeKind::Value) },
        { TypeKind::State, TestType(TypeKind::State) } });

  const auto instance = instances.find(kind);
  JLM_ASSERT(instance != instances.end());

  return std::shared_ptr<const TestType>(std::shared_ptr<void>(), &instance->second);
}
}
