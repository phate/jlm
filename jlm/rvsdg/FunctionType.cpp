/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <sstream>

#include <jlm/rvsdg/FunctionType.hpp>
#include <jlm/util/Hash.hpp>

namespace jlm::rvsdg
{

FunctionType::~FunctionType() noexcept = default;

FunctionType::FunctionType(
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> argumentTypes,
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> resultTypes)
    : jlm::rvsdg::ValueType(),
      ArgumentTypes_(std::move(argumentTypes)),
      ResultTypes_(std::move(resultTypes))
{}

std::string
FunctionType::debug_string() const
{
  std::stringstream ss;
  ss << "fun (";
  bool first = true;
  for (const auto & argtype : ArgumentTypes_)
  {
    if (!first)
    {
      ss << ", ";
    }
    first = false;
    ss << argtype->debug_string();
  }

  ss << ") -> (";
  first = true;
  for (const auto & restype : ResultTypes_)
  {
    if (!first)
    {
      ss << ", ";
    }
    first = false;
    ss << restype->debug_string();
  }
  ss << ")";
  return ss.str();
}

bool
FunctionType::operator==(const jlm::rvsdg::Type & other) const noexcept
{
  if (auto fn = dynamic_cast<const FunctionType *>(&other))
  {
    if (ArgumentTypes_.size() != fn->ArgumentTypes_.size()
        || ResultTypes_.size() != fn->ResultTypes_.size())
    {
      return false;
    }
    for (std::size_t n = 0; n < ArgumentTypes_.size(); ++n)
    {
      if (*ArgumentTypes_[n] != *fn->ArgumentTypes_[n])
      {
        return false;
      }
    }
    for (std::size_t n = 0; n < ResultTypes_.size(); ++n)
    {
      if (*ResultTypes_[n] != *fn->ResultTypes_[n])
      {
        return false;
      }
    }
    return true;
  }
  else
  {
    return false;
  }
}

std::size_t
FunctionType::ComputeHash() const noexcept
{
  std::size_t seed = typeid(FunctionType).hash_code();

  util::CombineHashesWithSeed(seed, ArgumentTypes_.size());
  for (auto argumentType : ArgumentTypes_)
  {
    util::CombineHashesWithSeed(seed, argumentType->ComputeHash());
  }

  util::CombineHashesWithSeed(seed, ResultTypes_.size());
  for (auto resultType : ResultTypes_)
  {
    util::CombineHashesWithSeed(seed, resultType->ComputeHash());
  }

  return seed;
}

std::shared_ptr<const FunctionType>
FunctionType::Create(
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> argumentTypes,
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> resultTypes)
{
  return std::make_shared<FunctionType>(std::move(argumentTypes), std::move(resultTypes));
}

}
