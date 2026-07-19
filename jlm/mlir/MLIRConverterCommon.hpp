/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_MLIRCONVERTERCOMMON_HPP
#define JLM_MLIR_MLIRCONVERTERCOMMON_HPP

#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/rvsdg/FunctionType.hpp>
#include <jlm/util/BijectiveMap.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <functional>
#include <unordered_map>

namespace jlm::mlir
{

/**
 * Get a bijective mapping between MLIR floating-point comparison predicates and JLM fpcmp values.
 *
 * @return A reference to the static mapping between MLIR CmpFPredicate and JLM fpcmp
 */
const util::BijectiveMap<::mlir::arith::CmpFPredicate, llvm::fpcmp> &
GetFpCmpPredicateMap();

/**
 * Compute a hash for a FunctionType based on its structure (num args, num results, and type
 * hashes). This allows comparing FunctionTypes even when they are different instances.
 */
struct FunctionTypeHash
{
  std::size_t
  operator()(const std::shared_ptr<const jlm::rvsdg::FunctionType> & type) const noexcept
  {
    if (!type)
      return 0;

    std::size_t hash = 0;

    // Combine number of arguments and results
    hash ^= std::hash<size_t>()(type->NumArguments()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<size_t>()(type->NumResults()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    // Combine hashes of all argument types
    for (const auto & argType : type->Arguments())
    {
      hash ^= argType->ComputeHash() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    // Combine hashes of all result types
    for (const auto & resType : type->Results())
    {
      hash ^= resType->ComputeHash() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    return hash;
  }
};

/**
 * Compare two FunctionTypes by their structure, not pointer identity.
 */
struct FunctionTypeEqual
{
  bool
  operator()(
      const std::shared_ptr<const jlm::rvsdg::FunctionType> & lhs,
      const std::shared_ptr<const jlm::rvsdg::FunctionType> & rhs) const noexcept
  {
    if (!lhs || !rhs)
      return lhs.get() == rhs.get();

    // Quick check: different number of args or results means not equal
    if (lhs->NumArguments() != rhs->NumArguments())
      return false;
    if (lhs->NumResults() != rhs->NumResults())
      return false;

    // Compare argument types using Type's operator==
    for (size_t i = 0; i < lhs->NumArguments(); ++i)
    {
      if (!(lhs->ArgumentType(i) == rhs->ArgumentType(i)))
        return false;
    }

    // Compare result types using Type's operator==
    for (size_t i = 0; i < lhs->NumResults(); ++i)
    {
      if (!(lhs->ResultType(i) == rhs->ResultType(i)))
        return false;
    }

    return true;
  }
};

} // namespace jlm::mlir

#endif // JLM_MLIR_MLIRCONVERTERCOMMON_HPP
