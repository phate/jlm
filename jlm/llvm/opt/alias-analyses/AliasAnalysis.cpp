/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm::aa
{

AliasAnalysis::AliasAnalysis() = default;

AliasAnalysis::~AliasAnalysis() noexcept = default;

/**
 * Checks if two alias query responses are compatible with each other.
 * NoAlias is incompatible with MustAlias, and vice versa.
 * MayAlias is compatible with all responses.
 *
 * @param a first alias query response to check
 * @param b second alias query response to check
 * @return true if the responses are compatible, false otherwise
 */
static bool
AreAliasResponsesCompatible(
    AliasAnalysis::AliasQueryResponse a,
    AliasAnalysis::AliasQueryResponse b)
{
  if (a == AliasAnalysis::NoAlias)
    return b != AliasAnalysis::MustAlias;
  if (a == AliasAnalysis::MayAlias)
    return true;
  if (a == AliasAnalysis::MustAlias)
    return b != AliasAnalysis::NoAlias;
  JLM_UNREACHABLE("Unknown alias response");
}

ChainedAliasAnalysis::ChainedAliasAnalysis(AliasAnalysis & first, AliasAnalysis & second)
    : First_(first),
      Second_(second)
{}

ChainedAliasAnalysis::~ChainedAliasAnalysis() noexcept = default;

AliasAnalysis::AliasQueryResponse
ChainedAliasAnalysis::Query(
    const rvsdg::Output & p1,
    size_t s1,
    const rvsdg::Output & p2,
    size_t s2)
{
  const auto firstResponse = First_.Query(p1, s1, p2, s2);
  if (firstResponse == MayAlias)
    return Second_.Query(p1, s1, p2, s2);

  // When building with asserts, always query the second analysis and double check
  JLM_ASSERT(AreAliasResponsesCompatible(firstResponse, Second_.Query(p1, s1, p2, s2)));
  return firstResponse;
}

std::string
ChainedAliasAnalysis::ToString() const
{
  return util::strfmt("ChainedAA(", First_.ToString(), ",", Second_.ToString(), ")");
}

bool
IsPointerCompatible(const rvsdg::Output & value)
{
  return IsOrContains<PointerType>(*value.Type());
}

}
