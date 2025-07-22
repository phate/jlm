/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>

#include <cassert>

static void
Expect(
    jlm::llvm::aa::AliasAnalysis & aa,
    const jlm::rvsdg::Output & p1,
    size_t s1,
    const jlm::rvsdg::Output & p2,
    size_t s2,
    jlm::llvm::aa::AliasAnalysis::AliasQueryResponse expected)
{
  const auto actual = aa.Query(p1, s1, p2, s2);
  assert(actual == expected);

  // An alias analysis query should always be symmetrical, so check the opposite as well
  const auto mirror = aa.Query(p2, s2, p1, s1);
  assert(mirror == expected);
}

void
TestLocalAliasAnalysis()
{

}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/AliasAnalysisTests-TestLocalAliasAnalysis",
    TestLocalAliasAnalysis);
