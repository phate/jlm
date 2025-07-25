/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/view.hpp>

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
  using namespace jlm::llvm::aa;

  // Arrange
  jlm::tests::LocalAliasAnalysisTest1 rvsdg;
  rvsdg.InitializeTest();
  const auto & outputs = rvsdg.GetOutputs();

  jlm::rvsdg::view(&rvsdg.graph().GetRootRegion(), stdout);

  LocalAliasAnalysis aa;

  // Assert

  // Distinct global variables do not alias
  Expect(aa, *outputs.Global, 4, *outputs.GlobalShort, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Global, 4, *outputs.Arr1, 4, AliasAnalysis::NoAlias);

  // An alloca never aliases any other memory allocating operation
  Expect(aa, *outputs.Alloca2, 4, *outputs.Alloca1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Array, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Arr1, 4, AliasAnalysis::NoAlias);

  // An alloca that has not "escaped" can not alias external pointers
  Expect(aa, *outputs.Alloca1, 4, *outputs.BytePtr, 4, AliasAnalysis::NoAlias);

  // An alloca that has "escaped" may alias external pointers
  Expect(aa, *outputs.Alloca2, 4, *outputs.BytePtr, 4, AliasAnalysis::MayAlias);

  // Distinct offsets can not alias, unless the access regions overlap
  Expect(aa, *outputs.Q, 8, *outputs.QPlus2, 8, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Q, 9, *outputs.QPlus2, 8, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Q, 8, *outputs.QPlus2, 16, AliasAnalysis::NoAlias);

  // Identical offsets are MustAlias
  Expect(aa, *outputs.Q, 4, *outputs.QAgain, 4, AliasAnalysis::MustAlias);

  // We know that arr1, arr2 and arr3 are close to the beginning of its storage instance
  // Meanwhile q is at least 16 bytes into the storage instance of *p
  Expect(aa, *outputs.Arr1, 4, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Arr2, 4, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Arr2, 8, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Arr2, 9, *outputs.Q, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Arr3, 4, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Arr3, 5, *outputs.Q, 4, AliasAnalysis::MayAlias);

  // We know that q is at least 16 bytes into its storage instance,
  // so it may not alias with storage instances that are 16 bytes or less
  Expect(aa, *outputs.Q, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);

  // A five byte operation can never target the 4 byte global variable
  Expect(aa, *outputs.BytePtr, 5, *outputs.Global, 4, AliasAnalysis::NoAlias);
  // A four byte operation can, however
  Expect(aa, *outputs.BytePtr, 4, *outputs.Global, 4, AliasAnalysis::MayAlias);
  // A five byte operation can target the 40 byte global array
  Expect(aa, *outputs.BytePtr, 5, *outputs.Array, 4, AliasAnalysis::MayAlias);

  // BytePtrPlus2 has an offset of at least 2, so can not alias with the first 2 bytes of alloca2
  Expect(aa, *outputs.Alloca2, 2, *outputs.BytePtrPlus2, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 3, *outputs.BytePtrPlus2, 2, AliasAnalysis::MayAlias);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/AliasAnalysisTests-TestLocalAliasAnalysis",
    TestLocalAliasAnalysis);
