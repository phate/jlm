/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/opt/alias-analyses/LocalAliasAnalysis.hpp>
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

  // q is at least 8 bytes into the storage instance of *p
  // so it can not alias with the first 8 bytes of array
  Expect(aa, *outputs.Array, 8, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Array, 9, *outputs.Q, 4, AliasAnalysis::MayAlias);
  // We know that arr1, arr2 and arr3 are 4, 8 and 12 bytes into array
  Expect(aa, *outputs.Arr1, 4, *outputs.Q, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Arr1, 5, *outputs.Q, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Arr2, 4, *outputs.Q, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Arr3, 4, *outputs.Q, 4, AliasAnalysis::MayAlias);

  // An unknown offset into array can only alias with array, at all offsets
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Array, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Arr1, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Arr2, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Arr3, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);
  // Q may be a pointer into array, so it is also "MayAlias"
  Expect(aa, *outputs.ArrUnknown, 4, *outputs.Q, 4, AliasAnalysis::MayAlias);

  // We know that q is at least 16 bytes into its storage instance,
  // so it may not alias with storage instances that are 16 bytes or less
  Expect(aa, *outputs.Q, 4, *outputs.Global, 4, AliasAnalysis::NoAlias);

  // A five byte operation can never target the 4 byte global variable
  Expect(aa, *outputs.BytePtr, 5, *outputs.Global, 4, AliasAnalysis::NoAlias);
  // A four byte operation can, however
  Expect(aa, *outputs.BytePtr, 4, *outputs.Global, 4, AliasAnalysis::MayAlias);
  // Even a 40 byte operation can target the 40 byte global array
  Expect(aa, *outputs.BytePtr, 40, *outputs.Array, 4, AliasAnalysis::MayAlias);
  // The 40 byte operation may overlap with 4 bytes at any offset within the Array
  Expect(aa, *outputs.BytePtr, 40, *outputs.Arr3, 4, AliasAnalysis::MayAlias);

  // BytePtrPlus2 has an offset of at least 2, so can not alias with the first 2 bytes of anything
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Alloca2, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Alloca2, 3, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Array, 2, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Array, 3, AliasAnalysis::MayAlias);
  // Arr1 is already 4 bytes into array, so BytePtrPlus2 can alias with it
  Expect(aa, *outputs.BytePtrPlus2, 2, *outputs.Arr1, 2, AliasAnalysis::MayAlias);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/AliasAnalysisTests-TestLocalAliasAnalysis",
    TestLocalAliasAnalysis);

void
TestLocalAliasAnalysisMultipleOrigins()
{
  using namespace jlm::llvm::aa;

  // Arrange
  jlm::tests::LocalAliasAnalysisTest2 rvsdg;
  rvsdg.InitializeTest();
  const auto & outputs = rvsdg.GetOutputs();

  jlm::rvsdg::view(&rvsdg.graph().GetRootRegion(), stdout);

  LocalAliasAnalysis aa;

  // Assert

  // First check that none of the allocas have been mixed up with unknown pointers
  Expect(aa, *outputs.Alloca1, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca2, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3Plus1, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);

  // Check that allocaUnknown may alias only alloca1 or alloca2
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca1, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca2, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca3, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Alloca3Plus1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);

  // If performing an 8 byte operation, it may only alias alloca2, becoming a must alias
  Expect(aa, *outputs.AllocaUnknown, 8, *outputs.Alloca1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 8, *outputs.Alloca2, 4, AliasAnalysis::MustAlias);
  // Performing a 9 byte operation is neither legal for alloca1 nor alloca2
  Expect(aa, *outputs.AllocaUnknown, 9, *outputs.Alloca2, 4, AliasAnalysis::NoAlias);

  // Adding a 4 byte offset forces all operations to be on alloca2
  Expect(aa, *outputs.AllocaUnknownPlus1, 1, *outputs.Alloca1, 1, AliasAnalysis::NoAlias);
  // We also know that we are 4 bytes into alloca2
  Expect(aa, *outputs.AllocaUnknownPlus1, 4, *outputs.Alloca2, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknownPlus1, 4, *outputs.Alloca2, 5, AliasAnalysis::MayAlias);
  // Performing a 5 byte operation is neither legal for alloca1 nor alloca2
  Expect(aa, *outputs.AllocaUnknownPlus1, 5, *outputs.Alloca2, 8, AliasAnalysis::NoAlias);

  // Check that the offset of allocaUnknown is correctly calculated (4 bytes)
  Expect(aa, *outputs.AllocaUnknown, 4, *outputs.AllocaUnknownPlus1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.AllocaUnknown, 5, *outputs.AllocaUnknownPlus1, 4, AliasAnalysis::MayAlias);

  // Check that the pointer with an unknown offset into alloca3 does not alias anything else
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca1, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca2, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.AllocaUnknown, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Ptr, 4, AliasAnalysis::NoAlias);

  // It may alias alloca3 and alloca3 + 1
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca3, 4, AliasAnalysis::MayAlias);
  Expect(aa, *outputs.Alloca3UnknownOffset, 4, *outputs.Alloca3Plus1, 4, AliasAnalysis::MayAlias);

  // If performing an 8 byte operation, we know that we are at the start of alloca3
  Expect(aa, *outputs.Alloca3UnknownOffset, 8, *outputs.Alloca3, 4, AliasAnalysis::MustAlias);
  // We still overlap with the second half of alloca3
  Expect(aa, *outputs.Alloca3UnknownOffset, 8, *outputs.Alloca3Plus1, 4, AliasAnalysis::MayAlias);

  // The select with duplicate operands should be a single origin: alloca3
  Expect(aa, *outputs.Alloca3KnownOffset, 4, *outputs.Alloca3, 4, AliasAnalysis::NoAlias);
  Expect(aa, *outputs.Alloca3KnownOffset, 4, *outputs.Alloca3Plus1, 4, AliasAnalysis::MustAlias);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/AliasAnalysisTests-TestLocalAliasAnalysisMultipleOrigins",
    TestLocalAliasAnalysisMultipleOrigins);
