/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/RegionSequentializer.hpp>
#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#if 0
static int
Test()
{
  jlm::tests::LoadTest1 test;
  test.InitializeTest();

  jlm::llvm::ExhaustiveSingleRegionSequentializer sequentializer(*test.GetLambdaNode().subregion());
  assert(sequentializer.HasMoreSequentializations());

  auto sequentializationMap = sequentializer.ComputeNextSequentialization();
  assert(sequentializationMap.has_value());
  assert(!sequentializer.HasMoreSequentializations());

  sequentializationMap = sequentializer.ComputeNextSequentialization();
  assert(!sequentializationMap.has_value());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/ExhaustiveSequentializerTests-Test", Test)
#endif
