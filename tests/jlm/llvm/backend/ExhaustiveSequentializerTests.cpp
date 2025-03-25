/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/backend/RegionSequentializer.hpp>
#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

static int
Test()
{
  jlm::tests::LoadTest1 test;
  test.InitializeTest();

  jlm::llvm::ExhaustiveRegionTreeSequentializer sequentializer(*test.GetLambdaNode().subregion());
  assert(sequentializer.HasMoreSequentializations());

  auto & regionSequentializer = sequentializer.GetSequentializer(*test.GetLambdaNode().subregion());
  while (sequentializer.HasMoreSequentializations())
  {
    auto sequentialization = regionSequentializer.GetSequentialization();
    for (const auto & node : sequentialization)
    {
      std::cout << node->GetOperation().debug_string() << std::endl;
    }
    std::cout << std::endl;

    sequentializer.ComputeNextSequentializations();
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/ExhaustiveSequentializerTests-Test", Test)
