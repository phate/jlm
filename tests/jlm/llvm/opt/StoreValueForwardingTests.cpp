/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/StoreValueForwarding.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/util/Statistics.hpp>

static void
RunStoreValueForwarding(jlm::llvm::LlvmRvsdgModule & rvsdgModule)
{
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::StoreValueForwarding storeValueForwarding;
  storeValueForwarding.Run(rvsdgModule, statisticsCollector);
}

TEST(StoreValueForwardingTests, BasicTest)
{
  using namespace jlm::llvm;

  // Arrange
  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  // Act - This should not crash or throw exceptions
  RunStoreValueForwarding(rvsdgModule);

  // Assert - Basic structure should remain intact
  EXPECT_EQ(graph.GetRootRegion().narguments(), 0u);
}
