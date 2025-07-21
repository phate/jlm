/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/memstate-conv.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/view.hpp>

static void
SplitConversion()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & importX = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & importY = jlm::tests::GraphImport::Create(rvsdg, memoryStateType, "y");

  auto structuralNode = jlm::tests::TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);
  auto & i0 = structuralNode->AddInputWithArguments(importX);

  auto entrySplitResults = LambdaEntryMemoryStateSplitOperation::Create(i0.Argument(0), 3);

  auto & o0 = structuralNode->AddOutputWithResults({ entrySplitResults[0] });
  auto & o1 = structuralNode->AddOutputWithResults({ entrySplitResults[1] });
  auto & o2 = structuralNode->AddOutputWithResults({ entrySplitResults[2] });

  auto splitResults = MemoryStateSplitOperation::Create(importY, 2);

  jlm::tests::GraphExport::Create(o0, "o0");
  jlm::tests::GraphExport::Create(o1, "o1");
  jlm::tests::GraphExport::Create(o2, "o2");

  jlm::tests::GraphExport::Create(*splitResults[0], "o3");
  jlm::tests::GraphExport::Create(*splitResults[1], "o4");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  MemoryStateSplitConversion::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 2);
  assert(structuralNode->subregion(0)->nnodes() == 1);

  // The memory state split conversion pass should have replaced the
  // LambdaEntryMemoryStateSplitOperation node with a ForkOperation node
  {
    assert(o0.nusers() == 1);
    auto [forkNode, forkOperation] =
        TryGetSimpleNodeAndOp<ForkOperation>(*i0.Argument(0).Users().begin());
    assert(forkNode && forkOperation);
  }

  // The memory state split conversion pass should have replaced the
  // MemoryStateSplitOperation node with a ForkOperation node
  {
    auto [forkNode, forkOperation] = TryGetSimpleNodeAndOp<ForkOperation>(*importY.Users().begin());
    assert(forkNode && forkOperation);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/MemoryStateSplitConversionTests-SplitConversion",
    SplitConversion)
