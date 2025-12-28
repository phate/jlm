/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/memstate-conv.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
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

  auto & importX = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "x");
  auto & importY = jlm::rvsdg::GraphImport::Create(rvsdg, memoryStateType, "y");

  auto structuralNode = TestStructuralNode::create(&rvsdg.GetRootRegion(), 1);
  const auto inputVar = structuralNode->addInputWithArguments(importX);

  auto & entrySplitNode =
      LambdaEntryMemoryStateSplitOperation::CreateNode(*inputVar.argument[0], { 0, 1, 2 });

  const auto outputVar0 = structuralNode->addOutputWithResults({ entrySplitNode.output(0) });
  const auto outputVar1 = structuralNode->addOutputWithResults({ entrySplitNode.output(1) });
  const auto outputVar2 = structuralNode->addOutputWithResults({ entrySplitNode.output(2) });

  auto splitResults = MemoryStateSplitOperation::Create(importY, 2);

  jlm::rvsdg::GraphExport::Create(*outputVar0.output, "o0");
  jlm::rvsdg::GraphExport::Create(*outputVar1.output, "o1");
  jlm::rvsdg::GraphExport::Create(*outputVar2.output, "o2");

  jlm::rvsdg::GraphExport::Create(*splitResults[0], "o3");
  jlm::rvsdg::GraphExport::Create(*splitResults[1], "o4");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  MemoryStateSplitConversion::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  assert(rvsdg.GetRootRegion().numNodes() == 2);
  assert(structuralNode->subregion(0)->numNodes() == 1);

  // The memory state split conversion pass should have replaced the
  // LambdaEntryMemoryStateSplitOperation node with a ForkOperation node
  {
    assert(outputVar0.output->nusers() == 1);
    assert(IsOwnerNodeOperation<ForkOperation>(*inputVar.argument[0]->Users().begin()));
  }

  // The memory state split conversion pass should have replaced the
  // MemoryStateSplitOperation node with a ForkOperation node
  {
    assert(IsOwnerNodeOperation<ForkOperation>(*importY.Users().begin()));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/MemoryStateSplitConversionTests-SplitConversion",
    SplitConversion)
