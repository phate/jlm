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
  const auto inputVar = structuralNode->AddInputWithArguments(importX);

  auto entrySplitResults = LambdaEntryMemoryStateSplitOperation::Create(*inputVar.argument[0], 3);

  const auto outputVar0 = structuralNode->AddOutputWithResults({ entrySplitResults[0] });
  const auto outputVar1 = structuralNode->AddOutputWithResults({ entrySplitResults[1] });
  const auto outputVar2 = structuralNode->AddOutputWithResults({ entrySplitResults[2] });

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
  assert(rvsdg.GetRootRegion().nnodes() == 2);
  assert(structuralNode->subregion(0)->nnodes() == 1);

  // The memory state split conversion pass should have replaced the
  // LambdaEntryMemoryStateSplitOperation node with a ForkOperation node
  {
    assert(outputVar0.output->nusers() == 1);
    auto [forkNode, forkOperation] =
        TryGetSimpleNodeAndOp<ForkOperation>(*inputVar.argument[0]->Users().begin());
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
