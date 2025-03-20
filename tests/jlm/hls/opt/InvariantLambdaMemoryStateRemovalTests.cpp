/*
 * Copyright 2025 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/opt/InvariantLambdaMemoryStateRemoval.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/view.hpp>

static int
TestEliminateSplitAndMergeNodes()
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

  auto functionArguments = lambda->GetFunctionArguments();

  // LambdaEntryMemoryStateSplit node
  auto memoryStateSplit = LambdaEntryMemoryStateSplitOperation::Create(*functionArguments[1], 2);

  // Load node
  auto loadOutput = LoadNonVolatileNode::Create(
      functionArguments[0],
      { memoryStateSplit[0] },
      PointerType::Create(),
      32);

  // LambdaExitMemoryStateMerge node
  std::vector<jlm::rvsdg::output *> outputs;
  auto & memoryStateMerge = LambdaExitMemoryStateMergeOperation::Create(
      *lambda->subregion(),
      { loadOutput[1], memoryStateSplit[1] });

  auto lambdaOutput = lambda->finalize({ &memoryStateMerge });
  GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  // This pass should remove the Lambda[Entry/Exit]MemoryState[Split/Merge] nodes
  jlm::util::StatisticsCollector collector;
  InvariantLambdaMemoryStateRemoval::CreateAndRun(*rvsdgModule, collector);
  // Assert
  auto * node =
      jlm::rvsdg::output::GetNode(*rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(lambdaSubregion->narguments() == 2);
  assert(lambdaSubregion->nresults() == 1);
  assert(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));
  auto loadNode = jlm::rvsdg::output::GetNode(*lambdaSubregion->result(0)->origin());
  assert(is<LoadNonVolatileOperation>(loadNode->GetOperation()));
  jlm::util::AssertedCast<jlm::rvsdg::RegionArgument>(loadNode->input(1)->origin());

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/hls/opt/InvariantLambdaMemoryStateRemovalTests-EliminateSplitAndMergeNodes",
    TestEliminateSplitAndMergeNodes)

static int
TestInvariantMemoryState()
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

  auto functionArguments = lambda->GetFunctionArguments();

  // LambdaEntryMemoryStateSplit node
  auto memoryStateSplit = LambdaEntryMemoryStateSplitOperation::Create(*functionArguments[1], 3);

  // Load node
  auto loadOutput1 = LoadNonVolatileNode::Create(
      functionArguments[0],
      { memoryStateSplit[0] },
      PointerType::Create(),
      32);

  // Load node
  auto loadOutput2 = LoadNonVolatileNode::Create(
      functionArguments[0],
      { memoryStateSplit[2] },
      PointerType::Create(),
      32);

  // LambdaExitMemoryStateMerge node
  std::vector<jlm::rvsdg::output *> outputs;
  auto & memoryStateMerge = LambdaExitMemoryStateMergeOperation::Create(
      *lambda->subregion(),
      { loadOutput1[1], memoryStateSplit[1], loadOutput2[1] });

  auto lambdaOutput = lambda->finalize({ &memoryStateMerge });
  GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  // This pass should remove the Lambda[Entry/Exit]MemoryState[Split/Merge] nodes
  jlm::util::StatisticsCollector collector;
  InvariantLambdaMemoryStateRemoval memStateRemoval;
  memStateRemoval.Run(*rvsdgModule, collector);
  // Assert
  auto * node =
      jlm::rvsdg::output::GetNode(*rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(lambdaSubregion->narguments() == 2);
  assert(lambdaSubregion->nresults() == 1);
  assert(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));
  // Since there is more than one invariant memory state edge, the MemoryStateMerge node should
  // still exists
  node = jlm::rvsdg::output::GetNode(*lambdaSubregion->result(0)->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(node->GetOperation()));
  assert(node->ninputs() == 2);
  // Need to pass a load node to reach the MemoryStateSplit node
  node = jlm::rvsdg::output::GetNode(*node->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(node->GetOperation()));
  // Check that the MemoryStateSplit node is still present
  node = jlm::rvsdg::output::GetNode(*node->input(1)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(node->GetOperation()));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/hls/opt/InvariantLambdaMemoryStateRemovalTests-InvariantMemoryState",
    TestInvariantMemoryState)
