/*
 * Copyright 2025 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/remove-unused-state.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static int
TestUsedMemoryState()
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

  // Load node
  auto functionArguments = lambda->GetFunctionArguments();
  auto loadOutput = LoadNonVolatileNode::Create(
      functionArguments[0],
      { functionArguments[1] },
      PointerType::Create(),
      32);

  auto lambdaOutput = lambda->finalize({ loadOutput[1] });
  GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  RemoveInvariantLambdaStateEdges(*rvsdgModule);
  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto * node =
      jlm::rvsdg::output::GetNode(*rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  assert(lambdaSubregion->nresults() == 1);
  assert(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/InvariantLambdaMemoryStateRemovalTests-UsedMemoryState",
    TestUsedMemoryState)

static int
TestUnusedMemoryState()
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create(), MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

  // Load node
  auto functionArguments = lambda->GetFunctionArguments();
  auto loadOutput = LoadNonVolatileNode::Create(
      functionArguments[0],
      { functionArguments[1] },
      PointerType::Create(),
      32);

  auto lambdaOutput = lambda->finalize({ loadOutput[1], functionArguments[2] });
  GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  RemoveInvariantLambdaStateEdges(*rvsdgModule);
  // Assert
  auto * node =
      jlm::rvsdg::output::GetNode(*rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(lambdaSubregion->narguments() == 2);
  assert(lambdaSubregion->nresults() == 1);
  assert(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/InvariantLambdaMemoryStateRemovalTests-UnusedMemoryState",
    TestUnusedMemoryState)

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
  // This pass should have no effect on the graph
  RemoveInvariantLambdaStateEdges(*rvsdgModule);
  // Assert
  auto * node =
      jlm::rvsdg::output::GetNode(*rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(lambdaSubregion->narguments() == 2);
  assert(lambdaSubregion->nresults() == 1);
  assert(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));
  assert(
      jlm::rvsdg::Region::Contains<LambdaEntryMemoryStateSplitOperation>(*lambdaSubregion, true));
  assert(jlm::rvsdg::Region::Contains<LambdaExitMemoryStateMergeOperation>(*lambdaSubregion, true));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/InvariantLambdaMemoryStateRemovalTests-InvariantMemoryState",
    TestInvariantMemoryState)
