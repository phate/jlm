/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/hls/backend/rvsdg2rhls/UnusedStateRemoval.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestGamma()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();

  auto rvsdgModule = RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto p = &jlm::tests::GraphImport::Create(rvsdg, jlm::rvsdg::ControlType::Create(2), "p");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::tests::GraphImport::Create(rvsdg, valueType, "z");

  auto gammaNode = jlm::rvsdg::GammaNode::create(p, 2);

  auto gammaInput1 = gammaNode->AddEntryVar(x);
  auto gammaInput2 = gammaNode->AddEntryVar(y);
  auto gammaInput3 = gammaNode->AddEntryVar(z);
  auto gammaInput4 = gammaNode->AddEntryVar(x);
  auto gammaInput5 = gammaNode->AddEntryVar(x);
  auto gammaInput6 = gammaNode->AddEntryVar(x);
  auto gammaInput7 = gammaNode->AddEntryVar(x);

  auto gammaOutput1 = gammaNode->AddExitVar(gammaInput1.branchArgument);
  auto gammaOutput2 =
      gammaNode->AddExitVar({ gammaInput2.branchArgument[0], gammaInput3.branchArgument[1] });
  auto gammaOutput3 =
      gammaNode->AddExitVar({ gammaInput4.branchArgument[0], gammaInput5.branchArgument[1] });
  auto gammaOutput4 =
      gammaNode->AddExitVar({ gammaInput6.branchArgument[0], gammaInput6.branchArgument[1] });
  auto gammaOutput5 =
      gammaNode->AddExitVar({ gammaInput6.branchArgument[0], gammaInput7.branchArgument[1] });

  GraphExport::Create(*gammaOutput1.output, "");
  GraphExport::Create(*gammaOutput2.output, "");
  GraphExport::Create(*gammaOutput3.output, "");
  GraphExport::Create(*gammaOutput4.output, "");
  GraphExport::Create(*gammaOutput5.output, "");

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  assert(gammaNode->ninputs() == 7);  // gammaInput1 was removed
  assert(gammaNode->noutputs() == 4); // gammaOutput1 was removed
  assert(gammaInput2.input->index() == 1);
  assert(gammaOutput2.output->index() == 0);
  // FIXME: The transformation is way too conservative here. The only input and output it removes
  // are gammaInput1 and gammaOutput1, respectively. However, it could also remove gammaOutput3,
  // gammaOutput4, and gammaOutput5 as they are all invariant. This in turn would also render some
  // more inputs dead.
}

static void
TestTheta()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::ControlType::Create(2), valueType, valueType, valueType },
      { valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();
  auto p = &jlm::tests::GraphImport::Create(rvsdg, jlm::rvsdg::ControlType::Create(2), "p");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::tests::GraphImport::Create(rvsdg, valueType, "z");

  auto thetaNode = jlm::rvsdg::ThetaNode::create(&rvsdg.GetRootRegion());

  auto thetaOutput0 = thetaNode->AddLoopVar(p);
  auto thetaOutput1 = thetaNode->AddLoopVar(x);
  auto thetaOutput2 = thetaNode->AddLoopVar(y);
  auto thetaOutput3 = thetaNode->AddLoopVar(z);

  thetaOutput2.post->divert_to(thetaOutput3.pre);
  thetaOutput3.post->divert_to(thetaOutput2.pre);
  thetaNode->set_predicate(thetaOutput0.pre);

  auto result =
      jlm::rvsdg::CreateOpNode<jlm::tests::test_op>(
          { thetaOutput0.output, thetaOutput1.output, thetaOutput2.output, thetaOutput3.output },
          std::vector<std::shared_ptr<const Type>>{ ControlType::Create(2),
                                                    valueType,
                                                    valueType,
                                                    valueType },
          std::vector<std::shared_ptr<const Type>>{ valueType })
          .output(0);

  jlm::tests::GraphExport::Create(*result, "f");

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  // This assert is only here so that we do not forget this test when we refactor the code
  assert(thetaNode->ninputs() == 1);

  // FIXME: This transformation is broken for theta nodes. For the setup above, it
  // removes all inputs/outputs, except the predicate. However, the only
  // input and output it should remove are input 1 and output 0, respectively.
}

static void
TestLambda()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { valueType, valueType },
      { valueType, valueType, valueType, valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
  auto argument0 = lambdaNode->GetFunctionArguments()[0];
  auto argument1 = lambdaNode->GetFunctionArguments()[1];
  auto argument2 = lambdaNode->AddContextVar(*x).inner;
  auto argument3 = lambdaNode->AddContextVar(*x).inner;

  auto result1 = jlm::rvsdg::CreateOpNode<jlm::tests::test_op>(
                     { argument1 },
                     std::vector<std::shared_ptr<const Type>>{ valueType },
                     std::vector<std::shared_ptr<const Type>>{ valueType })
                     .output(0);

  auto result3 = jlm::rvsdg::CreateOpNode<jlm::tests::test_op>(
                     { argument3 },
                     std::vector<std::shared_ptr<const Type>>{ valueType },
                     std::vector<std::shared_ptr<const Type>>{ valueType })
                     .output(0);

  auto lambdaOutput = lambdaNode->finalize({ argument0, result1, argument2, result3 });

  jlm::tests::GraphExport::Create(*lambdaOutput, "f");

  // Act
  jlm::hls::RemoveUnusedStates(*rvsdgModule);

  // Assert
  assert(rvsdg.GetRootRegion().nnodes() == 1);
  auto & newLambdaNode =
      dynamic_cast<const jlm::rvsdg::LambdaNode &>(*rvsdg.GetRootRegion().Nodes().begin());
  assert(newLambdaNode.ninputs() == 2);
  assert(newLambdaNode.subregion()->narguments() == 3);
  assert(newLambdaNode.subregion()->nresults() == 2);
  // FIXME For lambdas, the transformation has the following issues:
  // 1. It works only for lambda nodes in the root region. It throws an assert for all other lambdas
  // 2. It does not check whether the lambda is only exported. Removing passthrough values works
  // only for lambda nodes that do not have any calls.
  // 3. It removes all pass through values, regardless of whether they are value or state types.
  // Removing value types does change the signature of a lambda node.
  // 4. It does not remove the arguments and inputs of context variables that are just passed
  // through. It only renders them dead.
  //
  // There might be more issues.
}

static int
TestUnusedStateRemoval()
{
  TestGamma();
  TestTheta();
  TestLambda();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/UnusedStateRemovalTests", TestUnusedStateRemoval)

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
  auto loadOutput = LoadNonVolatileOperation::Create(
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
  auto * node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  assert(lambdaSubregion->nresults() == 1);
  assert(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/UnusedStateRemovalTests-UsedMemoryState",
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
  auto loadOutput = LoadNonVolatileOperation::Create(
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
  auto * node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::AssertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(lambdaSubregion->narguments() == 2);
  assert(lambdaSubregion->nresults() == 1);
  assert(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/UnusedStateRemovalTests-UnusedMemoryState",
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
  auto loadOutput = LoadNonVolatileOperation::Create(
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
  auto * node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
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
    "jlm/hls/backend/rvsdg2rhls/UnusedStateRemovalTests-InvariantMemoryState",
    TestInvariantMemoryState)
