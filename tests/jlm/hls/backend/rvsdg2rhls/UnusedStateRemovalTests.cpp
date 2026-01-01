/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rvsdg2rhls/UnusedStateRemoval.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(UnusedStateRemovalTests, TestGamma)
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto p = &jlm::rvsdg::GraphImport::Create(rvsdg, jlm::rvsdg::ControlType::Create(2), "p");
  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "z");

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

  jlm::rvsdg::GraphExport::Create(*gammaOutput1.output, "");
  jlm::rvsdg::GraphExport::Create(*gammaOutput2.output, "");
  jlm::rvsdg::GraphExport::Create(*gammaOutput3.output, "");
  jlm::rvsdg::GraphExport::Create(*gammaOutput4.output, "");
  jlm::rvsdg::GraphExport::Create(*gammaOutput5.output, "");

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::hls::UnusedStateRemoval::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  EXPECT_EQ(gammaNode->ninputs(), 7);  // gammaInput1 was removed
  EXPECT_EQ(gammaNode->noutputs(), 4); // gammaOutput1 was removed
  EXPECT_EQ(gammaInput2.input->index(), 1);
  EXPECT_EQ(gammaOutput2.output->index(), 0);
  // FIXME: The transformation is way too conservative here. The only input and output it removes
  // are gammaInput1 and gammaOutput1, respectively. However, it could also remove gammaOutput3,
  // gammaOutput4, and gammaOutput5 as they are all invariant. This in turn would also render some
  // more inputs dead.
}

TEST(UnusedStateRemovalTests, TestTheta)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = FunctionType::Create(
      { ControlType::Create(2), valueType, valueType, valueType },
      { valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto importP = &jlm::rvsdg::GraphImport::Create(rvsdg, ControlType::Create(2), "p");
  auto importX = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");
  auto importY = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "y");
  auto importZ = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "z");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto loopVarP = thetaNode->AddLoopVar(importP);
  auto loopVarX = thetaNode->AddLoopVar(importX);
  auto loopVarY = thetaNode->AddLoopVar(importY);
  auto loopVarZ = thetaNode->AddLoopVar(importZ);

  loopVarY.post->divert_to(loopVarZ.pre);
  loopVarZ.post->divert_to(loopVarY.pre);
  thetaNode->set_predicate(loopVarP.pre);

  auto & exportP = GraphExport::Create(*loopVarP.output, "p");
  auto & exportX = GraphExport::Create(*loopVarX.output, "x");
  auto & exportY = GraphExport::Create(*loopVarY.output, "y");
  auto & exportZ = GraphExport::Create(*loopVarZ.output, "z");

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::hls::UnusedStateRemoval::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  EXPECT_EQ(thetaNode->ninputs(), 3);
  EXPECT_EQ(thetaNode->noutputs(), 3);

  EXPECT_EQ(TryGetOwnerNode<ThetaNode>(*exportP.origin()), thetaNode);
  EXPECT_EQ(exportX.origin(), importX);
  EXPECT_EQ(TryGetOwnerNode<ThetaNode>(*exportY.origin()), thetaNode);
  EXPECT_EQ(TryGetOwnerNode<ThetaNode>(*exportZ.origin()), thetaNode);
}

TEST(UnusedStateRemovalTests, TestLambda)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { valueType, valueType },
      { valueType, valueType, valueType, valueType });

  auto rvsdgModule = jlm::llvm::RvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule->Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto argument0 = lambdaNode->GetFunctionArguments()[0];
  auto argument1 = lambdaNode->GetFunctionArguments()[1];
  auto argument2 = lambdaNode->AddContextVar(*x).inner;
  auto argument3 = lambdaNode->AddContextVar(*x).inner;

  auto result1 = jlm::rvsdg::CreateOpNode<TestOperation>(
                     { argument1 },
                     std::vector<std::shared_ptr<const Type>>{ valueType },
                     std::vector<std::shared_ptr<const Type>>{ valueType })
                     .output(0);

  auto result3 = jlm::rvsdg::CreateOpNode<TestOperation>(
                     { argument3 },
                     std::vector<std::shared_ptr<const Type>>{ valueType },
                     std::vector<std::shared_ptr<const Type>>{ valueType })
                     .output(0);

  auto lambdaOutput = lambdaNode->finalize({ argument0, result1, argument2, result3 });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::hls::UnusedStateRemoval::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 1);
  auto & newLambdaNode =
      dynamic_cast<const jlm::rvsdg::LambdaNode &>(*rvsdg.GetRootRegion().Nodes().begin());
  EXPECT_EQ(newLambdaNode.ninputs(), 2);
  EXPECT_EQ(newLambdaNode.subregion()->narguments(), 3);
  EXPECT_EQ(newLambdaNode.subregion()->nresults(), 2);
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

TEST(UnusedStateRemovalTests, TestUsedMemoryState)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Load node
  auto functionArguments = lambda->GetFunctionArguments();
  auto loadOutput = LoadNonVolatileOperation::Create(
      functionArguments[0],
      { functionArguments[1] },
      PointerType::Create(),
      32);

  auto lambdaOutput = lambda->finalize({ loadOutput[1] });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  UnusedStateRemoval::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto * node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  EXPECT_EQ(lambdaSubregion->nresults(), 1);
  EXPECT_TRUE(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));
}

TEST(UnusedStateRemovalTests, TestUnusedMemoryState)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create(), MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Load node
  auto functionArguments = lambda->GetFunctionArguments();
  auto loadOutput = LoadNonVolatileOperation::Create(
      functionArguments[0],
      { functionArguments[1] },
      PointerType::Create(),
      32);

  auto lambdaOutput = lambda->finalize({ loadOutput[1], functionArguments[2] });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  UnusedStateRemoval::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  auto * node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_EQ(lambdaSubregion->narguments(), 2);
  EXPECT_EQ(lambdaSubregion->nresults(), 1);
  EXPECT_TRUE(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));
}

TEST(UnusedStateRemovalTests, TestInvariantMemoryState)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  auto functionArguments = lambda->GetFunctionArguments();

  // LambdaEntryMemoryStateSplit node
  auto & memoryStateSplitNode =
      LambdaEntryMemoryStateSplitOperation::CreateNode(*functionArguments[1], { 0, 1 });

  // Load node
  auto loadOutput = LoadNonVolatileOperation::Create(
      functionArguments[0],
      { memoryStateSplitNode.output(0) },
      PointerType::Create(),
      32);

  // LambdaExitMemoryStateMerge node
  std::vector<jlm::rvsdg::Output *> outputs;
  auto & memoryStateMerge = LambdaExitMemoryStateMergeOperation::CreateNode(
      *lambda->subregion(),
      { loadOutput[1], memoryStateSplitNode.output(1) },
      { 0, 1 });

  auto lambdaOutput = lambda->finalize({ memoryStateMerge.output(0) });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  // This pass should have no effect on the graph
  jlm::util::StatisticsCollector statisticsCollector;
  UnusedStateRemoval::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  auto * node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin());
  auto lambdaSubregion = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(node)->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_EQ(lambdaSubregion->narguments(), 2);
  EXPECT_EQ(lambdaSubregion->nresults(), 1);
  EXPECT_TRUE(is<MemoryStateType>(lambdaSubregion->result(0)->Type()));
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<LambdaEntryMemoryStateSplitOperation>(
          *lambdaSubregion,
          true));
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<LambdaExitMemoryStateMergeOperation>(
          *lambdaSubregion,
          true));
}
