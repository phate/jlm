/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
TEST(NodeHoistingTests, simpleGamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto valueArgument = lambdaNode->GetFunctionArguments()[1];

  auto gammaNode = GammaNode::create(controlArgument, 2);
  auto entryVar = gammaNode->AddEntryVar(valueArgument);

  // gamma subregion 0
  auto constantNode = TestOperation::createNode(gammaNode->subregion(0), {}, { valueType });
  auto binaryNode = TestOperation::createNode(
      gammaNode->subregion(0),
      { entryVar.branchArgument[0], constantNode->output(0) },
      { valueType });

  // gamma subregion 1
  auto unaryNode = TestOperation::createNode(
      gammaNode->subregion(1),
      { entryVar.branchArgument[1] },
      { valueType });

  auto exitVar = gammaNode->AddExitVar({ binaryNode->output(0), unaryNode->output(0) });

  auto lambdaOutput = lambdaNode->finalize({ exitVar.output });

  GraphExport::Create(*lambdaOutput, "x");

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // All nodes from the gamma subregions should have been hoisted to the lambda subregion
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 4u);

  // The original nodes in the gamma subregions should have been removed
  EXPECT_EQ(gammaNode->subregion(0)->numNodes(), 0u);
  EXPECT_EQ(gammaNode->subregion(1)->numNodes(), 0u);
}

TEST(NodeHoistingTests, nestedGamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto valueArgument = lambdaNode->GetFunctionArguments()[1];

  auto gammaNode1 = GammaNode::create(controlArgument, 2);
  auto controlEntryVar = gammaNode1->AddEntryVar(controlArgument);
  auto valueEntryVar1 = gammaNode1->AddEntryVar(valueArgument);

  // gamma1 subregion 0
  auto constantNode1 = TestOperation::createNode(gammaNode1->subregion(0), {}, { valueType });

  auto gammaNode2 = GammaNode::create(controlEntryVar.branchArgument[0], 2);
  auto valueEntryVar2 = gammaNode2->AddEntryVar(valueEntryVar1.branchArgument[0]);
  auto valueEntryVar3 = gammaNode2->AddEntryVar(constantNode1->output(0));

  // gamma2 subregion 0
  auto binaryNode = TestOperation::createNode(
      gammaNode1->subregion(0),
      { valueEntryVar2.branchArgument[0], valueEntryVar3.branchArgument[0] },
      { valueType });

  // gamma2 subregion 1
  auto unaryNode = TestOperation::createNode(
      gammaNode1->subregion(1),
      { valueEntryVar2.branchArgument[1] },
      { valueType });

  auto exitVar1 = gammaNode2->AddExitVar({ binaryNode->output(0), unaryNode->output(0) });

  // gamma1 subregion 1
  auto constantNode2 = TestOperation::createNode(gammaNode1->subregion(1), {}, { valueType });

  auto exitVar2 = gammaNode1->AddExitVar({ exitVar1.output, constantNode2->output(0) });

  auto lambdaOutput = lambdaNode->finalize({ exitVar2.output });

  GraphExport::Create(*lambdaOutput, "x");

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // All simple nodes from both gamma subregions should have been hoisted to the lambda subregion
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 5u);

  // Only gamma node 2 should be left in gamma node 1 subregion 0
  EXPECT_EQ(gammaNode1->subregion(0)->numNodes(), 1u);
  EXPECT_EQ(gammaNode1->subregion(1)->numNodes(), 0u);

  // All nodes should have been hoisted out
  EXPECT_EQ(gammaNode2->subregion(0)->numNodes(), 0u);
  EXPECT_EQ(gammaNode2->subregion(1)->numNodes(), 0u);
}

TEST(NodeHoistingTests, simpleTheta)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto valueArgument = lambdaNode->GetFunctionArguments()[1];

  auto thetaNode = ThetaNode::create(lambdaNode->subregion());

  auto lv1 = thetaNode->AddLoopVar(controlArgument);
  auto lv2 = thetaNode->AddLoopVar(valueArgument);
  auto lv3 = thetaNode->AddLoopVar(valueArgument);
  auto lv4 = thetaNode->AddLoopVar(valueArgument);

  auto node1 = TestOperation::createNode(thetaNode->subregion(), {}, { valueType });
  auto node2 = TestOperation::createNode(
      thetaNode->subregion(),
      { node1->output(0), lv3.pre },
      { valueType });
  auto node3 = TestOperation::createNode(
      thetaNode->subregion(),
      { lv2.pre, node2->output(0) },
      { valueType });
  auto node4 =
      TestOperation::createNode(thetaNode->subregion(), { lv3.pre, lv4.pre }, { valueType });

  lv2.post->divert_to(node3->output(0));
  lv4.post->divert_to(node4->output(0));

  thetaNode->set_predicate(lv1.pre);

  lambdaNode->finalize({ thetaNode->output(1) });

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect node1 and node2 to be hoisted out of the theta subregion
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 3u);
  EXPECT_EQ(thetaNode->subregion()->numNodes(), 2u);

  EXPECT_EQ(lv2.post->origin(), node3->output(0));
  EXPECT_EQ(lv4.post->origin(), node4->output(0));
}

TEST(NodeHoistingTests, invariantMemoryOperation)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();
  const auto pointerType = PointerType::Create();
  const auto controlType = ControlType::Create(2);
  const auto valueType = TestType::createValueType();
  const auto functionType = FunctionType::Create(
      { controlType, pointerType, valueType, memoryStateType },
      { memoryStateType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto pointerArgument = lambdaNode->GetFunctionArguments()[1];
  auto valueArgument = lambdaNode->GetFunctionArguments()[2];
  auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];

  auto thetaNode = ThetaNode::create(lambdaNode->subregion());

  auto lvc = thetaNode->AddLoopVar(controlArgument);
  auto lva = thetaNode->AddLoopVar(pointerArgument);
  auto lvv = thetaNode->AddLoopVar(valueArgument);
  auto lvs = thetaNode->AddLoopVar(memoryStateArgument);

  auto & storeNode = StoreNonVolatileOperation::CreateNode(*lva.pre, *lvv.pre, { lvs.pre }, 4);

  lvs.post->divert_to(storeNode.output(0));
  thetaNode->set_predicate(lvc.pre);

  lambdaNode->finalize({ lvs.output });

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect the store node hoisted out of the theta subregion
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 2u);
  EXPECT_EQ(thetaNode->subregion()->numNodes(), 0u);

  // We expect no new input to be added to the theta node as the store node should have been
  // "hoisted along" its memory state edges.
  EXPECT_EQ(thetaNode->ninputs(), 4u);
}

TEST(NodeHoistingTests, statefulOperations)
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto controlType = ControlType::Create(2);
  auto valueType = TestType::createValueType();
  auto stateType = TestType::createStateType();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
          stateType,
      },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto valueArgument = lambdaNode->GetFunctionArguments()[1];
  auto stateArgument = lambdaNode->GetFunctionArguments()[2];

  auto gammaNode1 = GammaNode::create(controlArgument, 2);
  auto controlEntryVar = gammaNode1->AddEntryVar(controlArgument);
  auto valueEntryVar1 = gammaNode1->AddEntryVar(valueArgument);
  auto stateEntryVar = gammaNode1->AddEntryVar(stateArgument);

  auto stateNode = TestOperation::createNode(
      gammaNode1->subregion(0),
      { valueEntryVar1.branchArgument[0], stateEntryVar.branchArgument[0] },
      { valueType });

  auto gammaNode2 = GammaNode::create(controlEntryVar.branchArgument[0], 2);
  auto valueEntryVar2 = gammaNode2->AddEntryVar(stateNode->output(0));
  auto valueEntryVar3 = gammaNode2->AddEntryVar(valueEntryVar1.branchArgument[0]);

  auto binaryNode = TestOperation::createNode(
      gammaNode2->subregion(0),
      { valueEntryVar2.branchArgument[0], valueEntryVar3.branchArgument[0] },
      { valueType });

  auto exitVar2 =
      gammaNode2->AddExitVar({ binaryNode->output(0), valueEntryVar2.branchArgument[1] });

  auto exitVar = gammaNode1->AddExitVar({ exitVar2.output, valueEntryVar1.branchArgument[1] });

  lambdaNode->finalize({ exitVar.output });

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect that stateNode stays where it is and only the binaryNode is hoisted to the same
  // region as stateNode

  // Gamma node and undef node
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 1u);

  // stateNode, gammaNode2, and binaryNode
  EXPECT_EQ(gammaNode1->subregion(0)->numNodes(), 3u);
  EXPECT_EQ(gammaNode1->subregion(1)->numNodes(), 0u);

  EXPECT_EQ(gammaNode2->subregion(0)->numNodes(), 0u);
  EXPECT_EQ(gammaNode2->subregion(1)->numNodes(), 0u);
}

TEST(NodeHoistingTests, controlConstants)
{
  /**
   * Creates an RVSDG that looks like
   *
   * +-lambda-----------------------x-x-+
   * |           undef              | | |
   * |             v                | | |
   * | +-theta-----x--------+       | | |
   * | |                    |       | | |
   * | | Ctrl(0) Ctrl(1)    |       | | |
   * | |   v       v        |       | | |
   * | +---x-------x--------+       | | |
   * |             v                | | |
   * | +-gamma1----+-----------+    | | |
   * | |           |           |    | | |
   * | |  Ctrl(1)  |  Ctrl(0)  |    | | |
   * | |    v      |     v     |    | | |
   * | +----x------+-----x-----+    | | |
   * |      v                       | | |
   * | +-gamma2----+-----------+    | | |
   * | |           |           |    | | |
   * | | Int32(3)  |  Int32(7) |    | | |
   * | |    v      |     v     |    | | |
   * | +----x------+-----x-----+    | | |
   * |      v                       v v |
   * +------x-----------------------x-x-+
   *
   * and checks that none of the control constants are moved by the node hoisting pass.
   */

  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto controlType = ControlType::Create(2);
  auto int32Type = BitType::Create(32);
  auto memoryStateType = MemoryStateType::Create();
  auto ioStateType = IOStateType::Create();
  const auto functionType = FunctionType::Create(
      { ioStateType, memoryStateType },
      { int32Type, ioStateType, memoryStateType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "func", Linkage::externalLinkage));

  auto ioStateArgument = lambdaNode->GetFunctionArguments()[0];
  auto memStateArgument = lambdaNode->GetFunctionArguments()[1];

  // Theta node
  auto thetaNode = ThetaNode::create(lambdaNode->subregion());
  auto thetaUndef = UndefValueOperation::Create(*lambdaNode->subregion(), controlType);
  auto thetaCtrlLoopVar = thetaNode->AddLoopVar(thetaUndef);
  auto & thetaInnerCtrl1 = ControlConstantOperation::create(*thetaNode->subregion(), 2, 1);
  thetaCtrlLoopVar.post->divert_to(&thetaInnerCtrl1);

  // gamma1: takes theta's control loop var output as its predicate
  auto & gamma1 = GammaNode::Create(*thetaCtrlLoopVar.output, 2, {});

  // gamma1 exit variable taking Ctrl(0) and Ctrl(1) in the respective subregions
  auto & gamma1Ctrl1 = ControlConstantOperation::create(*gamma1.subregion(0), 2, 1);
  auto & gamma1Ctrl0 = ControlConstantOperation::create(*gamma1.subregion(1), 2, 0);
  auto gamma1Exit = gamma1.AddExitVar({ &gamma1Ctrl1, &gamma1Ctrl0 });

  // gamma2: takes gamma1's exit (control) as its predicate
  auto & gamma2 = GammaNode::Create(*gamma1Exit.output, 2, {});

  // gamma2 exit variable takes integer constants
  auto & gamma2Int3 = *IntegerConstantOperation::Create(*gamma2.subregion(0), 32, 3).output(0);
  auto & gamma2Int7 = *IntegerConstantOperation::Create(*gamma2.subregion(1), 32, 7).output(0);

  auto gamma2Exit = gamma2.AddExitVar({ &gamma2Int3, &gamma2Int7 });

  lambdaNode->finalize({ gamma2Exit.output, ioStateArgument, memStateArgument });

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // Control constants must stay in their original regions and not be hoisted.
  // Bit constants (Int32) should be hoisted to the lambda subregion.

  // Lambda subregion: theta node + gamma1 + gamma2 + CTL(0) (loop var entry) + two hoisted Int32
  // constants
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 6u);

  // Theta subregion: The Ctrl(0) and Ctrl(1) remain inside the theta
  EXPECT_EQ(thetaNode->subregion()->numNodes(), 2u);
  auto thetaPredicateOwner = TryGetOwnerNode<SimpleNode>(*thetaNode->predicate()->origin());
  EXPECT_TRUE(thetaPredicateOwner);
  EXPECT_EQ(thetaPredicateOwner->region(), thetaNode->subregion());
  auto thetaPostOwner = TryGetOwnerNode<SimpleNode>(*thetaCtrlLoopVar.post->origin());
  EXPECT_TRUE(thetaPostOwner);
  EXPECT_EQ(thetaPostOwner->region(), thetaNode->subregion());

  // Gamma1 subregions: control constants stay in place (no structural nodes inside)
  EXPECT_EQ(gamma1.subregion(0)->numNodes(), 1u); // Ctrl(1) only
  auto gamma1LeftCtrlOwner = TryGetOwnerNode<SimpleNode>(*gamma1Exit.branchResult[0]->origin());
  EXPECT_TRUE(gamma1LeftCtrlOwner);
  EXPECT_EQ(gamma1LeftCtrlOwner->region(), gamma1.subregion(0));

  EXPECT_EQ(gamma1.subregion(1)->numNodes(), 1u); // Ctrl(0) only
  auto gamma1RightCtrlOwner = TryGetOwnerNode<SimpleNode>(*gamma1Exit.branchResult[1]->origin());
  EXPECT_TRUE(gamma1RightCtrlOwner);
  EXPECT_EQ(gamma1RightCtrlOwner->region(), gamma1.subregion(1));

  // Gamma2 subregions: Int32 constants should have been hoisted out to lambda level
  EXPECT_EQ(gamma2.subregion(0)->numNodes(), 0u);
  EXPECT_EQ(gamma2.subregion(1)->numNodes(), 0u);
}

TEST(NodeHoistingTests, hoistLoadNodesOutOfGamma)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto ptrType = PointerType::Create();
  const auto i32Type = BitType::Create(32);
  const auto ioStateType = IOStateType::Create();
  const auto memoryStateType = MemoryStateType::Create();
  const auto controlType = ControlType::Create(2);
  const auto functionType = FunctionType::Create(
      { controlType, ptrType, ioStateType, memoryStateType },
      { i32Type, ioStateType, memoryStateType });

  LlvmRvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto ptrArgument = lambdaNode->GetFunctionArguments()[1];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];
  auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];

  auto gammaNode = GammaNode::create(controlArgument, 2);
  auto ptrEntryVar = gammaNode->AddEntryVar(ptrArgument);
  auto ioStateEntryVar = gammaNode->AddEntryVar(ioStateArgument);
  auto memoryStateEntryVar = gammaNode->AddEntryVar(memoryStateArgument);

  // gamma subregion 0
  auto & ioBarrierNode = IOBarrierOperation::createNode(
      *ptrEntryVar.branchArgument[0],
      *ioStateEntryVar.branchArgument[0]);
  auto & loadNode0 = LoadNonVolatileOperation::CreateNode(
      *ioBarrierNode.output(0),
      { memoryStateEntryVar.branchArgument[0] },
      i32Type,
      4);

  // gamma subregion 1
  auto & loadNode1 = LoadNonVolatileOperation::CreateNode(
      *ptrEntryVar.branchArgument[1],
      { memoryStateEntryVar.branchArgument[1] },
      i32Type,
      4);

  auto i32ExitVar = gammaNode->AddExitVar({ loadNode0.output(0), loadNode1.output(0) });
  auto ioStateExitVar = gammaNode->AddExitVar(
      { ioStateEntryVar.branchArgument[0], ioStateEntryVar.branchArgument[1] });
  auto memoryStateExitVar = gammaNode->AddExitVar({ loadNode0.output(1), loadNode1.output(1) });

  auto lambdaOutput =
      lambdaNode->finalize({ i32ExitVar.output, ioStateExitVar.output, memoryStateExitVar.output });

  GraphExport::Create(*lambdaOutput, "x");

  // Act
  NodeHoisting nodeHoisting;
  util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  // Assert
  // We expect the load node from gamma subregion 1 to be hoisted out
  EXPECT_EQ(lambdaNode->subregion()->numNodes(), 2u);
  EXPECT_EQ(gammaNode->subregion(0)->numNodes(), 2u);
  EXPECT_EQ(gammaNode->subregion(1)->numNodes(), 0u);

  // We expect that only one input was added to the gamma node: the loaded value of the hoisted load
  // node. We do not expect that a new input for the memory state of the load node was added as the
  // load node should have been "hoisted along" its state edge.
  EXPECT_EQ(gammaNode->ninputs(), 5u);
}

}
