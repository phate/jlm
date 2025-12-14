/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
simpleGamma()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto valueArgument = lambdaNode->GetFunctionArguments()[1];

  auto gammaNode = GammaNode::create(controlArgument, 2);
  auto entryVar = gammaNode->AddEntryVar(valueArgument);

  // gamma subregion 0
  auto constantNode = TestOperation::create(gammaNode->subregion(0), {}, { valueType });
  auto binaryNode = TestOperation::create(
      gammaNode->subregion(0),
      { entryVar.branchArgument[0], constantNode->output(0) },
      { valueType });

  // gamma subregion 1
  auto unaryNode =
      TestOperation::create(gammaNode->subregion(1), { entryVar.branchArgument[1] }, { valueType });

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
  assert(lambdaNode->subregion()->numNodes() == 4);

  // The original nodes in the gamma subregions should have been removed
  assert(gammaNode->subregion(0)->numNodes() == 0);
  assert(gammaNode->subregion(1)->numNodes() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-simpleGamma", simpleGamma)

static void
nestedGamma()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  const auto controlType = ControlType::Create(2);
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  auto constantNode1 = TestOperation::create(gammaNode1->subregion(0), {}, { valueType });

  auto gammaNode2 = GammaNode::create(controlEntryVar.branchArgument[0], 2);
  auto valueEntryVar2 = gammaNode2->AddEntryVar(valueEntryVar1.branchArgument[0]);
  auto valueEntryVar3 = gammaNode2->AddEntryVar(constantNode1->output(0));

  // gamma2 subregion 0
  auto binaryNode = TestOperation::create(
      gammaNode1->subregion(0),
      { valueEntryVar2.branchArgument[0], valueEntryVar3.branchArgument[0] },
      { valueType });

  // gamma2 subregion 1
  auto unaryNode = TestOperation::create(
      gammaNode1->subregion(1),
      { valueEntryVar2.branchArgument[1] },
      { valueType });

  auto exitVar1 = gammaNode2->AddExitVar({ binaryNode->output(0), unaryNode->output(0) });

  // gamma1 subregion 1
  auto constantNode2 = TestOperation::create(gammaNode1->subregion(1), {}, { valueType });

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
  assert(lambdaNode->subregion()->numNodes() == 5);

  // Only gamma node 2 should be left in gamma node 1 subregion 0
  assert(gammaNode1->subregion(0)->numNodes() == 1);
  assert(gammaNode1->subregion(1)->numNodes() == 0);

  // All nodes should have been hoisted out
  assert(gammaNode2->subregion(0)->numNodes() == 0);
  assert(gammaNode2->subregion(1)->numNodes() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-nestedGamma", nestedGamma)

static void
simpleTheta()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  auto controlType = ControlType::Create(2);
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          valueType,
      },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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

  auto node1 = TestOperation::create(thetaNode->subregion(), {}, { valueType });
  auto node2 =
      TestOperation::create(thetaNode->subregion(), { node1->output(0), lv3.pre }, { valueType });
  auto node3 =
      TestOperation::create(thetaNode->subregion(), { lv2.pre, node2->output(0) }, { valueType });
  auto node4 = TestOperation::create(thetaNode->subregion(), { lv3.pre, lv4.pre }, { valueType });

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
  assert(lambdaNode->subregion()->numNodes() == 3);
  assert(thetaNode->subregion()->numNodes() == 2);

  assert(lv2.post->origin() == node3->output(0));
  assert(lv4.post->origin() == node4->output(0));
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-simpleTheta", simpleTheta)

static void
invariantMemoryOperation()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange
  const auto memoryStateType = MemoryStateType::Create();
  const auto pointerType = PointerType::Create();
  const auto controlType = ControlType::Create(2);
  const auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      { controlType, pointerType, valueType, memoryStateType },
      { memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  assert(lambdaNode->subregion()->numNodes() == 2);
  assert(thetaNode->subregion()->numNodes() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-invariantMemoryOperation", invariantMemoryOperation)

static void
ioBarrier()
{
  // Arrange
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  auto controlType = ControlType::Create(2);
  auto pointerType = PointerType::Create();
  auto ioStateType = IOStateType::Create();
  auto valueType = ValueType::Create();
  const auto functionType = FunctionType::Create(
      {
          controlType,
          pointerType,
          ioStateType,
      },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto pointerArgument = lambdaNode->GetFunctionArguments()[1];
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[2];

  auto gammaNode = GammaNode::create(controlArgument, 2);

  auto addressEntryVar = gammaNode->AddEntryVar(pointerArgument);
  auto ioStateEntryVar = gammaNode->AddEntryVar(ioStateArgument);

  auto & ioBarrierNode = IOBarrierOperation::createNode(
      *addressEntryVar.branchArgument[0],
      *ioStateEntryVar.branchArgument[0]);

  auto & loadNode =
      LoadNonVolatileOperation::CreateNode(*ioBarrierNode.output(0), {}, valueType, 4);

  auto undefValue = UndefValueOperation::Create(*gammaNode->subregion(1), valueType);

  auto exitVar = gammaNode->AddExitVar({ loadNode.output(0), undefValue });

  lambdaNode->finalize({ exitVar.output });

  view(rvsdg, stdout);

  // Act
  NodeHoisting nodeHoisting;
  jlm::util::StatisticsCollector statisticsCollector;
  nodeHoisting.Run(rvsdgModule, statisticsCollector);

  view(rvsdg, stdout);

  // Assert
  // We expect that only the undef value was hoisted

  // Gamma node and undef node
  assert(lambdaNode->subregion()->numNodes() == 2);

  // IOBarrier and load node
  assert(gammaNode->subregion(0)->numNodes() == 2);
  assert(gammaNode->subregion(1)->numNodes() == 0);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push-ioBarrier", ioBarrier)
