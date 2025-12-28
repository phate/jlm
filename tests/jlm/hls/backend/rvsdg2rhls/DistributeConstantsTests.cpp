/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/hls/backend/rvsdg2rhls/distribute-constants.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
GammaSubregionUsage()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto controlType = ControlType::Create(3);
  auto bit32Type = BitType::Create(32);
  auto functionType = FunctionType::Create({ controlType }, { bit32Type });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];

  auto & constantNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 5);

  auto gammaNode = GammaNode::create(controlArgument, 3);
  auto entryVariable = gammaNode->AddEntryVar(constantNode.output(0));

  auto testNode0 = TestOperation::createNode(
      gammaNode->subregion(0),
      { entryVariable.branchArgument[0] },
      { bit32Type });

  auto testNode1 = TestOperation::createNode(
      gammaNode->subregion(1),
      { entryVariable.branchArgument[1] },
      { bit32Type });

  auto exitVariable = gammaNode->AddExitVar(
      { testNode0->output(0), testNode1->output(0), entryVariable.branchArgument[2] });

  auto lambdaOutput = lambdaNode->finalize({ exitVariable.output });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  ConstantDistribution::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  assert(lambdaNode->subregion()->numNodes() == 2);

  {
    // check subregion 0 - we expect the constantNode to be distributed into this subregion
    assert(gammaNode->subregion(0)->numNodes() == 2);
    assert(IsOwnerNodeOperation<IntegerConstantOperation>(*testNode0->input(0)->origin()));
  }

  {
    // check subregion 1 - we expect the constantNode to be distributed into this subregion
    assert(gammaNode->subregion(1)->numNodes() == 2);
    assert(IsOwnerNodeOperation<IntegerConstantOperation>(*testNode1->input(0)->origin()));
  }

  {
    // check subregion 2 - we expect the constantNode to be distributed into this subregion
    assert(gammaNode->subregion(2)->numNodes() == 1);
    assert(IsOwnerNodeOperation<IntegerConstantOperation>(*exitVariable.branchResult[2]->origin()));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/DistributeConstantsTests-GammaSubregionUsage",
    GammaSubregionUsage)

static void
NestedGammas()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto controlType = ControlType::Create(2);
  auto bit32Type = BitType::Create(32);
  auto functionType = FunctionType::Create({ controlType }, { bit32Type });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];

  auto & constantNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 5);

  auto gammaNodeOuter = GammaNode::create(controlArgument, 2);
  auto entryVarConstant = gammaNodeOuter->AddEntryVar(constantNode.output(0));

  // gammaNodeOuter subregion 0
  auto testNode0 = TestOperation::createNode(
      gammaNodeOuter->subregion(0),
      { entryVarConstant.branchArgument[0] },
      { bit32Type });

  // gammaNodeOuter subregion 1
  auto controlConstant = &ControlConstantOperation::create(*gammaNodeOuter->subregion(1), 2, 0);
  auto gammaNodeInner = GammaNode::create(controlConstant, 2);
  auto entryVariable = gammaNodeInner->AddEntryVar(entryVarConstant.branchArgument[1]);
  auto exitVariableInner = gammaNodeInner->AddExitVar(
      { entryVariable.branchArgument[0], entryVariable.branchArgument[1] });

  auto exitVariableOuter =
      gammaNodeOuter->AddExitVar({ testNode0->output(0), exitVariableInner.output });

  auto testNode1 = TestOperation::createNode(
      lambdaNode->subregion(),
      { exitVariableOuter.output },
      { bit32Type });

  auto lambdaOutput = lambdaNode->finalize({ testNode1->output(0) });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  ConstantDistribution::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  assert(lambdaNode->subregion()->numNodes() == 3);

  {
    // check gammaNodeOuter subregion 0
    assert(gammaNodeOuter->subregion(0)->numNodes() == 2);
    assert(IsOwnerNodeOperation<IntegerConstantOperation>(*testNode0->input(0)->origin()));
  }

  {
    // check gammaNodeOuter subregion 1
    // The constantNode was copied into this region (even though it does not have a user), so we
    // expect one more node than before the transformation.
    assert(gammaNodeOuter->subregion(1)->numNodes() == 3);

    {
      // check gammaNodeInner subregion 0
      assert(gammaNodeInner->subregion(0)->numNodes() == 0);
    }

    {
      // check gammaNodeInner subregion 1
      assert(gammaNodeInner->subregion(1)->numNodes() == 0);
    }
  }

  assert(TryGetOwnerNode<GammaNode>(*testNode1->input(0)->origin()));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/DistributeConstantsTests-NestedGammas",
    NestedGammas)

static void
Theta()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto controlType = ControlType::Create(3);
  auto bit32Type = BitType::Create(32);
  auto functionType = FunctionType::Create({}, { bit32Type, bit32Type, bit32Type });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & constantNode0 = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 0);
  auto & constantNode2 = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 2);

  auto thetaNode = ThetaNode::create(lambdaNode->subregion());

  auto loopVar0 = thetaNode->AddLoopVar(constantNode0.output(0));
  auto loopVar1 = thetaNode->AddLoopVar(constantNode0.output(0));
  auto loopVar2 = thetaNode->AddLoopVar(constantNode2.output(0));

  auto testNode0 =
      TestOperation::createNode(thetaNode->subregion(), { loopVar0.pre }, { bit32Type });
  auto & constantNode1 = IntegerConstantOperation::Create(*thetaNode->subregion(), 32, 1);
  auto testNode2 =
      TestOperation::createNode(thetaNode->subregion(), { loopVar2.pre }, { bit32Type });

  loopVar0.post->divert_to(testNode0->output(0));
  loopVar1.post->divert_to(constantNode1.output(0));

  auto testNode1 =
      TestOperation::createNode(thetaNode->subregion(), { loopVar0.output }, { bit32Type });

  auto lambdaOutput =
      lambdaNode->finalize({ testNode1->output(0), loopVar1.output, loopVar2.output });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  ConstantDistribution::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Arrange
  // We expect constantNode1 to be distributed from the theta subregion to the lambda subregion
  assert(lambdaNode->subregion()->numNodes() == 5);

  // We expect constantNode2 to be distributed from the lambda subregion to the theta subregion
  assert(thetaNode->subregion()->numNodes() == 5);

  {
    // We expect no changes with loopVar0
    auto loopVar = thetaNode->MapOutputLoopVar(*thetaNode->output(0));
    assert(lambdaNode->subregion()->result(0)->origin() == testNode1->output(0));
    assert(loopVar.output == testNode1->input(0)->origin());
    assert(loopVar.post->origin() == testNode0->output(0));
    assert(testNode0->input(0)->origin() == loopVar.pre);
    assert(loopVar.input->origin() == constantNode0.output(0));
  }

  {
    // We expect constantNode1 to be distributed from the theta subregion to the lambda subregion,
    // rendering loopVar1 to be dead
    auto loopVar = thetaNode->MapOutputLoopVar(*thetaNode->output(1));
    assert(loopVar.output->IsDead());
    assert(loopVar.pre->IsDead());

    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
            *lambdaNode->subregion()->result(1)->origin());
    assert(constantNode && constantOperation);
    assert(constantOperation->Representation() == 1);
  }

  {
    // LoopVar2 was a passthrough so we expect it to be redirected to constantNode2
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(
            *lambdaNode->subregion()->result(2)->origin());
    assert(constantNode && constantOperation);
    assert(constantNode == &constantNode2);
  }

  {
    // We expect constantNode2 to be distributed t o the theta subregion for testNode2
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*testNode2->input(0)->origin());
    assert(constantNode && constantOperation);
    assert(constantOperation->Representation() == 2);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/DistributeConstantsTests-Theta", Theta)

static void
Lambda()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto bit32Type = BitType::Create(32);
  auto functionType = FunctionType::Create({}, { bit32Type });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & constantNode0 = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 0);

  auto lambdaOutput = lambdaNode->finalize({ constantNode0.output(0) });

  GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  ConstantDistribution::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Arrange
  // We expect no change at all in the graph
  assert(lambdaNode->subregion()->numNodes() == 1);
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/DistributeConstantsTests-Lambda", Lambda)
