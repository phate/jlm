/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/hls/backend/rvsdg2rhls/distribute-constants.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
GammaSubregionUsage()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto controlType = ControlType::Create(3);
  auto bit32Type = bittype::Create(32);
  auto functionType = FunctionType::Create({ controlType }, { bit32Type });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];

  auto & constantNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 5);

  auto gammaNode = GammaNode::create(controlArgument, 3);
  auto entryVariable = gammaNode->AddEntryVar(constantNode.output(0));

  auto testNode0 = TestOperation::create(
      gammaNode->subregion(0),
      { entryVariable.branchArgument[0] },
      { bit32Type });

  auto testNode1 = TestOperation::create(
      gammaNode->subregion(1),
      { entryVariable.branchArgument[1] },
      { bit32Type });

  auto exitVariable = gammaNode->AddExitVar(
      { testNode0->output(0), testNode1->output(0), entryVariable.branchArgument[2] });

  auto lambdaOutput = lambdaNode->finalize({ exitVariable.output });

  jlm::tests::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  distribute_constants(rvsdgModule);
  view(rvsdg, stdout);

  // Assert
  assert(lambdaNode->subregion()->nnodes() == 2);
  assert(constantNode.output(0)->IsDead());

  {
    // check subregion 0 - we expect the constantNode to be distributed into this subregion
    assert(gammaNode->subregion(0)->nnodes() == 2);
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOp<IntegerConstantOperation>(*testNode0->input(0)->origin());
    assert(constantNode && constantOperation);
  }

  {
    // check subregion 1 - we expect the constantNode to be distributed into this subregion
    assert(gammaNode->subregion(1)->nnodes() == 2);
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOp<IntegerConstantOperation>(*testNode1->input(0)->origin());
    assert(constantNode && constantOperation);
  }

  {
    // check subregion 2 - we expect the constantNode to be distributed into this subregion
    assert(gammaNode->subregion(2)->nnodes() == 1);
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOp<IntegerConstantOperation>(*exitVariable.branchResult[2]->origin());
    assert(constantNode && constantOperation);
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
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto controlType = ControlType::Create(2);
  auto bit32Type = bittype::Create(32);
  auto functionType = FunctionType::Create({ controlType }, { bit32Type });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", linkage::external_linkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];

  auto & constantNode = IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 5);

  auto gammaNodeOuter = GammaNode::create(controlArgument, 2);
  auto entryVarConstant = gammaNodeOuter->AddEntryVar(constantNode.output(0));

  // gammaNodeOuter subregion 0
  auto testNode0 = TestOperation::create(
      gammaNodeOuter->subregion(0),
      { entryVarConstant.branchArgument[0] },
      { bit32Type });

  // gammaNodeOuter subregion 1
  auto controlConstant = control_constant(gammaNodeOuter->subregion(1), 2, 0);
  auto gammaNodeInner = GammaNode::create(controlConstant, 2);
  auto entryVariable = gammaNodeInner->AddEntryVar(entryVarConstant.branchArgument[1]);
  auto exitVariableInner = gammaNodeInner->AddExitVar(
      { entryVariable.branchArgument[0], entryVariable.branchArgument[1] });

  auto exitVariableOuter =
      gammaNodeOuter->AddExitVar({ testNode0->output(0), exitVariableInner.output });

  auto testNode1 =
      TestOperation::create(lambdaNode->subregion(), { exitVariableOuter.output }, { bit32Type });

  auto lambdaOutput = lambdaNode->finalize({ testNode1->output(0) });

  jlm::tests::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  distribute_constants(rvsdgModule);
  view(rvsdg, stdout);

  // Assert
  assert(lambdaNode->subregion()->nnodes() == 3);

  {
    // check gammaNodeOuter subregion 0
    assert(gammaNodeOuter->subregion(0)->nnodes() == 2);
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOp<IntegerConstantOperation>(*testNode0->input(0)->origin());
    assert(constantNode && constantOperation);
  }

  {
    // check gammaNodeOuter subregion 1
    // FIXME: The constantNode was copied in this region even though there is no user for this
    // node in this region after(!) the pass is finished. DNE should later take care of this, but it
    // is unnecessary.
    assert(gammaNodeOuter->subregion(1)->nnodes() == 3);

    {
      // check gammaNodeInner subregion 0
      assert(gammaNodeInner->subregion(0)->nnodes() == 1);
      auto [constantNode, constantOperation] = TryGetSimpleNodeAndOp<IntegerConstantOperation>(
          *exitVariableInner.branchResult[0]->origin());
      assert(constantNode && constantOperation);
    }

    {
      // check gammaNodeInner subregion 1
      assert(gammaNodeInner->subregion(1)->nnodes() == 1);
      auto [constantNode, constantOperation] = TryGetSimpleNodeAndOp<IntegerConstantOperation>(
          *exitVariableInner.branchResult[1]->origin());
      assert(constantNode && constantOperation);
    }
  }

  assert(TryGetOwnerNode<GammaNode>(*testNode1->input(0)->origin()));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/DistributeConstantsTests-NestedGammas",
    NestedGammas)
