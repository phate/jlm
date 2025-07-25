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
  auto valueType = jlm::tests::ValueType::Create();
  auto functionType = FunctionType::Create({ controlType }, { valueType });

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
      { valueType });

  auto testNode1 = TestOperation::create(
      gammaNode->subregion(1),
      { entryVariable.branchArgument[1] },
      { valueType });

  auto testNode2 = TestOperation::create(
      gammaNode->subregion(2),
      { entryVariable.branchArgument[2] },
      { valueType });

  auto exitVariable =
      gammaNode->AddExitVar({ testNode0->output(0), testNode1->output(0), testNode2->output(0) });

  auto lambdaOutput = lambdaNode->finalize({ exitVariable.output });

  jlm::tests::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  distribute_constants(rvsdgModule);
  view(rvsdg, stdout);

  // Assert
  // We expect the constantNode to be distributed into each gammaNode subregion
  assert(lambdaNode->subregion()->nnodes() == 2);
  assert(gammaNode->subregion(0)->nnodes() == 2);
  assert(gammaNode->subregion(1)->nnodes() == 2);
  assert(gammaNode->subregion(2)->nnodes() == 2);

  {
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOp<IntegerConstantOperation>(*testNode0->input(0)->origin());
    assert(constantNode && constantOperation);
  }

  {
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOp<IntegerConstantOperation>(*testNode1->input(0)->origin());
    assert(constantNode && constantOperation);
  }

  {
    auto [constantNode, constantOperation] =
        TryGetSimpleNodeAndOp<IntegerConstantOperation>(*testNode2->input(0)->origin());
    assert(constantNode && constantOperation);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/DistributeConstantsTests-GammaSubregionUsage",
    GammaSubregionUsage)
