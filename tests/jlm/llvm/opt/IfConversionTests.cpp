/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/IfConversion.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
GammaWithoutMatch()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = ValueType::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::ControlType::Create(2), valueType, valueType },
      { valueType });

  RvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));
  const auto conditionValue = lambdaNode->GetFunctionArguments()[0];
  const auto trueValue = lambdaNode->GetFunctionArguments()[1];
  const auto falseValue = lambdaNode->GetFunctionArguments()[2];

  auto gammaNode = jlm::rvsdg::GammaNode::create(conditionValue, 2);
  auto gammaInput1 = gammaNode->AddEntryVar(trueValue);
  auto gammaInput2 = gammaNode->AddEntryVar(falseValue);
  auto gammaOutput =
      gammaNode->AddExitVar({ gammaInput1.branchArgument[0], gammaInput2.branchArgument[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  IfConversion ifConversion;
  ifConversion.Run(rvsdgModule, statisticsCollector);

  view(rvsdgModule.Rvsdg(), stdout);

  // Assert

  assert(gammaNode->IsDead());
  const auto selectNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaNode->subregion()->result(0)->origin());
  assert(selectNode && is<SelectOperation>(selectNode));
  assert(selectNode->input(1)->origin() == falseValue);
  assert(selectNode->input(2)->origin() == trueValue);

  const auto controlToBitsNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*selectNode->input(0)->origin());
  assert(controlToBitsNode && is<ControlToIntOperation>(controlToBitsNode));
  assert(controlToBitsNode->input(0)->origin() == conditionValue);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/IfConversionTests-GammaWithoutMatch",
    GammaWithoutMatch)

static void
EmptyGammaWithTwoSubregionsAndMatch()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = ValueType::Create();
  const auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  RvsdgModule rvsdgModule(FilePath(""), "", "");

  const auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));
  const auto conditionValue = lambdaNode->GetFunctionArguments()[0];
  const auto trueValue = lambdaNode->GetFunctionArguments()[1];
  const auto falseValue = lambdaNode->GetFunctionArguments()[2];

  auto caseValue = 24;
  const auto matchResult = match(32, { { caseValue, 0 } }, 1, 2, conditionValue);

  const auto gammaNode = jlm::rvsdg::GammaNode::create(matchResult, 2);
  auto [inputTrue, branchArgumentTrue] = gammaNode->AddEntryVar(trueValue);
  auto [inputFalse, branchArgumentFalse] = gammaNode->AddEntryVar(falseValue);
  auto [_, gammaOutput] = gammaNode->AddExitVar({ branchArgumentTrue[0], branchArgumentFalse[1] });

  const auto lambdaOutput = lambdaNode->finalize({ gammaOutput });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  IfConversion ifConversion;
  ifConversion.Run(rvsdgModule, statisticsCollector);

  view(rvsdgModule.Rvsdg(), stdout);

  // Assert

  assert(gammaNode->IsDead());
  const auto selectNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaNode->subregion()->result(0)->origin());
  assert(selectNode && is<SelectOperation>(selectNode));
  assert(selectNode->input(1)->origin() == trueValue);
  assert(selectNode->input(2)->origin() == falseValue);

  const auto eqNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*selectNode->input(0)->origin());
  assert(eqNode && is<IntegerEqOperation>(eqNode));

  auto constantNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*eqNode->input(0)->origin());
  if (constantNode)
  {
    assert(eqNode->input(1)->origin() == conditionValue);
    auto constantOperation =
        dynamic_cast<const IntegerConstantOperation *>(&constantNode->GetOperation());
    assert(constantOperation);
    assert(constantOperation->Representation() == 24);
  }
  else
  {
    assert(eqNode->input(0)->origin() == conditionValue);
    constantNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*eqNode->input(1)->origin());
    auto constantOperation =
        dynamic_cast<const IntegerConstantOperation *>(&constantNode->GetOperation());
    assert(constantOperation);
    assert(constantOperation->Representation() == 24);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/IfConversionTests-EmptyGammaWithTwoSubregionsAndMatch",
    EmptyGammaWithTwoSubregionsAndMatch)

static void
EmptyGammaWithTwoSubregions()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = ValueType::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  RvsdgModule rvsdgModule(FilePath(""), "", "");

  const auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));
  const auto trueValue = lambdaNode->GetFunctionArguments()[1];
  const auto falseValue = lambdaNode->GetFunctionArguments()[2];

  const auto matchResult = match(32, { { 0, 0 } }, 1, 2, lambdaNode->GetFunctionArguments()[0]);

  const auto gammaNode0 = jlm::rvsdg::GammaNode::create(matchResult, 2);
  const auto & c0 = jlm::rvsdg::CreateOpNode<jlm::rvsdg::ctlconstant_op>(
      *gammaNode0->subregion(0),
      jlm::rvsdg::ControlValueRepresentation(0, 2));
  const auto & c1 = jlm::rvsdg::CreateOpNode<jlm::rvsdg::ctlconstant_op>(
      *gammaNode0->subregion(1),
      jlm::rvsdg::ControlValueRepresentation(1, 2));
  auto c = gammaNode0->AddExitVar({ c0.output(0), c1.output(0) });

  const auto gammaNode1 = jlm::rvsdg::GammaNode::create(c.output, 2);
  auto [inputTrue, branchArgumentTrue] = gammaNode1->AddEntryVar(trueValue);
  auto [inputFalse, branchArgumentFalse] = gammaNode1->AddEntryVar(falseValue);
  auto [_, gammaOutput] = gammaNode1->AddExitVar({ branchArgumentFalse[0], branchArgumentTrue[1] });

  const auto lambdaOutput = lambdaNode->finalize({ gammaOutput });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  IfConversion ifConversion;
  ifConversion.Run(rvsdgModule, statisticsCollector);

  view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  assert(gammaNode1->IsDead());
  const auto selectNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaNode->subregion()->result(0)->origin());
  assert(selectNode && is<SelectOperation>(selectNode));
  assert(selectNode->input(1)->origin() == trueValue);
  assert(selectNode->input(2)->origin() == falseValue);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/IfConversionTests-EmptyGammaWithTwoSubregions",
    EmptyGammaWithTwoSubregions)

static void
EmptyGammaWithThreeSubregions()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = ValueType::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  RvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));

  auto match =
      jlm::rvsdg::match(32, { { 0, 0 }, { 1, 1 } }, 2, 3, lambdaNode->GetFunctionArguments()[0]);

  auto gammaNode = jlm::rvsdg::GammaNode::create(match, 3);
  auto gammaInput1 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto gammaInput2 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[2]);
  auto gammaOutput = gammaNode->AddExitVar({ gammaInput1.branchArgument[0],
                                             gammaInput1.branchArgument[1],
                                             gammaInput2.branchArgument[2] });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  IfConversion ifConversion;
  ifConversion.Run(rvsdgModule, statisticsCollector);

  view(rvsdgModule.Rvsdg(), stdout);

  // Assert

  // Only the gamma and match nodes should be in the lambda region. No select operation
  // should have been created.
  assert(lambdaNode->subregion()->nnodes() == 2);
  assert(!gammaNode->IsDead());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/IfConversionTests-EmptyGammaWithThreeSubregions",
    EmptyGammaWithThreeSubregions)

static void
PartialEmptyGamma()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = ValueType::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(1), valueType },
      { valueType });

  RvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambdaNode->GetFunctionArguments()[0]);
  auto gammaNode = jlm::rvsdg::GammaNode::create(match, 2);
  auto gammaInput = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto output = TestOperation::create(
                    gammaNode->subregion(1),
                    { gammaInput.branchArgument[1] },
                    { valueType })
                    ->output(0);
  auto gammaOutput = gammaNode->AddExitVar({ gammaInput.branchArgument[0], output });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  IfConversion ifConversion;
  ifConversion.Run(rvsdgModule, statisticsCollector);

  view(rvsdgModule.Rvsdg(), stdout);

  // Assert

  // Only the gamma and match nodes should be in the lambda region. No select operation
  // should have been created.
  assert(lambdaNode->subregion()->nnodes() == 2);
  assert(!gammaNode->IsDead());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/opt/IfConversionTests-PartialEmptyGamma",
    PartialEmptyGamma)
