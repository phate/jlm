/*
 * Copyright 2018, 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
GammaWithMatch()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(1), valueType, valueType },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambdaNode->GetFunctionArguments()[0]);
  auto gamma = jlm::rvsdg::GammaNode::create(match, 2);
  auto gammaInput1 = gamma->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto gammaInput2 = gamma->AddEntryVar(lambdaNode->GetFunctionArguments()[2]);
  auto gammaOutput =
      gamma->AddExitVar({ gammaInput1.branchArgument[0], gammaInput2.branchArgument[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });
  GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  assert(cfg->nnodes() == 4);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-GammaWithMatch",
    GammaWithMatch)

static void
GammaWithoutMatch()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::ControlType::Create(2), valueType, valueType },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));

  auto gammaNode = jlm::rvsdg::GammaNode::create(lambdaNode->GetFunctionArguments()[0], 2);
  auto gammaInput1 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto gammaInput2 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[2]);
  auto gammaOutput =
      gammaNode->AddExitVar({ gammaInput1.branchArgument[0], gammaInput2.branchArgument[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });
  GraphExport::Create(*lambdaOutput, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  assert(cfg->nnodes() == 4);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-GammaWithoutMatch",
    GammaWithoutMatch)

static void
EmptyGammaWithTwoSubregionsAndMatch()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  const auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");

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
  GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  const auto module =
      RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  const auto & ipGraph = module->ipgraph();
  assert(ipGraph.nnodes() == 1);

  const auto controlFlowGraph = dynamic_cast<const FunctionNode &>(*ipGraph.begin()).cfg();
  assert(is_closed(*controlFlowGraph));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-EmptyGammaWithTwoSubregionsAndMatch",
    EmptyGammaWithTwoSubregionsAndMatch)

static void
EmptyGammaWithTwoSubregions()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");

  const auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));
  const auto trueValue = lambdaNode->GetFunctionArguments()[1];
  const auto falseValue = lambdaNode->GetFunctionArguments()[2];

  const auto matchResult = match(32, { { 0, 0 } }, 1, 2, lambdaNode->GetFunctionArguments()[0]);

  const auto gammaNode0 = jlm::rvsdg::GammaNode::create(matchResult, 2);
  const auto & c0 = jlm::rvsdg::CreateOpNode<ControlConstantOperation>(
      *gammaNode0->subregion(0),
      ControlValueRepresentation(0, 2));
  const auto & c1 = jlm::rvsdg::CreateOpNode<ControlConstantOperation>(
      *gammaNode0->subregion(1),
      ControlValueRepresentation(1, 2));
  auto c = gammaNode0->AddExitVar({ c0.output(0), c1.output(0) });

  const auto gammaNode1 = jlm::rvsdg::GammaNode::create(c.output, 2);
  auto [inputTrue, branchArgumentTrue] = gammaNode1->AddEntryVar(trueValue);
  auto [inputFalse, branchArgumentFalse] = gammaNode1->AddEntryVar(falseValue);
  auto [_, gammaOutput] = gammaNode1->AddExitVar({ branchArgumentFalse[0], branchArgumentTrue[1] });

  const auto lambdaOutput = lambdaNode->finalize({ gammaOutput });
  GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  const auto module =
      RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  const auto & ipGraph = module->ipgraph();
  assert(ipGraph.nnodes() == 1);

  const auto controlFlowGraph = dynamic_cast<const FunctionNode &>(*ipGraph.begin()).cfg();
  assert(is_closed(*controlFlowGraph));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-EmptyGammaWithTwoSubregions",
    EmptyGammaWithTwoSubregions)

static void
EmptyGammaWithThreeSubregions()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");

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
  GraphExport::Create(*lambdaOutput, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  assert(is_closed(*cfg));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-EmptyGammaWithThreeSubregions",
    EmptyGammaWithThreeSubregions)

static void
PartialEmptyGamma()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(1), valueType },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", Linkage::externalLinkage));

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambdaNode->GetFunctionArguments()[0]);
  auto gammaNode = jlm::rvsdg::GammaNode::create(match, 2);
  auto gammaInput = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto output = TestOperation::createNode(
                    gammaNode->subregion(1),
                    { gammaInput.branchArgument[1] },
                    { valueType })
                    ->output(0);
  auto gammaOutput = gammaNode->AddExitVar({ gammaInput.branchArgument[0], output });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });

  GraphExport::Create(*lambdaOutput, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  std::cout << ControlFlowGraph::ToAscii(*cfg) << std::flush;

  assert(is_proper_structured(*cfg));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-PartialEmptyGamma",
    PartialEmptyGamma)

static void
RecursiveData()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = jlm::rvsdg::TestType::createValueType();
  auto pt = PointerType::Create();

  jlm::llvm::RvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto imp = &jlm::llvm::GraphImport::Create(rm.Rvsdg(), vt, pt, "", Linkage::externalLinkage);

  jlm::rvsdg::PhiBuilder pb;
  pb.begin(&rm.Rvsdg().GetRootRegion());
  auto region = pb.subregion();
  auto r1 = pb.AddFixVar(pt);
  auto r2 = pb.AddFixVar(pt);
  auto dep = pb.AddContextVar(*imp);

  jlm::rvsdg::Output *delta1 = nullptr, *delta2 = nullptr;
  {
    auto delta = jlm::rvsdg::DeltaNode::Create(
        region,
        jlm::llvm::DeltaOperation::Create(vt, "test-delta1", Linkage::externalLinkage, "", false));
    auto dep1 = delta->AddContextVar(*r2.recref).inner;
    auto dep2 = delta->AddContextVar(*dep.inner).inner;
    delta1 = &delta->finalize(
        TestOperation::createNode(delta->subregion(), { dep1, dep2 }, { vt })->output(0));
  }

  {
    auto delta = jlm::rvsdg::DeltaNode::Create(
        region,
        jlm::llvm::DeltaOperation::Create(vt, "test-delta2", Linkage::externalLinkage, "", false));
    auto dep1 = delta->AddContextVar(*r1.recref).inner;
    auto dep2 = delta->AddContextVar(*dep.inner).inner;
    delta2 = &delta->finalize(
        TestOperation::createNode(delta->subregion(), { dep1, dep2 }, { vt })->output(0));
  }

  r1.result->divert_to(delta1);
  r2.result->divert_to(delta2);

  auto phi = pb.end();
  GraphExport::Create(*phi->output(0), "");

  jlm::rvsdg::view(rm.Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rm, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 3);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-RecursiveData",
    RecursiveData)
