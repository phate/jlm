/*
 * Copyright 2018, 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static int
GammaWithMatch()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(1), valueType, valueType },
      { valueType });

  RvsdgModule rvsdgModule(filepath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", linkage::external_linkage));

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambdaNode->GetFunctionArguments()[0]);
  auto gamma = jlm::rvsdg::GammaNode::create(match, 2);
  auto gammaInput1 = gamma->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto gammaInput2 = gamma->AddEntryVar(lambdaNode->GetFunctionArguments()[2]);
  auto gammaOutput =
      gamma->AddExitVar({ gammaInput1.branchArgument[0], gammaInput2.branchArgument[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });
  jlm::llvm::GraphExport::Create(*lambdaOutput, "");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const function_node &>(*ipg.begin()).cfg();
  assert(cfg->nnodes() == 1);
  auto node = cfg->entry()->OutEdge(0)->sink();
  auto bb = dynamic_cast<const basic_block *>(node);
  assert(is<SelectOperation>(bb->tacs().last()->operation()));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-GammaWithMatch",
    GammaWithMatch)

static int
GammaWithoutMatch()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::ControlType::Create(2), valueType, valueType },
      { valueType });

  RvsdgModule rvsdgModule(filepath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", linkage::external_linkage));

  auto gammaNode = jlm::rvsdg::GammaNode::create(lambdaNode->GetFunctionArguments()[0], 2);
  auto gammaInput1 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto gammaInput2 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[2]);
  auto gammaOutput =
      gammaNode->AddExitVar({ gammaInput1.branchArgument[0], gammaInput2.branchArgument[1] });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });
  jlm::llvm::GraphExport::Create(*lambdaOutput, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const function_node &>(*ipg.begin()).cfg();
  assert(cfg->nnodes() == 1);
  auto node = cfg->entry()->OutEdge(0)->sink();
  auto bb = dynamic_cast<const basic_block *>(node);
  assert(is<ctl2bits_op>(bb->tacs().first()->operation()));
  assert(is<SelectOperation>(bb->tacs().last()->operation()));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-GammaWithoutMatch",
    GammaWithoutMatch)

static int
EmptyGammaWithThreeSubregions()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(32), valueType, valueType },
      { valueType });

  RvsdgModule rvsdgModule(filepath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", linkage::external_linkage));

  auto match =
      jlm::rvsdg::match(32, { { 0, 0 }, { 1, 1 } }, 2, 3, lambdaNode->GetFunctionArguments()[0]);

  auto gammaNode = jlm::rvsdg::GammaNode::create(match, 3);
  auto gammaInput1 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto gammaInput2 = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[2]);
  auto gammaOutput = gammaNode->AddExitVar({ gammaInput1.branchArgument[0],
                                             gammaInput1.branchArgument[1],
                                             gammaInput2.branchArgument[2] });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });
  jlm::llvm::GraphExport::Create(*lambdaOutput, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const function_node &>(*ipg.begin()).cfg();
  assert(is_closed(*cfg));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-EmptyGammaWithThreeSubregions",
    EmptyGammaWithThreeSubregions)

static int
PartialEmptyGamma()
{
  using namespace jlm::llvm;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::bittype::Create(1), valueType },
      { valueType });

  RvsdgModule rvsdgModule(filepath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "lambdaOutput", linkage::external_linkage));

  auto match = jlm::rvsdg::match(1, { { 0, 0 } }, 1, 2, lambdaNode->GetFunctionArguments()[0]);
  auto gammaNode = jlm::rvsdg::GammaNode::create(match, 2);
  auto gammaInput = gammaNode->AddEntryVar(lambdaNode->GetFunctionArguments()[1]);
  auto output = jlm::tests::create_testop(
      gammaNode->subregion(1),
      { gammaInput.branchArgument[1] },
      { valueType })[0];
  auto gammaOutput = gammaNode->AddExitVar({ gammaInput.branchArgument[0], output });

  auto lambdaOutput = lambdaNode->finalize({ gammaOutput.output });

  jlm::llvm::GraphExport::Create(*lambdaOutput, "");

  jlm::rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);

  // Assert
  auto & ipg = module->ipgraph();
  assert(ipg.nnodes() == 1);

  auto cfg = dynamic_cast<const function_node &>(*ipg.begin()).cfg();
  std::cout << cfg::ToAscii(*cfg) << std::flush;

  assert(!is_proper_structured(*cfg));
  assert(is_structured(*cfg));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-PartialEmptyGamma",
    PartialEmptyGamma)

static int
RecursiveData()
{
  using namespace jlm::llvm;

  // Arrange
  auto vt = jlm::tests::valuetype::Create();
  auto pt = PointerType::Create();

  RvsdgModule rm(jlm::util::filepath(""), "", "");

  auto imp = &GraphImport::Create(rm.Rvsdg(), vt, pt, "", linkage::external_linkage);

  phi::builder pb;
  pb.begin(&rm.Rvsdg().GetRootRegion());
  auto region = pb.subregion();
  auto r1 = pb.add_recvar(pt);
  auto r2 = pb.add_recvar(pt);
  auto dep = pb.add_ctxvar(imp);

  jlm::rvsdg::output *delta1, *delta2;
  {
    auto delta =
        delta::node::Create(region, vt, "test-delta1", linkage::external_linkage, "", false);
    auto dep1 = delta->add_ctxvar(r2->argument());
    auto dep2 = delta->add_ctxvar(dep);
    delta1 =
        delta->finalize(jlm::tests::create_testop(delta->subregion(), { dep1, dep2 }, { vt })[0]);
  }

  {
    auto delta =
        delta::node::Create(region, vt, "test-delta2", linkage::external_linkage, "", false);
    auto dep1 = delta->add_ctxvar(r1->argument());
    auto dep2 = delta->add_ctxvar(dep);
    delta2 =
        delta->finalize(jlm::tests::create_testop(delta->subregion(), { dep1, dep2 }, { vt })[0]);
  }

  r1->set_rvorigin(delta1);
  r2->set_rvorigin(delta2);

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

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/tests/jlm/llvm/backend/RvsdgToIpGraphConverterTests-RecursiveData",
    RecursiveData)
