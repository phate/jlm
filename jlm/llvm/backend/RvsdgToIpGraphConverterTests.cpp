/*
 * Copyright 2018, 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/backend/RvsdgToIpGraphConverter.hpp>
#include <jlm/llvm/ir/CallingConvention.hpp>
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

TEST(RvsdgToIpGraphConverterTests, GammaWithMatch)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(1), valueType, valueType },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode =
      MatchOperation::CreateNode(*lambdaNode->GetFunctionArguments()[0], { { 0, 0 } }, 1, 2);
  auto gamma = GammaNode::create(matchNode.output(0), 2);
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
  EXPECT_EQ(ipg.nnodes(), 1u);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  EXPECT_EQ(cfg->nnodes(), 4u);
}

TEST(RvsdgToIpGraphConverterTests, GammaWithoutMatch)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::ControlType::Create(2), valueType, valueType },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

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
  EXPECT_EQ(ipg.nnodes(), 1u);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  EXPECT_EQ(cfg->nnodes(), 4u);
}

TEST(RvsdgToIpGraphConverterTests, EmptyGammaWithTwoSubregionsAndMatch)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  const auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  const auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  const auto conditionValue = lambdaNode->GetFunctionArguments()[0];
  const auto trueValue = lambdaNode->GetFunctionArguments()[1];
  const auto falseValue = lambdaNode->GetFunctionArguments()[2];

  auto caseValue = 24;
  auto & matchNode = MatchOperation::CreateNode(*conditionValue, { { caseValue, 0 } }, 1, 2);

  const auto gammaNode = jlm::rvsdg::GammaNode::create(matchNode.output(0), 2);
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
  EXPECT_EQ(ipGraph.nnodes(), 1u);

  const auto controlFlowGraph = dynamic_cast<const FunctionNode &>(*ipGraph.begin()).cfg();
  EXPECT_TRUE(is_closed(*controlFlowGraph));
}

TEST(RvsdgToIpGraphConverterTests, EmptyGammaWithTwoSubregions)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  const auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  const auto trueValue = lambdaNode->GetFunctionArguments()[1];
  const auto falseValue = lambdaNode->GetFunctionArguments()[2];

  auto & matchNode =
      MatchOperation::CreateNode(*lambdaNode->GetFunctionArguments()[0], { { 0, 0 } }, 1, 2);

  const auto gammaNode0 = jlm::rvsdg::GammaNode::create(matchNode.output(0), 2);
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
  EXPECT_EQ(ipGraph.nnodes(), 1u);

  const auto controlFlowGraph = dynamic_cast<const FunctionNode &>(*ipGraph.begin()).cfg();
  EXPECT_TRUE(is_closed(*controlFlowGraph));
}

TEST(RvsdgToIpGraphConverterTests, EmptyGammaWithThreeSubregions)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32), valueType, valueType },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode = MatchOperation::CreateNode(
      *lambdaNode->GetFunctionArguments()[0],
      { { 0, 0 }, { 1, 1 } },
      2,
      3);

  auto gammaNode = jlm::rvsdg::GammaNode::create(matchNode.output(0), 3);
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
  EXPECT_EQ(ipg.nnodes(), 1u);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  EXPECT_TRUE(is_closed(*cfg));
}

TEST(RvsdgToIpGraphConverterTests, PartialEmptyGamma)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(1), valueType },
      { valueType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode =
      MatchOperation::CreateNode(*lambdaNode->GetFunctionArguments()[0], { { 0, 0 } }, 1, 2);
  auto gammaNode = GammaNode::create(matchNode.output(0), 2);
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
  EXPECT_EQ(ipg.nnodes(), 1u);

  auto cfg = dynamic_cast<const FunctionNode &>(*ipg.begin()).cfg();
  std::cout << ControlFlowGraph::ToAscii(*cfg) << std::flush;

  EXPECT_TRUE(is_proper_structured(*cfg));
}

TEST(RvsdgToIpGraphConverterTests, RecursiveData)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto vt = TestType::createValueType();
  auto pt = PointerType::Create();

  LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");

  auto imp = &LlvmGraphImport::createGlobalImport(
      rm.Rvsdg(),
      vt,
      pt,
      "import",
      Linkage::externalLinkage,
      false,
      4);

  PhiBuilder phiBuilder;
  phiBuilder.begin(&rm.Rvsdg().GetRootRegion());
  auto region = phiBuilder.subregion();
  auto fixVar1 = phiBuilder.AddFixVar(pt);
  auto fixVar2 = phiBuilder.AddFixVar(pt);
  auto dep = phiBuilder.AddContextVar(*imp);

  Output *delta1 = nullptr, *delta2 = nullptr;
  {
    auto delta = DeltaNode::Create(
        region,
        jlm::llvm::DeltaOperation::Create(vt, "delta1", Linkage::externalLinkage, "", false, 4));
    auto dep1 = delta->AddContextVar(*fixVar2.recref).inner;
    auto dep2 = delta->AddContextVar(*dep.inner).inner;
    delta1 = &delta->finalize(
        TestOperation::createNode(delta->subregion(), { dep1, dep2 }, { vt })->output(0));
  }

  {
    auto delta = DeltaNode::Create(
        region,
        jlm::llvm::DeltaOperation::Create(vt, "delta2", Linkage::externalLinkage, "", false, 4));
    auto dep1 = delta->AddContextVar(*fixVar1.recref).inner;
    auto dep2 = delta->AddContextVar(*dep.inner).inner;
    delta2 = &delta->finalize(
        TestOperation::createNode(delta->subregion(), { dep1, dep2 }, { vt })->output(0));
  }

  fixVar1.result->divert_to(delta1);
  fixVar2.result->divert_to(delta2);

  auto phi = phiBuilder.end();
  GraphExport::Create(*phi->output(0), "");

  view(rm.Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  auto module = RvsdgToIpGraphConverter::CreateAndConvertModule(rm, statisticsCollector);
  print(*module, stdout);

  // Assert
  auto & ipGraph = module->ipgraph();
  EXPECT_EQ(ipGraph.nnodes(), 3u);

  auto delta1Node = ipGraph.find("delta1");
  auto delta2Node = ipGraph.find("delta2");
  auto importNode = ipGraph.find("import");
  EXPECT_EQ(delta1Node->numDependencies(), 2u);
  for (auto depNode : *delta1Node)
  {
    EXPECT_TRUE(depNode == delta2Node || depNode == importNode);
  }

  EXPECT_EQ(delta2Node->numDependencies(), 2u);
  for (auto depNode : *delta2Node)
  {
    EXPECT_TRUE(depNode == delta1Node || depNode == importNode);
  }

  EXPECT_EQ(importNode->numDependencies(), 0u);
}

static size_t
numSsaPhiOperations(const jlm::llvm::BasicBlock & basicBlock)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  size_t numSsaPhiOperations = 0;
  for (const auto tac : basicBlock.tacs())
  {
    if (is<SsaPhiOperation>(tac))
      numSsaPhiOperations++;
  }

  return numSsaPhiOperations;
}

TEST(RvsdgToIpGraphConverterTests, NestedLoopWithCall)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto functionType =
      FunctionType::Create({ ioStateType, memoryStateType }, { ioStateType, memoryStateType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto & opaque = LlvmGraphImport::createFunctionImport(
      rvsdg,
      functionType,
      "opaque",
      Linkage::externalLinkage,
      CallingConvention::Default);

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto ioStateArgument = lambdaNode->GetFunctionArguments()[0];
  auto memoryStateArgument = lambdaNode->GetFunctionArguments()[1];
  auto opaqueCtxVar = lambdaNode->AddContextVar(opaque);

  auto outerThetaNode = ThetaNode::create(lambdaNode->subregion());
  auto outerIOStateLoopVar = outerThetaNode->AddLoopVar(ioStateArgument);
  auto outerMemoryStateLoopVar = outerThetaNode->AddLoopVar(memoryStateArgument);
  auto outerOpaque = outerThetaNode->AddLoopVar(opaqueCtxVar.inner);

  auto innerThetaNode = ThetaNode::create(outerThetaNode->subregion());
  auto innerIOStateLoopVar = innerThetaNode->AddLoopVar(outerIOStateLoopVar.pre);
  auto innerMemoryStateLoopVar = innerThetaNode->AddLoopVar(outerMemoryStateLoopVar.pre);
  auto innerOpaque = innerThetaNode->AddLoopVar(outerOpaque.pre);

  auto & callNode = CallOperation::CreateNode(
      innerOpaque.pre,
      functionType,
      { innerIOStateLoopVar.pre, innerMemoryStateLoopVar.pre });

  innerIOStateLoopVar.post->divert_to(callNode.output(0));
  innerMemoryStateLoopVar.post->divert_to(callNode.output(1));

  outerIOStateLoopVar.post->divert_to(innerIOStateLoopVar.output);
  outerMemoryStateLoopVar.post->divert_to(innerMemoryStateLoopVar.output);
  outerOpaque.post->divert_to(innerOpaque.output);

  auto lambdaOutput =
      lambdaNode->finalize({ outerIOStateLoopVar.output, outerMemoryStateLoopVar.output });

  GraphExport::Create(*lambdaOutput, "f");

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  auto ipGraphModule =
      RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);

  print(*ipGraphModule, stdout);

  // Assert
  // We expect that we only create SSA phi operations for the IO and memory state edges, but NOT
  // for the context variable in the inner loop. Ultimately, this means that every basic block
  // should either have zero or two SSA phi operations.
  auto & ipGraph = ipGraphModule->ipgraph();
  EXPECT_EQ(ipGraph.nnodes(), 2u);

  auto functionNode = ipGraph.find("f");
  auto importNode = ipGraph.find("opaque");
  EXPECT_EQ(functionNode->numDependencies(), 1u);
  EXPECT_EQ(importNode->numDependencies(), 0u);
  EXPECT_EQ(*functionNode->begin(), importNode);

  auto controlFlowGraph = dynamic_cast<const FunctionNode *>(ipGraph.find("f"))->cfg();
  EXPECT_EQ(controlFlowGraph->nnodes(), 5u);

  for (auto & basicBlock : *controlFlowGraph)
  {
    auto numSsaPhis = numSsaPhiOperations(basicBlock);
    EXPECT_TRUE(numSsaPhis == 2 || numSsaPhis == 0);
  }
}

class DataImportConversionTest : public testing::TestWithParam<std::tuple<
                                     std::shared_ptr<const jlm::rvsdg::Type>,
                                     std::string,
                                     jlm::llvm::Linkage,
                                     bool,
                                     size_t>>
{
};

TEST_P(DataImportConversionTest, Test)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto [valueType, name, linkage, isConstant, alignment] = GetParam();

  const auto pointerType = PointerType::Create();

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");
  [[maybe_unused]] auto & import = LlvmGraphImport::createGlobalImport(
      rvsdgModule.Rvsdg(),
      valueType,
      pointerType,
      name,
      linkage,
      isConstant,
      alignment);

  view(rvsdgModule.Rvsdg(), stdout);

  // Act
  StatisticsCollector statisticsCollector;
  const auto ipGraphModule =
      RvsdgToIpGraphConverter::CreateAndConvertModule(rvsdgModule, statisticsCollector);

  print(*ipGraphModule, stdout);

  // Assert
  const auto & ipGraph = ipGraphModule->ipgraph();
  EXPECT_EQ(ipGraph.nnodes(), 1u);

  const auto dataNode = dynamic_cast<const DataNode *>(&*ipGraph.begin());
  EXPECT_NE(dataNode, nullptr);

  EXPECT_EQ(dataNode->GetValueType(), valueType);
  EXPECT_EQ(dataNode->name(), name);
  EXPECT_EQ(dataNode->linkage(), linkage);
  EXPECT_EQ(dataNode->constant(), isConstant);
  EXPECT_EQ(dataNode->getAlignment(), alignment);
}

INSTANTIATE_TEST_SUITE_P(
    Test1,
    DataImportConversionTest,
    ::testing::Values(std::make_tuple(
        jlm::rvsdg::TestType::createValueType(),
        "name",
        jlm::llvm::Linkage::externalLinkage,
        false,
        4)));
INSTANTIATE_TEST_SUITE_P(
    Test2,
    DataImportConversionTest,
    ::testing::Values(std::make_tuple(
        jlm::rvsdg::TestType::createValueType(),
        "name",
        jlm::llvm::Linkage::externalLinkage,
        true,
        8)));
INSTANTIATE_TEST_SUITE_P(
    Test3,
    DataImportConversionTest,
    ::testing::Values(std::make_tuple(
        jlm::rvsdg::TestType::createValueType(),
        "foo",
        jlm::llvm::Linkage::privateLinkage,
        false,
        1)));
