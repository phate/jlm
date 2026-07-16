/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 *                Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rvsdg2rhls/GammaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/traverser.hpp>

using namespace jlm::rvsdg;
using namespace jlm::hls;
using namespace jlm::llvm;
using namespace jlm::util;

namespace jlm::hls
{

static void
TestGammaConversion(RvsdgModule & rvsdgModule, size_t expectedMuxCount)
{
  StatisticsCollector statisticsCollector;
  GammaNodeConversion::CreateAndRun(rvsdgModule, statisticsCollector);

  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  JLM_ASSERT(rootRegion.numNodes() == 1);
  auto * lambda = dynamic_cast<LambdaNode *>(&*rootRegion.Nodes().begin());
  JLM_ASSERT(lambda != nullptr);
  EXPECT_FALSE(Region::ContainsOperation<GammaOperation>(rootRegion, true));

  size_t muxCount = 0;
  for (auto & subnode : TopDownTraverser(lambda->subregion()))
  {
    if (is<MuxOperation>(subnode->GetOperation()))
    {
      muxCount++;
    }
  }
  EXPECT_EQ(muxCount, expectedMuxCount);
}

} // namespace

TEST(GammaConversionTests, WithMatchOperation)
{
  auto valueType = TestType::createValueType();
  auto functionType =
      FunctionType::Create({ BitType::Create(1), valueType, valueType }, { valueType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode =
      MatchOperation::CreateNode(*lambda->GetFunctionArguments()[0], { { 0, 0 } }, 1, 2);
  auto gamma = GammaNode::create(matchNode.output(0), 2);
  auto entryVar1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto entryVar2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto exitVar = gamma->AddExitVar({ entryVar1.branchArgument[0], entryVar2.branchArgument[1] });

  auto lambdaOutput = lambda->finalize({ exitVar.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 1);
}

TEST(GammaConversionTests, WithoutMatchOperation)
{
  auto valueType = TestType::createValueType();
  auto functionType =
      FunctionType::Create({ ControlType::Create(2), valueType, valueType }, { valueType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto gamma = GammaNode::create(lambda->GetFunctionArguments()[0], 2);
  auto entryVar1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto entryVar2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto exitVar = gamma->AddExitVar({ entryVar1.branchArgument[0], entryVar2.branchArgument[1] });

  auto lambdaOutput = lambda->finalize({ exitVar.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 1);
}

TEST(GammaConversionTests, NestedGammas)
{
  auto controlType = ControlType::Create(2);
  auto bit32Type = BitType::Create(32);
  auto functionType = FunctionType::Create({ controlType }, { bit32Type });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & outerControlConstant =
      BitConstantOperation::create(*lambda->subregion(), BitValueRepresentation(32, 0));
  auto outerGamma = GammaNode::create(lambda->GetFunctionArguments()[0], 2);
  auto outerVar = outerGamma->AddEntryVar(&outerControlConstant);

  auto & innerControlConstant =
      ControlConstantOperation::create(*outerGamma->subregion(1), ControlValueRepresentation(0, 2));
  auto innerGamma = GammaNode::create(&innerControlConstant, 2);
  auto innerVar = innerGamma->AddEntryVar(outerVar.branchArgument[1]);
  auto innerExit =
      innerGamma->AddExitVar({ innerVar.branchArgument[0], innerVar.branchArgument[1] });

  auto outerExit = outerGamma->AddExitVar({ outerVar.branchArgument[0], innerExit.output });

  auto lambdaOutput = lambda->finalize({ outerExit.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 2);
}

TEST(GammaConversionTests, MuxPredicateMapping)
{
  auto valueType = TestType::createValueType();
  auto bitType = BitType::Create(1);
  auto functionType = FunctionType::Create({ bitType, valueType, valueType }, { valueType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode =
      MatchOperation::CreateNode(*lambda->GetFunctionArguments()[0], { { 0, 0 } }, 1, 2);

  auto gamma = GammaNode::create(matchNode.output(0), 2);
  auto entryVar1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto entryVar2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto exitVar = gamma->AddExitVar({ entryVar1.branchArgument[0], entryVar2.branchArgument[1] });

  auto lambdaOutput = lambda->finalize({ exitVar.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 1);

  for (auto & node : TopDownTraverser(lambda->subregion()))
  {
    if (is<MuxOperation>(node->GetOperation()))
    {
      EXPECT_EQ(node->input(0)->origin(), matchNode.output(0));
    }
  }
}

TEST(GammaConversionTests, MuxAlternativeSelection)
{
  auto valueType = TestType::createValueType();
  auto functionType = FunctionType::Create({ BitType::Create(2), valueType }, { valueType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode = MatchOperation::CreateNode(
      *lambda->GetFunctionArguments()[0],
      { { 0, 0 }, { 1, 1 }, { 2, 2 } },
      3,
      3);

  auto gamma = GammaNode::create(matchNode.output(0), 3);
  auto entryVar1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto exitVar = gamma->AddExitVar(
      { entryVar1.branchArgument[0], entryVar1.branchArgument[1], entryVar1.branchArgument[2] });

  auto lambdaOutput = lambda->finalize({ exitVar.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 1);

  for (auto & node : TopDownTraverser(lambda->subregion()))
  {
    if (is<MuxOperation>(node->GetOperation()))
    {
      auto & muxOp = static_cast<const MuxOperation &>(node->GetOperation());
      EXPECT_EQ(muxOp.narguments(), 4u);
      EXPECT_EQ(node->ninputs(), 4u);
    }
  }
}

TEST(GammaConversionTests, MuxControlPredicateMapping)
{
  auto valueType = TestType::createValueType();
  auto functionType =
      FunctionType::Create({ BitType::Create(1), valueType, valueType }, { valueType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode =
      MatchOperation::CreateNode(*lambda->GetFunctionArguments()[0], { { 0, 0 } }, 1, 2);

  auto gamma = GammaNode::create(matchNode.output(0), 2);
  auto entryVar1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto entryVar2 = gamma->AddEntryVar(lambda->GetFunctionArguments()[2]);
  auto exitVar = gamma->AddExitVar({ entryVar1.branchArgument[0], entryVar2.branchArgument[1] });

  auto lambdaOutput = lambda->finalize({ exitVar.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 1);

  for (auto & node : TopDownTraverser(lambda->subregion()))
  {
    if (is<MuxOperation>(node->GetOperation()))
    {
      EXPECT_EQ(node->ninputs(), 3u);

      auto * muxPredicate = node->input(0)->origin();
      EXPECT_EQ(muxPredicate, matchNode.output(0));
    }
  }
}

TEST(GammaConversionTests, SpeculativeConversionUsesDiscardingMux)
{
  auto valueType = TestType::createValueType();
  auto functionType = FunctionType::Create({ BitType::Create(2), valueType }, { valueType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode = MatchOperation::CreateNode(
      *lambda->GetFunctionArguments()[0],
      { { 0, 0 }, { 1, 1 }, { 2, 2 } },
      3,
      3);

  auto gamma = GammaNode::create(matchNode.output(0), 3);
  auto entryVar1 = gamma->AddEntryVar(lambda->GetFunctionArguments()[1]);
  auto exitVar = gamma->AddExitVar(
      { entryVar1.branchArgument[0], entryVar1.branchArgument[1], entryVar1.branchArgument[2] });

  auto lambdaOutput = lambda->finalize({ exitVar.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 1);

  for (auto & node : TopDownTraverser(lambda->subregion()))
  {
    if (is<MuxOperation>(node->GetOperation()))
    {
      auto & muxOp = static_cast<const MuxOperation &>(node->GetOperation());
      EXPECT_TRUE(muxOp.discarding);
    }
  }
}

TEST(GammaConversionTests, NonSpeculativeModeUsesBranches)
{
  auto valueType = TestType::createValueType();
  auto stateType = TestType::createStateType();
  auto functionType =
      FunctionType::Create({ BitType::Create(2), valueType, valueType, stateType }, { stateType });

  LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");

  auto lambda = LambdaNode::Create(
      rvsdgModule.Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto & matchNode = MatchOperation::CreateNode(
      *lambda->GetFunctionArguments()[0],
      { { 0, 0 }, { 1, 1 }, { 2, 2 } },
      3,
      3);

  auto gamma = GammaNode::create(matchNode.output(0), 3);
  auto stateVar = gamma->AddEntryVar(lambda->GetFunctionArguments()[3]);
  auto stateExit = gamma->AddExitVar(
      { stateVar.branchArgument[0], stateVar.branchArgument[1], stateVar.branchArgument[2] });

  auto lambdaOutput = lambda->finalize({ stateExit.output });
  GraphExport::Create(*lambdaOutput, "");

  TestGammaConversion(rvsdgModule, 1);

  size_t branchCount = 0;
  for (auto & node : TopDownTraverser(lambda->subregion()))
  {
    if (is<BranchOperation>(node->GetOperation()))
    {
      EXPECT_EQ(node->ninputs(), 2u);
      branchCount++;
    }
  }
  EXPECT_GE(branchCount, 1u);
}
