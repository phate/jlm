/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rvsdg2rhls/add-forks.hpp>
#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/view.hpp>

TEST(ForkInsertionTests, ForkInsertion)
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Arrange
  auto bit32Type = rvsdg::BitType::Create(32);
  const auto functionType = jlm::rvsdg::FunctionType::Create(
      { bit32Type, bit32Type, bit32Type },
      { bit32Type, bit32Type, bit32Type });

  LlvmRvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto loop = hls::LoopNode::create(lambda->subregion());
  rvsdg::Output * idvBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[0], &idvBuffer);
  rvsdg::Output * lvsBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[1], &lvsBuffer);
  rvsdg::Output * lveBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[2], &lveBuffer);

  auto arm = rvsdg::CreateOpNode<rvsdg::bitadd_op>({ idvBuffer, lvsBuffer }, 32).output(0);
  auto cmp = rvsdg::CreateOpNode<rvsdg::bitult_op>({ arm, lveBuffer }, 32).output(0);
  auto & matchNode = rvsdg::MatchOperation::CreateNode(*cmp, { { 1, 1 } }, 0, 2);

  loop->set_predicate(matchNode.output(0));

  auto lambdaOutput = lambda->finalize({ loop->output(0), loop->output(1), loop->output(2) });
  rvsdg::GraphExport::Create(*lambdaOutput, "");

  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  util::StatisticsCollector statisticsCollector;
  hls::ForkInsertion::CreateAndRun(rvsdgModule, statisticsCollector);
  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  {
    EXPECT_EQ(rootRegion.numNodes(), 1);
    auto lambda = util::assertedCast<jlm::rvsdg::LambdaNode>(rootRegion.Nodes().begin().ptr());
    EXPECT_NE(dynamic_cast<const jlm::rvsdg::LambdaNode *>(lambda), nullptr);

    auto lambdaSubregion = lambda->subregion();
    EXPECT_EQ(lambdaSubregion->numNodes(), 1);
    auto loop = util::assertedCast<hls::LoopNode>(lambdaSubregion->Nodes().begin().ptr());
    EXPECT_NE(dynamic_cast<const hls::LoopNode *>(loop), nullptr);

    auto [forkNode, forkOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<hls::ForkOperation>(
        *loop->subregion()->result(0)->origin());
    EXPECT_TRUE(forkNode && forkOperation);
    EXPECT_EQ(forkNode->ninputs(), 1);
    EXPECT_EQ(forkNode->noutputs(), 4);
    EXPECT_FALSE(forkOperation->IsConstant());
  }
}

TEST(SinkInsertionTests, ConstantForkInsertion)
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Arrange
  auto bit32Type = rvsdg::BitType::Create(32);
  const auto functionType = rvsdg::FunctionType::Create({ bit32Type }, { bit32Type });

  LlvmRvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();

  auto lambda = rvsdg::LambdaNode::Create(
      rootRegion,
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));

  auto loop = hls::LoopNode::create(lambda->subregion());
  auto subregion = loop->subregion();
  rvsdg::Output * idvBuffer = nullptr;
  loop->AddLoopVar(lambda->GetFunctionArguments()[0], &idvBuffer);
  auto bitConstant1 = &rvsdg::BitConstantOperation::create(*subregion, { 32, 1 });

  auto arm = rvsdg::CreateOpNode<rvsdg::bitadd_op>({ idvBuffer, bitConstant1 }, 32).output(0);
  auto cmp = rvsdg::CreateOpNode<rvsdg::bitult_op>({ arm, bitConstant1 }, 32).output(0);
  auto & matchNode = rvsdg::MatchOperation::CreateNode(*cmp, { { 1, 1 } }, 0, 2);

  loop->set_predicate(matchNode.output(0));

  auto lambdaOutput = lambda->finalize({ loop->output(0) });
  rvsdg::GraphExport::Create(*lambdaOutput, "");

  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Act
  util::StatisticsCollector statisticsCollector;
  hls::ForkInsertion::CreateAndRun(rvsdgModule, statisticsCollector);
  rvsdg::view(rvsdgModule.Rvsdg(), stdout);

  // Assert
  {
    EXPECT_EQ(rootRegion.numNodes(), 1);
    auto lambda = util::assertedCast<jlm::rvsdg::LambdaNode>(rootRegion.Nodes().begin().ptr());
    EXPECT_TRUE(rvsdg::is<jlm::rvsdg::LambdaOperation>(lambda));

    auto lambdaRegion = lambda->subregion();
    EXPECT_EQ(lambdaRegion->numNodes(), 1);

    const rvsdg::NodeOutput * loopOutput =
        dynamic_cast<jlm::rvsdg::NodeOutput *>(lambdaRegion->result(0)->origin());
    EXPECT_NE(loopOutput, nullptr);
    auto loopNode = loopOutput->node();
    EXPECT_TRUE(rvsdg::is<hls::LoopOperation>(loopNode));
    auto loop = util::assertedCast<hls::LoopNode>(loopNode);

    auto [forkNode, forkOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<hls::ForkOperation>(
        *loop->subregion()->result(0)->origin());
    EXPECT_TRUE(forkNode && forkOperation);
    EXPECT_EQ(forkNode->ninputs(), 1);
    EXPECT_EQ(forkNode->noutputs(), 2);
    EXPECT_FALSE(forkOperation->IsConstant());

    auto matchNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*forkNode->input(0)->origin());
    auto bitsUltNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*matchNode->input(0)->origin());
    auto [cForkNode, cForkOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<hls::ForkOperation>(*bitsUltNode->input(1)->origin());
    EXPECT_EQ(cForkNode->ninputs(), 1);
    EXPECT_EQ(cForkNode->noutputs(), 2);
    EXPECT_TRUE(cForkOperation->IsConstant());
  }
}
