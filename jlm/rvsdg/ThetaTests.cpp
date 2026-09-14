/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>
#include <jlm/rvsdg/TestOperations.hpp>

#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::rvsdg
{

TEST(ThetaTests, TestThetaCreation)
{
  using namespace jlm::rvsdg;

  Graph graph;
  auto t = TestType::createValueType();

  auto imp1 = &jlm::rvsdg::GraphImport::Create(graph, ControlType::Create(2), "imp1");
  auto imp2 = &jlm::rvsdg::GraphImport::Create(graph, t, "imp2");
  auto imp3 = &jlm::rvsdg::GraphImport::Create(graph, t, "imp3");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lv1 = theta->AddLoopVar(imp1);
  auto lv2 = theta->AddLoopVar(imp2);
  auto lv3 = theta->AddLoopVar(imp3);

  lv2.post->divert_to(lv3.pre);
  lv3.post->divert_to(lv3.pre);
  theta->set_predicate(lv1.pre);

  GraphExport::Create(*theta->output(0), "exp");
  auto theta2 = static_cast<jlm::rvsdg::StructuralNode *>(theta)->copy(
      &graph.GetRootRegion(),
      { imp1, imp2, imp3 });
  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  EXPECT_EQ(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*lv1.output), theta);
  EXPECT_EQ(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*lv2.output), theta);
  EXPECT_EQ(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*lv3.output), theta);

  EXPECT_EQ(theta->predicate(), theta->subregion()->result(0));
  EXPECT_EQ(theta->GetLoopVars().size(), 3u);
  EXPECT_EQ(theta->GetLoopVars()[0].post, theta->subregion()->result(1));

  EXPECT_NE(dynamic_cast<const jlm::rvsdg::ThetaNode *>(theta2), nullptr);
}

TEST(ThetaTests, TestThetaLoopVarRemoval)
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto valueType = TestType::createValueType();

  auto ctl = &jlm::rvsdg::GraphImport::Create(rvsdg, ControlType::Create(2), "ctl");
  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto lv0 = thetaNode->AddLoopVar(ctl);
  auto lv1 = thetaNode->AddLoopVar(x);
  auto lv2 = thetaNode->AddLoopVar(y);
  thetaNode->set_predicate(lv0.pre);

  GraphExport::Create(*lv0.output, "");

  // Act & Assert
  thetaNode->RemoveLoopVars({ lv1 });
  auto loopvars = thetaNode->GetLoopVars();
  EXPECT_EQ(loopvars.size(), 2u);
  EXPECT_EQ(loopvars[0].input, lv0.input);
  EXPECT_EQ(loopvars[0].pre, lv0.pre);
  EXPECT_EQ(loopvars[0].post, lv0.post);
  EXPECT_EQ(loopvars[0].output, lv0.output);
  EXPECT_EQ(loopvars[1].input, lv2.input);
  EXPECT_EQ(loopvars[1].pre, lv2.pre);
  EXPECT_EQ(loopvars[1].post, lv2.post);
  EXPECT_EQ(loopvars[1].output, lv2.output);
}

TEST(ThetaTests, reduceStaticallyKnownPredicate)
{
  // Arrange
  Graph rvsdg;
  auto valueType = TestType::createValueType();

  auto x = &GraphImport::Create(rvsdg, valueType, "x");
  auto y = &GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());
  auto loopVar1 = thetaNode->AddLoopVar(x);
  auto loopVar2 = thetaNode->AddLoopVar(y);

  auto testNode1 = TestOperation::createNode(
      thetaNode->subregion(),
      { loopVar1.pre, loopVar2.pre },
      { valueType });
  auto testNode2 =
      TestOperation::createNode(thetaNode->subregion(), { loopVar2.pre }, { valueType });

  auto & ctlConstant = ControlConstantOperation::createFalse(*thetaNode->subregion());

  thetaNode->set_predicate(&ctlConstant);
  loopVar1.post->divert_to(testNode1->output(0));
  loopVar2.post->divert_to(testNode2->output(0));

  auto & x1 = GraphExport::Create(*loopVar1.output, "");
  auto & x2 = GraphExport::Create(*loopVar2.output, "");

  // Act
  ThetaNode::reduceStaticallyKnownPredicate(*thetaNode);

  // Assert
  EXPECT_FALSE(Region::containsNodeType<ThetaNode>(rvsdg.GetRootRegion(), false));
  EXPECT_EQ(rvsdg.GetRootRegion().numNodes(), 3u);

  {
    auto [node, operation] = TryGetSimpleNodeAndOptionalOp<TestOperation>(*x1.origin());
    EXPECT_NE(operation, nullptr);
    EXPECT_EQ(node->ninputs(), 2u);
    EXPECT_EQ(node->input(0)->origin(), x);
    EXPECT_EQ(node->input(1)->origin(), y);
  }

  {
    auto [node, operation] = TryGetSimpleNodeAndOptionalOp<TestOperation>(*x2.origin());
    EXPECT_NE(operation, nullptr);
    EXPECT_EQ(node->ninputs(), 1u);
    EXPECT_EQ(node->input(0)->origin(), y);
  }
}

}
