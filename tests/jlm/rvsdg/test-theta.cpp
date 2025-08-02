/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestThetaCreation()
{
  using namespace jlm::rvsdg;

  Graph graph;
  auto t = jlm::tests::ValueType::Create();

  auto imp1 = &jlm::tests::GraphImport::Create(graph, ControlType::Create(2), "imp1");
  auto imp2 = &jlm::tests::GraphImport::Create(graph, t, "imp2");
  auto imp3 = &jlm::tests::GraphImport::Create(graph, t, "imp3");

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

  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*lv1.output) == theta);
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*lv2.output) == theta);
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*lv3.output) == theta);

  assert(theta->predicate() == theta->subregion()->result(0));
  assert(theta->GetLoopVars().size() == 3);
  assert(theta->GetLoopVars()[0].post == theta->subregion()->result(1));

  assert(dynamic_cast<const jlm::rvsdg::ThetaNode *>(theta2));
}

static void
TestThetaLoopVarRemoval()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto valueType = jlm::tests::ValueType::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto lv0 = thetaNode->AddLoopVar(ctl);
  auto lv1 = thetaNode->AddLoopVar(x);
  auto lv2 = thetaNode->AddLoopVar(y);
  thetaNode->set_predicate(lv0.pre);

  GraphExport::Create(*lv0.output, "");

  // Act & Assert
  thetaNode->RemoveLoopVars({ lv1 });
  auto loopvars = thetaNode->GetLoopVars();
  assert(loopvars.size() == 2);
  assert(loopvars[0].input = lv0.input);
  assert(loopvars[0].pre = lv0.pre);
  assert(loopvars[0].post = lv0.post);
  assert(loopvars[0].output = lv0.output);
  assert(loopvars[1].input = lv2.input);
  assert(loopvars[1].pre = lv2.pre);
  assert(loopvars[1].post = lv2.post);
  assert(loopvars[1].output = lv2.output);
}

static void
TestTheta()
{
  TestThetaCreation();
  TestThetaLoopVarRemoval();
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-theta", TestTheta)
