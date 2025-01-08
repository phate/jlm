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
  auto t = jlm::tests::valuetype::Create();

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

  jlm::tests::GraphExport::Create(*theta->output(0), "exp");
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
TestRemoveThetaOutputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto thetaOutput0 = thetaNode->AddLoopVar(ctl);
  auto thetaOutput1 = thetaNode->AddLoopVar(x);
  auto thetaOutput2 = thetaNode->AddLoopVar(y);
  thetaNode->set_predicate(thetaOutput0.pre);

  jlm::tests::GraphExport::Create(*thetaOutput0.output, "");

  // Act & Assert
  auto deadInputs = thetaNode->RemoveThetaOutputsWhere(
      [&](const jlm::rvsdg::output & output)
      {
        return output.index() == thetaOutput1.output->index();
      });
  assert(deadInputs.Size() == 1);
  assert(deadInputs.Contains(thetaNode->input(1)));
  assert(thetaNode->noutputs() == 2);
  assert(thetaNode->subregion()->nresults() == 3);
  assert(thetaOutput0.output->index() == 0);
  assert(thetaOutput0.post->index() == 1);
  assert(thetaOutput2.output->index() == 1);
  assert(thetaOutput2.post->index() == 2);

  deadInputs = thetaNode->RemoveThetaOutputsWhere(
      [](const jlm::rvsdg::output &)
      {
        return true;
      });
  assert(deadInputs.Size() == 1);
  assert(deadInputs.Contains(thetaNode->input(2)));
  assert(thetaNode->noutputs() == 1);
  assert(thetaNode->subregion()->nresults() == 2);
  assert(thetaOutput0.output->index() == 0);
  assert(thetaOutput0.post->index() == 1);
}

static void
TestPruneThetaOutputs()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto thetaOutput0 = thetaNode->AddLoopVar(ctl);
  thetaNode->AddLoopVar(x);
  thetaNode->AddLoopVar(y);
  thetaNode->set_predicate(thetaOutput0.pre);

  jlm::tests::GraphExport::Create(*thetaOutput0.output, "");

  // Act
  auto deadInputs = thetaNode->PruneThetaOutputs();

  // Assert
  assert(deadInputs.Size() == 2);
  assert(deadInputs.Contains(thetaNode->input(1)));
  assert(deadInputs.Contains(thetaNode->input(2)));
  assert(thetaNode->noutputs() == 1);
  assert(thetaNode->subregion()->nresults() == 2);
  assert(thetaOutput0.output->index() == 0);
  assert(thetaOutput0.post->index() == 1);
}

static void
TestRemoveThetaInputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto thetaOutput0 = thetaNode->AddLoopVar(ctl);
  auto thetaOutput1 = thetaNode->AddLoopVar(x);
  auto thetaOutput2 = thetaNode->AddLoopVar(y);
  thetaNode->set_predicate(thetaOutput0.pre);

  auto result =
      jlm::tests::SimpleNode::Create(*thetaNode->subregion(), {}, { valueType }).output(0);

  thetaOutput1.post->divert_to(result);
  thetaOutput2.post->divert_to(result);

  jlm::tests::GraphExport::Create(*thetaOutput0.output, "");

  // Act & Assert
  auto deadOutputs = thetaNode->RemoveThetaInputsWhere(
      [&](const jlm::rvsdg::input & input)
      {
        return input.index() == thetaOutput1.input->index();
      });
  assert(deadOutputs.Size() == 1);
  assert(deadOutputs.Contains(thetaNode->output(1)));
  assert(thetaNode->ninputs() == 2);
  assert(thetaNode->subregion()->narguments() == 2);
  assert(thetaOutput0.input->index() == 0);
  assert(thetaOutput0.pre->index() == 0);
  assert(thetaOutput2.input->index() == 1);
  assert(thetaOutput2.pre->index() == 1);

  auto expectDeadOutput = thetaNode->output(2);
  deadOutputs = thetaNode->RemoveThetaInputsWhere(
      [](const jlm::rvsdg::input & /* input */)
      {
        return true;
      });
  assert(deadOutputs.Size() == 1);
  assert(deadOutputs.Contains(expectDeadOutput));
  assert(thetaNode->ninputs() == 1);
  assert(thetaNode->subregion()->narguments() == 1);
  assert(thetaOutput0.input->index() == 0);
  assert(thetaOutput0.pre->index() == 0);
}

static void
TestPruneThetaInputs()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(&rvsdg.GetRootRegion());

  auto thetaOutput0 = thetaNode->AddLoopVar(ctl);
  auto thetaOutput1 = thetaNode->AddLoopVar(x);
  auto thetaOutput2 = thetaNode->AddLoopVar(y);
  thetaNode->set_predicate(thetaOutput0.pre);

  auto result =
      jlm::tests::SimpleNode::Create(*thetaNode->subregion(), {}, { valueType }).output(0);

  thetaOutput1.post->divert_to(result);
  thetaOutput2.post->divert_to(result);

  jlm::tests::GraphExport::Create(*thetaOutput0.output, "");

  // Act
  auto deadOutputs = thetaNode->PruneThetaInputs();

  // Assert
  assert(deadOutputs.Size() == 2);
  assert(deadOutputs.Contains(thetaNode->output(1)));
  assert(deadOutputs.Contains(thetaNode->output(2)));
  assert(thetaNode->ninputs() == 1);
  assert(thetaNode->subregion()->narguments() == 1);
  assert(thetaOutput0.input->index() == 0);
  assert(thetaOutput0.pre->index() == 0);
}

static int
TestTheta()
{
  TestThetaCreation();
  TestRemoveThetaOutputsWhere();
  TestPruneThetaOutputs();
  TestRemoveThetaInputsWhere();
  TestPruneThetaInputs();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-theta", TestTheta)
