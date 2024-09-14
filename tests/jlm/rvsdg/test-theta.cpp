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

  jlm::rvsdg::graph graph;
  auto t = jlm::tests::valuetype::Create();

  auto imp1 = &jlm::tests::GraphImport::Create(graph, ctltype::Create(2), "imp1");
  auto imp2 = &jlm::tests::GraphImport::Create(graph, t, "imp2");
  auto imp3 = &jlm::tests::GraphImport::Create(graph, t, "imp3");

  auto theta = jlm::rvsdg::ThetaNode::create(graph.root());

  auto lv1 = theta->add_loopvar(imp1);
  auto lv2 = theta->add_loopvar(imp2);
  auto lv3 = theta->add_loopvar(imp3);

  lv2->result()->divert_to(lv3->argument());
  lv3->result()->divert_to(lv3->argument());
  theta->set_predicate(lv1->argument());

  jlm::tests::GraphExport::Create(*theta->output(0), "exp");
  auto theta2 =
      static_cast<jlm::rvsdg::structural_node *>(theta)->copy(graph.root(), { imp1, imp2, imp3 });
  jlm::rvsdg::view(graph.root(), stdout);

  assert(lv1->node() == theta);
  assert(lv2->node() == theta);
  assert(lv3->node() == theta);

  assert(theta->predicate() == theta->subregion()->result(0));
  assert(theta->nloopvars() == 3);
  assert((*theta->begin())->result() == theta->subregion()->result(1));

  assert(dynamic_cast<const jlm::rvsdg::ThetaNode *>(theta2));
}

static void
TestRemoveThetaOutputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ctltype::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(rvsdg.root());

  auto thetaOutput0 = thetaNode->add_loopvar(ctl);
  auto thetaOutput1 = thetaNode->add_loopvar(x);
  auto thetaOutput2 = thetaNode->add_loopvar(y);
  thetaNode->set_predicate(thetaOutput0->argument());

  jlm::tests::GraphExport::Create(*thetaOutput0, "");

  // Act & Assert
  auto deadInputs = thetaNode->RemoveThetaOutputsWhere(
      [&](const theta_output & output)
      {
        return output.index() == thetaOutput1->index();
      });
  assert(deadInputs.Size() == 1);
  assert(deadInputs.Contains(thetaNode->input(1)));
  assert(thetaNode->noutputs() == 2);
  assert(thetaNode->subregion()->nresults() == 3);
  assert(thetaOutput0->index() == 0);
  assert(thetaOutput0->result()->index() == 1);
  assert(thetaOutput2->index() == 1);
  assert(thetaOutput2->result()->index() == 2);

  deadInputs = thetaNode->RemoveThetaOutputsWhere(
      [](const theta_output &)
      {
        return true;
      });
  assert(deadInputs.Size() == 1);
  assert(deadInputs.Contains(thetaNode->input(2)));
  assert(thetaNode->noutputs() == 1);
  assert(thetaNode->subregion()->nresults() == 2);
  assert(thetaOutput0->index() == 0);
  assert(thetaOutput0->result()->index() == 1);
}

static void
TestPruneThetaOutputs()
{
  using namespace jlm::rvsdg;

  // Arrange
  graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ctltype::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(rvsdg.root());

  auto thetaOutput0 = thetaNode->add_loopvar(ctl);
  thetaNode->add_loopvar(x);
  thetaNode->add_loopvar(y);
  thetaNode->set_predicate(thetaOutput0->argument());

  jlm::tests::GraphExport::Create(*thetaOutput0, "");

  // Act
  auto deadInputs = thetaNode->PruneThetaOutputs();

  // Assert
  assert(deadInputs.Size() == 2);
  assert(deadInputs.Contains(thetaNode->input(1)));
  assert(deadInputs.Contains(thetaNode->input(2)));
  assert(thetaNode->noutputs() == 1);
  assert(thetaNode->subregion()->nresults() == 2);
  assert(thetaOutput0->index() == 0);
  assert(thetaOutput0->result()->index() == 1);
}

static void
TestRemoveThetaInputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ctltype::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(rvsdg.root());

  auto thetaOutput0 = thetaNode->add_loopvar(ctl);
  auto thetaOutput1 = thetaNode->add_loopvar(x);
  auto thetaOutput2 = thetaNode->add_loopvar(y);
  thetaNode->set_predicate(thetaOutput0->argument());

  auto result =
      jlm::tests::SimpleNode::Create(*thetaNode->subregion(), {}, { valueType }).output(0);

  thetaOutput1->result()->divert_to(result);
  thetaOutput2->result()->divert_to(result);

  jlm::tests::GraphExport::Create(*thetaOutput0, "");

  // Act & Assert
  auto deadOutputs = thetaNode->RemoveThetaInputsWhere(
      [&](const theta_input & input)
      {
        return input.index() == thetaOutput1->input()->index();
      });
  assert(deadOutputs.Size() == 1);
  assert(deadOutputs.Contains(thetaNode->output(1)));
  assert(thetaNode->ninputs() == 2);
  assert(thetaNode->subregion()->narguments() == 2);
  assert(thetaOutput0->input()->index() == 0);
  assert(thetaOutput0->argument()->index() == 0);
  assert(thetaOutput2->input()->index() == 1);
  assert(thetaOutput2->argument()->index() == 1);

  deadOutputs = thetaNode->RemoveThetaInputsWhere(
      [](const theta_input & input)
      {
        return true;
      });
  assert(deadOutputs.Size() == 1);
  assert(deadOutputs.Contains(thetaNode->output(2)));
  assert(thetaNode->ninputs() == 1);
  assert(thetaNode->subregion()->narguments() == 1);
  assert(thetaOutput0->input()->index() == 0);
  assert(thetaOutput0->argument()->index() == 0);
}

static void
TestPruneThetaInputs()
{
  using namespace jlm::rvsdg;

  // Arrange
  graph rvsdg;
  auto valueType = jlm::tests::valuetype::Create();

  auto ctl = &jlm::tests::GraphImport::Create(rvsdg, ctltype::Create(2), "ctl");
  auto x = &jlm::tests::GraphImport::Create(rvsdg, valueType, "x");
  auto y = &jlm::tests::GraphImport::Create(rvsdg, valueType, "y");

  auto thetaNode = ThetaNode::create(rvsdg.root());

  auto thetaOutput0 = thetaNode->add_loopvar(ctl);
  auto thetaOutput1 = thetaNode->add_loopvar(x);
  auto thetaOutput2 = thetaNode->add_loopvar(y);
  thetaNode->set_predicate(thetaOutput0->argument());

  auto result =
      jlm::tests::SimpleNode::Create(*thetaNode->subregion(), {}, { valueType }).output(0);

  thetaOutput1->result()->divert_to(result);
  thetaOutput2->result()->divert_to(result);

  jlm::tests::GraphExport::Create(*thetaOutput0, "");

  // Act
  auto deadOutputs = thetaNode->PruneThetaInputs();

  // Assert
  assert(deadOutputs.Size() == 2);
  assert(deadOutputs.Contains(thetaNode->output(1)));
  assert(deadOutputs.Contains(thetaNode->output(2)));
  assert(thetaNode->ninputs() == 1);
  assert(thetaNode->subregion()->narguments() == 1);
  assert(thetaOutput0->input()->index() == 0);
  assert(thetaOutput0->argument()->index() == 0);
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
