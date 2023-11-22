/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

static void
TestThetaCreation()
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::graph graph;
  jlm::tests::valuetype t;

  auto imp1 = graph.add_import({ ctl2, "imp1" });
  auto imp2 = graph.add_import({ t, "imp2" });
  auto imp3 = graph.add_import({ t, "imp3" });

  auto theta = jlm::rvsdg::theta_node::create(graph.root());

  auto lv1 = theta->add_loopvar(imp1);
  auto lv2 = theta->add_loopvar(imp2);
  auto lv3 = theta->add_loopvar(imp3);

  lv2->result()->divert_to(lv3->argument());
  lv3->result()->divert_to(lv3->argument());
  theta->set_predicate(lv1->argument());

  graph.add_export(theta->output(0), { theta->output(0)->type(), "exp" });
  auto theta2 =
      static_cast<jlm::rvsdg::structural_node *>(theta)->copy(graph.root(), { imp1, imp2, imp3 });
  jlm::rvsdg::view(graph.root(), stdout);

  assert(lv1->node() == theta);
  assert(lv2->node() == theta);
  assert(lv3->node() == theta);

  assert(theta->predicate() == theta->subregion()->result(0));
  assert(theta->nloopvars() == 3);
  assert((*theta->begin())->result() == theta->subregion()->result(1));

  assert(dynamic_cast<const jlm::rvsdg::theta_node *>(theta2));
}

static void
TestRemoveThetaOutputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  graph rvsdg;
  jlm::tests::valuetype valueType;

  auto ctl = rvsdg.add_import({ ctl2, "ctl" });
  auto x = rvsdg.add_import({ valueType, "x" });
  auto y = rvsdg.add_import({ valueType, "y" });

  auto thetaNode = theta_node::create(rvsdg.root());

  auto thetaOutput0 = thetaNode->add_loopvar(ctl);
  auto thetaOutput1 = thetaNode->add_loopvar(x);
  auto thetaOutput2 = thetaNode->add_loopvar(y);
  thetaNode->set_predicate(thetaOutput0->argument());

  rvsdg.add_export(thetaOutput0, { ctl2, "" });

  // Act & Assert
  auto numRemovedOutputs = thetaNode->RemoveThetaOutputsWhere(
      [&](const theta_output & output)
      {
        return output.index() == thetaOutput1->index();
      });
  assert(numRemovedOutputs == 1);
  assert(thetaNode->noutputs() == 2);
  assert(thetaNode->subregion()->nresults() == 3);
  assert(thetaOutput0->index() == 0);
  assert(thetaOutput0->result()->index() == 1);
  assert(thetaOutput2->index() == 1);
  assert(thetaOutput2->result()->index() == 2);

  numRemovedOutputs = thetaNode->RemoveThetaOutputsWhere(
      [](const theta_output &)
      {
        return true;
      });
  assert(numRemovedOutputs == 1);
  assert(thetaNode->noutputs() == 1);
  assert(thetaNode->subregion()->nresults() == 2);
  assert(thetaOutput0->index() == 0);
  assert(thetaOutput0->result()->index() == 1);
}

static int
TestTheta()
{
  TestThetaCreation();
  TestRemoveThetaOutputsWhere();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-theta", TestTheta)
