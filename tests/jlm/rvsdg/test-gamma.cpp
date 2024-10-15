/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

static void
test_gamma(void)
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::graph graph;
  auto cmp = &jlm::tests::GraphImport::Create(graph, bittype::Create(2), "");
  auto v0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "");
  auto v1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "");
  auto v2 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "");
  auto v3 = &jlm::tests::GraphImport::Create(graph, ControlType::Create(2), "");

  auto pred = match(2, { { 0, 0 }, { 1, 1 } }, 2, 3, cmp);

  auto gamma = GammaNode::create(pred, 3);
  auto ev0 = gamma->add_entryvar(v0);
  auto ev1 = gamma->add_entryvar(v1);
  auto ev2 = gamma->add_entryvar(v2);
  gamma->add_exitvar({ ev0->argument(0), ev1->argument(1), ev2->argument(2) });

  jlm::tests::GraphExport::Create(*gamma->output(0), "dummy");

  assert(gamma && gamma->operation() == GammaOperation(3));

  /* test gamma copy */

  auto gamma2 = static_cast<StructuralNode *>(gamma)->copy(graph.root(), { pred, v0, v1, v2 });
  view(graph.root(), stdout);
  assert(is<GammaOperation>(gamma2));

  /* test entry and exit variable iterators */

  auto gamma3 = GammaNode::create(v3, 2);
  assert(gamma3->begin_entryvar() == gamma3->end_entryvar());
  assert(gamma3->begin_exitvar() == gamma3->end_exitvar());
}

static void
test_predicate_reduction(void)
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::graph graph;
  GammaOperation::normal_form(&graph)->set_predicate_reduction(true);

  bittype bits2(2);

  auto v0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "");
  auto v1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "");
  auto v2 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "");

  auto pred = jlm::rvsdg::control_constant(graph.root(), 3, 1);

  auto gamma = GammaNode::create(pred, 3);
  auto ev0 = gamma->add_entryvar(v0);
  auto ev1 = gamma->add_entryvar(v1);
  auto ev2 = gamma->add_entryvar(v2);
  gamma->add_exitvar({ ev0->argument(0), ev1->argument(1), ev2->argument(2) });

  auto & r = jlm::tests::GraphExport::Create(*gamma->output(0), "");

  graph.normalize();
  //	jlm::rvsdg::view(graph.root(), stdout);
  assert(r.origin() == v1);

  graph.prune();
  assert(graph.root()->nnodes() == 0);
}

static void
test_invariant_reduction(void)
{
  using namespace jlm::rvsdg;

  auto vtype = jlm::tests::valuetype::Create();

  jlm::rvsdg::graph graph;
  GammaOperation::normal_form(&graph)->set_invariant_reduction(true);

  auto pred = &jlm::tests::GraphImport::Create(graph, ControlType::Create(2), "");
  auto v = &jlm::tests::GraphImport::Create(graph, vtype, "");

  auto gamma = GammaNode::create(pred, 2);
  auto ev = gamma->add_entryvar(v);
  gamma->add_exitvar({ ev->argument(0), ev->argument(1) });

  auto & r = jlm::tests::GraphExport::Create(*gamma->output(0), "");

  graph.normalize();
  //	jlm::rvsdg::view(graph.root(), stdout);
  assert(r.origin() == v);

  graph.prune();
  assert(graph.root()->nnodes() == 0);
}

static void
test_control_constant_reduction()
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::graph graph;
  GammaOperation::normal_form(&graph)->set_control_constant_reduction(true);

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(1), "x");

  auto c = match(1, { { 0, 0 } }, 1, 2, x);

  auto gamma = GammaNode::create(c, 2);

  auto t = jlm::rvsdg::control_true(gamma->subregion(0));
  auto f = jlm::rvsdg::control_false(gamma->subregion(1));

  auto n0 = jlm::rvsdg::control_constant(gamma->subregion(0), 3, 0);
  auto n1 = jlm::rvsdg::control_constant(gamma->subregion(1), 3, 1);

  auto xv1 = gamma->add_exitvar({ t, f });
  auto xv2 = gamma->add_exitvar({ n0, n1 });

  auto & ex1 = jlm::tests::GraphExport::Create(*xv1, "");
  auto & ex2 = jlm::tests::GraphExport::Create(*xv2, "");

  jlm::rvsdg::view(graph.root(), stdout);
  graph.normalize();
  jlm::rvsdg::view(graph.root(), stdout);

  auto match = output::GetNode(*ex1.origin());
  assert(match && is<match_op>(match->operation()));
  auto & match_op = to_match_op(match->operation());
  assert(match_op.default_alternative() == 0);

  assert(output::GetNode(*ex2.origin()) == gamma);
}

static void
test_control_constant_reduction2()
{
  using namespace jlm::rvsdg;

  jlm::rvsdg::graph graph;
  GammaOperation::normal_form(&graph)->set_control_constant_reduction(true);

  auto import = &jlm::tests::GraphImport::Create(graph, bittype::Create(2), "import");

  auto c = match(2, { { 3, 2 }, { 2, 1 }, { 1, 0 } }, 3, 4, import);

  auto gamma = GammaNode::create(c, 4);

  auto t1 = jlm::rvsdg::control_true(gamma->subregion(0));
  auto t2 = jlm::rvsdg::control_true(gamma->subregion(1));
  auto t3 = jlm::rvsdg::control_true(gamma->subregion(2));
  auto f = jlm::rvsdg::control_false(gamma->subregion(3));

  auto xv = gamma->add_exitvar({ t1, t2, t3, f });

  auto & ex = jlm::tests::GraphExport::Create(*xv, "");

  jlm::rvsdg::view(graph.root(), stdout);
  graph.normalize();
  jlm::rvsdg::view(graph.root(), stdout);

  auto match = output::GetNode(*ex.origin());
  assert(is<match_op>(match));
}

static void
TestRemoveGammaOutputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  jlm::rvsdg::graph rvsdg;
  auto vt = jlm::tests::valuetype::Create();
  ControlType ct(2);

  auto predicate = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "");
  auto v0 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");
  auto v1 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");
  auto v2 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");
  auto v3 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");

  auto gammaNode = GammaNode::create(predicate, 2);
  auto gammaInput0 = gammaNode->add_entryvar(v0);
  auto gammaInput1 = gammaNode->add_entryvar(v1);
  auto gammaInput2 = gammaNode->add_entryvar(v2);
  auto gammaInput3 = gammaNode->add_entryvar(v3);

  auto gammaOutput0 =
      gammaNode->add_exitvar({ gammaInput0->argument(0), gammaInput0->argument(1) });
  auto gammaOutput1 =
      gammaNode->add_exitvar({ gammaInput1->argument(0), gammaInput1->argument(1) });
  auto gammaOutput2 =
      gammaNode->add_exitvar({ gammaInput2->argument(0), gammaInput2->argument(1) });
  auto gammaOutput3 =
      gammaNode->add_exitvar({ gammaInput3->argument(0), gammaInput3->argument(1) });

  jlm::tests::GraphExport::Create(*gammaOutput0, "");
  jlm::tests::GraphExport::Create(*gammaOutput2, "");

  // Act & Assert
  assert(gammaNode->noutputs() == 4);

  // Remove gammaOutput1
  gammaNode->RemoveGammaOutputsWhere(
      [&](const GammaOutput & output)
      {
        return output.index() == gammaOutput1->index();
      });
  assert(gammaNode->noutputs() == 3);
  assert(gammaNode->subregion(0)->nresults() == 3);
  assert(gammaNode->subregion(1)->nresults() == 3);
  assert(gammaOutput2->index() == 1);
  assert(gammaOutput3->index() == 2);

  // Try to remove gammaOutput2. This should result in no change as gammaOutput2 still has users.
  gammaNode->RemoveGammaOutputsWhere(
      [&](const GammaOutput & output)
      {
        return output.index() == gammaOutput2->index();
      });
  assert(gammaNode->noutputs() == 3);
  assert(gammaNode->subregion(0)->nresults() == 3);
  assert(gammaNode->subregion(1)->nresults() == 3);
  assert(gammaOutput2->index() == 1);
  assert(gammaOutput3->index() == 2);
}

static void
TestPruneOutputs()
{
  using namespace jlm::rvsdg;

  // Arrange
  jlm::rvsdg::graph rvsdg;
  auto vt = jlm::tests::valuetype::Create();
  ControlType ct(2);

  auto predicate = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "");
  auto v0 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");
  auto v1 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");
  auto v2 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");
  auto v3 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");

  auto gammaNode = GammaNode::create(predicate, 2);
  auto gammaInput0 = gammaNode->add_entryvar(v0);
  auto gammaInput1 = gammaNode->add_entryvar(v1);
  auto gammaInput2 = gammaNode->add_entryvar(v2);
  auto gammaInput3 = gammaNode->add_entryvar(v3);

  auto gammaOutput0 =
      gammaNode->add_exitvar({ gammaInput0->argument(0), gammaInput0->argument(1) });
  gammaNode->add_exitvar({ gammaInput1->argument(0), gammaInput1->argument(1) });
  auto gammaOutput2 =
      gammaNode->add_exitvar({ gammaInput2->argument(0), gammaInput2->argument(1) });
  gammaNode->add_exitvar({ gammaInput3->argument(0), gammaInput3->argument(1) });

  jlm::tests::GraphExport::Create(*gammaOutput0, "");
  jlm::tests::GraphExport::Create(*gammaOutput2, "");

  // Act
  gammaNode->PruneOutputs();

  // Assert
  assert(gammaNode->noutputs() == 2);
  assert(gammaNode->subregion(0)->nresults() == 2);
  assert(gammaNode->subregion(1)->nresults() == 2);

  assert(gammaOutput0->index() == 0);
  assert(gammaNode->subregion(0)->result(0)->output() == gammaOutput0);
  assert(gammaNode->subregion(1)->result(0)->output() == gammaOutput0);

  assert(gammaOutput2->index() == 1);
  assert(gammaNode->subregion(0)->result(1)->output() == gammaOutput2);
  assert(gammaNode->subregion(1)->result(1)->output() == gammaOutput2);
}

static void
TestIsInvariant()
{
  using namespace jlm::rvsdg;

  // Arrange
  jlm::rvsdg::graph rvsdg;
  auto vt = jlm::tests::valuetype::Create();
  ControlType ct(2);

  auto predicate = &jlm::tests::GraphImport::Create(rvsdg, ControlType::Create(2), "");
  auto v0 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");
  auto v1 = &jlm::tests::GraphImport::Create(rvsdg, vt, "");

  auto gammaNode = GammaNode::create(predicate, 2);
  auto gammaInput0 = gammaNode->add_entryvar(v0);
  auto gammaInput1 = gammaNode->add_entryvar(v1);
  auto gammaInput2 = gammaNode->add_entryvar(v1);

  auto gammaOutput0 =
      gammaNode->add_exitvar({ gammaInput0->argument(0), gammaInput0->argument(1) });
  auto gammaOutput1 =
      gammaNode->add_exitvar({ gammaInput1->argument(0), gammaInput2->argument(1) });
  auto gammaOutput2 =
      gammaNode->add_exitvar({ gammaInput0->argument(0), gammaInput2->argument(1) });

  // Act & Assert
  assert(gammaOutput0->IsInvariant());
  output * invariantOrigin = nullptr;
  gammaOutput0->IsInvariant(&invariantOrigin);
  assert(invariantOrigin == v0);

  assert(gammaOutput1->IsInvariant(&invariantOrigin));
  assert(invariantOrigin == v1);

  invariantOrigin = nullptr;
  assert(!gammaOutput2->IsInvariant(&invariantOrigin));
  // invariantOrigin should not have been touched as gammaOutput2 is not invariant
  assert(invariantOrigin == nullptr);
}

static int
test_main()
{
  test_gamma();
  TestRemoveGammaOutputsWhere();
  TestPruneOutputs();
  TestIsInvariant();

  test_predicate_reduction();
  test_invariant_reduction();
  test_control_constant_reduction();
  test_control_constant_reduction2();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-gamma", test_main)
