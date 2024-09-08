/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/util/Statistics.hpp>

static const auto vt = jlm::tests::valuetype::Create();
static jlm::util::StatisticsCollector statisticsCollector;

static inline void
test_pullin_top()
{
  using namespace jlm::llvm;

  auto ct = jlm::rvsdg::ctltype::Create(2);
  jlm::tests::test_op uop({ vt }, { vt });
  jlm::tests::test_op bop({ vt, vt }, { vt });
  jlm::tests::test_op cop({ ct, vt }, { ct });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto n1 = jlm::tests::create_testop(graph.root(), { x }, { vt })[0];
  auto n2 = jlm::tests::create_testop(graph.root(), { x }, { vt })[0];
  auto n3 = jlm::tests::create_testop(graph.root(), { n2 }, { vt })[0];
  auto n4 = jlm::tests::create_testop(graph.root(), { c, n1 }, { ct })[0];
  auto n5 = jlm::tests::create_testop(graph.root(), { n1, n3 }, { vt })[0];

  auto gamma = jlm::rvsdg::GammaNode::create(n4, 2);

  gamma->add_entryvar(n4);
  auto ev = gamma->add_entryvar(n5);
  gamma->add_exitvar({ ev->argument(0), ev->argument(1) });

  GraphExport::Create(*gamma->output(0), "x");
  GraphExport::Create(*n2, "y");

  //	jlm::rvsdg::view(graph, stdout);
  pullin_top(gamma);
  //	jlm::rvsdg::view(graph, stdout);

  assert(gamma->subregion(0)->nnodes() == 2);
  assert(gamma->subregion(1)->nnodes() == 2);
}

static inline void
test_pullin_bottom()
{
  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  jlm::rvsdg::graph graph;
  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto gamma = jlm::rvsdg::GammaNode::create(c, 2);

  auto ev = gamma->add_entryvar(x);
  gamma->add_exitvar({ ev->argument(0), ev->argument(1) });

  auto b1 = jlm::tests::create_testop(graph.root(), { gamma->output(0), x }, { vt })[0];
  auto b2 = jlm::tests::create_testop(graph.root(), { gamma->output(0), b1 }, { vt })[0];

  auto & xp = jlm::llvm::GraphExport::Create(*b2, "x");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::pullin_bottom(gamma);
  //	jlm::rvsdg::view(graph, stdout);

  assert(jlm::rvsdg::node_output::node(xp.origin()) == gamma);
  assert(gamma->subregion(0)->nnodes() == 2);
  assert(gamma->subregion(1)->nnodes() == 2);
}

static void
test_pull()
{
  using namespace jlm::llvm;

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto p = &jlm::tests::GraphImport::Create(graph, jlm::rvsdg::ctltype::Create(2), "");

  auto croot = jlm::tests::create_testop(graph.root(), {}, { vt })[0];

  /* outer gamma */
  auto gamma1 = jlm::rvsdg::GammaNode::create(p, 2);
  auto ev1 = gamma1->add_entryvar(p);
  auto ev2 = gamma1->add_entryvar(croot);

  auto cg1 = jlm::tests::create_testop(gamma1->subregion(0), {}, { vt })[0];

  /* inner gamma */
  auto gamma2 = jlm::rvsdg::GammaNode::create(ev1->argument(1), 2);
  auto ev3 = gamma2->add_entryvar(ev2->argument(1));
  auto cg2 = jlm::tests::create_testop(gamma2->subregion(0), {}, { vt })[0];
  auto un = jlm::tests::create_testop(gamma2->subregion(1), { ev3->argument(1) }, { vt })[0];
  auto g2xv = gamma2->add_exitvar({ cg2, un });

  auto g1xv = gamma1->add_exitvar({ cg1, g2xv });

  GraphExport::Create(*g1xv, "");

  jlm::rvsdg::view(graph, stdout);
  jlm::llvm::pullin pullin;
  pullin.run(rm, statisticsCollector);
  graph.prune();
  jlm::rvsdg::view(graph, stdout);

  assert(graph.root()->nnodes() == 1);
}

static int
verify()
{
  test_pullin_top();
  test_pullin_bottom();

  test_pull();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-pull", verify)
