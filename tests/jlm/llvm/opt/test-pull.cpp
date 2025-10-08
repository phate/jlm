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

static const auto vt = jlm::tests::ValueType::Create();
static jlm::util::StatisticsCollector statisticsCollector;

static inline void
test_pullin_top()
{
  using namespace jlm::llvm;

  auto ct = jlm::rvsdg::ControlType::Create(2);
  jlm::tests::TestOperation uop({ vt }, { vt });
  jlm::tests::TestOperation bop({ vt, vt }, { vt });
  jlm::tests::TestOperation cop({ ct, vt }, { ct });

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  auto n1 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { x }, { vt })->output(0);
  auto n2 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { x }, { vt })->output(0);
  auto n3 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n2 }, { vt })->output(0);
  auto n4 = jlm::tests::TestOperation::create(&graph.GetRootRegion(), { c, n1 }, { ct })->output(0);
  auto n5 =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { n1, n3 }, { vt })->output(0);

  auto gamma = jlm::rvsdg::GammaNode::create(n4, 2);

  gamma->AddEntryVar(n4);
  auto ev = gamma->AddEntryVar(n5);
  gamma->AddExitVar(ev.branchArgument);

  jlm::rvsdg::GraphExport::Create(*gamma->output(0), "x");
  jlm::rvsdg::GraphExport::Create(*n2, "y");

  //	jlm::rvsdg::view(graph, stdout);
  pullin_top(gamma);
  //	jlm::rvsdg::view(graph, stdout);

  assert(gamma->subregion(0)->numNodes() == 2);
  assert(gamma->subregion(1)->numNodes() == 2);
}

static inline void
test_pullin_bottom()
{
  auto vt = jlm::tests::ValueType::Create();
  auto ct = jlm::rvsdg::ControlType::Create(2);

  jlm::rvsdg::Graph graph;
  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  auto gamma = jlm::rvsdg::GammaNode::create(c, 2);

  auto ev = gamma->AddEntryVar(x);
  gamma->AddExitVar(ev.branchArgument);

  auto b1 =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { gamma->output(0), x }, { vt })
          ->output(0);
  auto b2 =
      jlm::tests::TestOperation::create(&graph.GetRootRegion(), { gamma->output(0), b1 }, { vt })
          ->output(0);

  auto & xp = jlm::rvsdg::GraphExport::Create(*b2, "x");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::pullin_bottom(gamma);
  //	jlm::rvsdg::view(graph, stdout);

  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*xp.origin()) == gamma);
  assert(gamma->subregion(0)->numNodes() == 2);
  assert(gamma->subregion(1)->numNodes() == 2);
}

static void
test_pull()
{
  using namespace jlm::llvm;

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto p = &jlm::rvsdg::GraphImport::Create(graph, jlm::rvsdg::ControlType::Create(2), "");

  auto croot = jlm::tests::TestOperation::create(&graph.GetRootRegion(), {}, { vt })->output(0);

  /* outer gamma */
  auto gamma1 = jlm::rvsdg::GammaNode::create(p, 2);
  auto ev1 = gamma1->AddEntryVar(p);
  auto ev2 = gamma1->AddEntryVar(croot);

  auto cg1 = jlm::tests::TestOperation::create(gamma1->subregion(0), {}, { vt })->output(0);

  /* inner gamma */
  auto gamma2 = jlm::rvsdg::GammaNode::create(ev1.branchArgument[1], 2);
  auto ev3 = gamma2->AddEntryVar(ev2.branchArgument[1]);
  auto cg2 = jlm::tests::TestOperation::create(gamma2->subregion(0), {}, { vt })->output(0);
  auto un =
      jlm::tests::TestOperation::create(gamma2->subregion(1), { ev3.branchArgument[1] }, { vt })
          ->output(0);
  auto g2xv = gamma2->AddExitVar({ cg2, un });

  auto g1xv = gamma1->AddExitVar({ cg1, g2xv.output });

  jlm::rvsdg::GraphExport::Create(*g1xv.output, "");

  jlm::rvsdg::view(graph, stdout);
  jlm::llvm::NodeSinking pullin;
  pullin.Run(rm, statisticsCollector);
  graph.PruneNodes();
  jlm::rvsdg::view(graph, stdout);

  assert(graph.GetRootRegion().numNodes() == 1);
}

static void
verify()
{
  test_pullin_top();
  test_pullin_bottom();

  test_pull();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-pull", verify)
