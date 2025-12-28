/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/pull.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static const auto vt = jlm::rvsdg::TestType::createValueType();
static jlm::util::StatisticsCollector statisticsCollector;

static void
testPullInTop()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto ct = jlm::rvsdg::ControlType::Create(2);
  TestOperation uop({ vt }, { vt });
  TestOperation bop({ vt, vt }, { vt });
  TestOperation cop({ ct, vt }, { ct });

  jlm::llvm::RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = &jlm::rvsdg::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), { x }, { vt })->output(0);
  auto n2 = TestOperation::createNode(&graph.GetRootRegion(), { x }, { vt })->output(0);
  auto n3 = TestOperation::createNode(&graph.GetRootRegion(), { n2 }, { vt })->output(0);
  auto n4 = TestOperation::createNode(&graph.GetRootRegion(), { c, n1 }, { ct })->output(0);
  auto n5 = TestOperation::createNode(&graph.GetRootRegion(), { n1, n3 }, { vt })->output(0);

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

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-pull-testPullInTop", testPullInTop)

static void
testPullInBottom()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  auto valueType = TestType::createValueType();
  const auto controlType = ControlType::Create(2);

  Graph rvsdg;
  auto c = &jlm::rvsdg::GraphImport::Create(rvsdg, controlType, "c");
  auto x = &jlm::rvsdg::GraphImport::Create(rvsdg, valueType, "x");

  auto gammaNode = GammaNode::create(c, 2);

  auto entryVar = gammaNode->AddEntryVar(x);
  gammaNode->AddExitVar(entryVar.branchArgument);

  auto node1 =
      TestOperation::createNode(&rvsdg.GetRootRegion(), { gammaNode->output(0), x }, { valueType });
  auto node2 = TestOperation::createNode(
      &rvsdg.GetRootRegion(),
      { gammaNode->output(0), node1->output(0) },
      { valueType });

  auto & xp = GraphExport::Create(*node2->output(0), "x");

  view(rvsdg, stdout);

  // Act
  const auto sunkNodes = NodeSinking::sinkDependentNodesIntoGamma(*gammaNode);
  rvsdg.PruneNodes();

  view(rvsdg, stdout);

  // Assert
  assert(sunkNodes == 2);
  assert(rvsdg.GetRootRegion().numNodes() == 1);

  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*xp.origin()) == gammaNode);
  assert(gammaNode->subregion(0)->numNodes() == 2);
  assert(gammaNode->subregion(1)->numNodes() == 2);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-pull-testPullInBottom", testPullInBottom)

static void
testPull()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  jlm::llvm::RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto p = &jlm::rvsdg::GraphImport::Create(graph, jlm::rvsdg::ControlType::Create(2), "");

  auto croot = TestOperation::createNode(&graph.GetRootRegion(), {}, { vt })->output(0);

  /* outer gamma */
  auto gamma1 = jlm::rvsdg::GammaNode::create(p, 2);
  auto ev1 = gamma1->AddEntryVar(p);
  auto ev2 = gamma1->AddEntryVar(croot);

  auto cg1 = TestOperation::createNode(gamma1->subregion(0), {}, { vt })->output(0);

  /* inner gamma */
  auto gamma2 = jlm::rvsdg::GammaNode::create(ev1.branchArgument[1], 2);
  auto ev3 = gamma2->AddEntryVar(ev2.branchArgument[1]);
  auto cg2 = TestOperation::createNode(gamma2->subregion(0), {}, { vt })->output(0);
  auto un =
      TestOperation::createNode(gamma2->subregion(1), { ev3.branchArgument[1] }, { vt })->output(0);
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

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-pull-testPull", testPull)
