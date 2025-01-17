/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/util/Statistics.hpp>

static const auto vt = jlm::tests::valuetype::Create();
static jlm::util::StatisticsCollector statisticsCollector;

static inline void
test1()
{
  using namespace jlm::llvm;

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, vt, "z");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lvx = theta->AddLoopVar(x);
  auto lvy = theta->AddLoopVar(y);
  theta->AddLoopVar(z);

  auto a = jlm::tests::create_testop(
      theta->subregion(),
      { lvx.pre, lvy.pre },
      { jlm::rvsdg::bittype::Create(1) })[0];
  auto predicate = jlm::rvsdg::match(1, { { 1, 0 } }, 1, 2, a);

  auto gamma = jlm::rvsdg::GammaNode::create(predicate, 2);

  auto evx = gamma->AddEntryVar(lvx.pre);
  auto evy = gamma->AddEntryVar(lvy.pre);

  auto b = jlm::tests::create_testop(
      gamma->subregion(0),
      { evx.branchArgument[0], evy.branchArgument[0] },
      { vt })[0];
  auto c = jlm::tests::create_testop(
      gamma->subregion(1),
      { evx.branchArgument[1], evy.branchArgument[1] },
      { vt })[0];

  auto xvy = gamma->AddExitVar({ b, c });

  lvy.post->divert_to(xvy.output);

  theta->set_predicate(predicate);

  auto & ex1 = GraphExport::Create(*theta->output(0), "x");
  auto & ex2 = GraphExport::Create(*theta->output(1), "y");
  auto & ex3 = GraphExport::Create(*theta->output(2), "z");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::tginversion tginversion;
  tginversion.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  assert(jlm::rvsdg::is<jlm::rvsdg::GammaOperation>(jlm::rvsdg::output::GetNode(*ex1.origin())));
  assert(jlm::rvsdg::is<jlm::rvsdg::GammaOperation>(jlm::rvsdg::output::GetNode(*ex2.origin())));
  assert(jlm::rvsdg::is<jlm::rvsdg::GammaOperation>(jlm::rvsdg::output::GetNode(*ex3.origin())));
}

static inline void
test2()
{
  using namespace jlm::llvm;

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lv1 = theta->AddLoopVar(x);

  auto n1 = jlm::tests::create_testop(
      theta->subregion(),
      { lv1.pre },
      { jlm::rvsdg::bittype::Create(1) })[0];
  auto n2 = jlm::tests::create_testop(theta->subregion(), { lv1.pre }, { vt })[0];
  auto predicate = jlm::rvsdg::match(1, { { 1, 0 } }, 1, 2, n1);

  auto gamma = jlm::rvsdg::GammaNode::create(predicate, 2);

  auto ev1 = gamma->AddEntryVar(n1);
  auto ev2 = gamma->AddEntryVar(lv1.pre);
  auto ev3 = gamma->AddEntryVar(n2);

  gamma->AddExitVar(ev1.branchArgument);
  gamma->AddExitVar(ev2.branchArgument);
  gamma->AddExitVar(ev3.branchArgument);

  lv1.post->divert_to(gamma->output(1));

  theta->set_predicate(predicate);

  auto & ex = GraphExport::Create(*theta->output(0), "x");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::tginversion tginversion;
  tginversion.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  assert(jlm::rvsdg::is<jlm::rvsdg::GammaOperation>(jlm::rvsdg::output::GetNode(*ex.origin())));
}

static int
verify()
{
  test1();
  test2();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inversion", verify)
