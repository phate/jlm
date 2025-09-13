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
#include <jlm/llvm/opt/LoopUnswitching.hpp>
#include <jlm/util/Statistics.hpp>

static const auto vt = jlm::tests::ValueType::Create();
static jlm::util::StatisticsCollector statisticsCollector;

static inline void
test1()
{
  using namespace jlm::llvm;

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, vt, "z");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lvx = theta->AddLoopVar(x);
  auto lvy = theta->AddLoopVar(y);
  theta->AddLoopVar(z);

  auto a = jlm::tests::TestOperation::create(
               theta->subregion(),
               { lvx.pre, lvy.pre },
               { jlm::rvsdg::BitType::Create(1) })
               ->output(0);
  auto predicate = jlm::rvsdg::match(1, { { 1, 0 } }, 1, 2, a);

  auto gamma = jlm::rvsdg::GammaNode::create(predicate, 2);

  auto evx = gamma->AddEntryVar(lvx.pre);
  auto evy = gamma->AddEntryVar(lvy.pre);

  auto b = jlm::tests::TestOperation::create(
               gamma->subregion(0),
               { evx.branchArgument[0], evy.branchArgument[0] },
               { vt })
               ->output(0);
  auto c = jlm::tests::TestOperation::create(
               gamma->subregion(1),
               { evx.branchArgument[1], evy.branchArgument[1] },
               { vt })
               ->output(0);

  auto xvy = gamma->AddExitVar({ b, c });

  lvy.post->divert_to(xvy.output);

  theta->set_predicate(predicate);

  auto & ex1 = jlm::rvsdg::GraphExport::Create(*theta->output(0), "x");
  auto & ex2 = jlm::rvsdg::GraphExport::Create(*theta->output(1), "y");
  auto & ex3 = jlm::rvsdg::GraphExport::Create(*theta->output(2), "z");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::LoopUnswitching tginversion;
  tginversion.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex1.origin()));
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex2.origin()));
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex3.origin()));
}

static inline void
test2()
{
  using namespace jlm::llvm;

  RvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lv1 = theta->AddLoopVar(x);

  auto n1 = jlm::tests::TestOperation::create(
                theta->subregion(),
                { lv1.pre },
                { jlm::rvsdg::BitType::Create(1) })
                ->output(0);
  auto n2 = jlm::tests::TestOperation::create(theta->subregion(), { lv1.pre }, { vt })->output(0);
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

  auto & ex = jlm::rvsdg::GraphExport::Create(*theta->output(0), "x");

  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);
  jlm::llvm::LoopUnswitching tginversion;
  tginversion.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.GetRootRegion(), stdout);

  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex.origin()));
}

static void
verify()
{
  test1();
  test2();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-inversion", verify)
