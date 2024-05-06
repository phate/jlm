/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/push.hpp>
#include <jlm/util/Statistics.hpp>

static const jlm::tests::statetype st;
static const jlm::tests::valuetype vt;
static jlm::util::StatisticsCollector statisticsCollector;

static inline void
test_gamma()
{
  using namespace jlm::llvm;

  jlm::rvsdg::ctltype ct(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = graph.add_import({ ct, "c" });
  auto x = graph.add_import({ vt, "x" });
  auto s = graph.add_import({ st, "s" });

  auto gamma = jlm::rvsdg::gamma_node::create(c, 2);
  auto evx = gamma->add_entryvar(x);
  auto evs = gamma->add_entryvar(s);

  auto null = jlm::tests::create_testop(gamma->subregion(0), {}, { &vt })[0];
  auto bin = jlm::tests::create_testop(gamma->subregion(0), { null, evx->argument(0) }, { &vt })[0];
  auto state =
      jlm::tests::create_testop(gamma->subregion(0), { bin, evs->argument(0) }, { &st })[0];

  gamma->add_exitvar({ state, evs->argument(1) });

  graph.add_export(gamma->output(0), { gamma->output(0)->type(), "x" });

  //	jlm::rvsdg::view(graph.root(), stdout);
  jlm::llvm::pushout pushout;
  pushout.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(graph.root()->nodes.size() == 3);
}

static inline void
test_theta()
{
  using namespace jlm::llvm;

  jlm::rvsdg::ctltype ct(2);

  jlm::tests::test_op nop({}, { &vt });
  jlm::tests::test_op bop({ &vt, &vt }, { &vt });
  jlm::tests::test_op sop({ &vt, &st }, { &st });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto c = graph.add_import({ ct, "c" });
  auto x = graph.add_import({ vt, "x" });
  auto s = graph.add_import({ st, "s" });

  auto theta = jlm::rvsdg::theta_node::create(graph.root());

  auto lv1 = theta->add_loopvar(c);
  auto lv2 = theta->add_loopvar(x);
  auto lv3 = theta->add_loopvar(x);
  auto lv4 = theta->add_loopvar(s);

  auto o1 = jlm::tests::create_testop(theta->subregion(), {}, { &vt })[0];
  auto o2 = jlm::tests::create_testop(theta->subregion(), { o1, lv3->argument() }, { &vt })[0];
  auto o3 = jlm::tests::create_testop(theta->subregion(), { lv2->argument(), o2 }, { &vt })[0];
  auto o4 = jlm::tests::create_testop(
      theta->subregion(),
      { lv3->argument(), lv4->argument() },
      { &st })[0];

  lv2->result()->divert_to(o3);
  lv4->result()->divert_to(o4);

  theta->set_predicate(lv1->argument());

  graph.add_export(theta->output(0), { theta->output(0)->type(), "c" });

  //	jlm::rvsdg::view(graph.root(), stdout);
  jlm::llvm::pushout pushout;
  pushout.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(graph.root()->nodes.size() == 3);
}

static inline void
test_push_theta_bottom()
{
  using namespace jlm::llvm;

  MemoryStateType mt;
  PointerType pt;
  jlm::rvsdg::ctltype ct(2);

  jlm::rvsdg::graph graph;
  auto c = graph.add_import({ ct, "c" });
  auto a = graph.add_import({ pt, "a" });
  auto v = graph.add_import({ vt, "v" });
  auto s = graph.add_import({ mt, "s" });

  auto theta = jlm::rvsdg::theta_node::create(graph.root());

  auto lvc = theta->add_loopvar(c);
  auto lva = theta->add_loopvar(a);
  auto lvv = theta->add_loopvar(v);
  auto lvs = theta->add_loopvar(s);

  auto s1 =
      StoreNonVolatileNode::Create(lva->argument(), lvv->argument(), { lvs->argument() }, 4)[0];

  lvs->result()->divert_to(s1);
  theta->set_predicate(lvc->argument());

  auto ex = graph.add_export(lvs, { lvs->type(), "s" });

  jlm::rvsdg::view(graph, stdout);
  jlm::llvm::push_bottom(theta);
  jlm::rvsdg::view(graph, stdout);

  auto storenode = jlm::rvsdg::node_output::node(ex->origin());
  assert(jlm::rvsdg::is<StoreNonVolatileOperation>(storenode));
  assert(storenode->input(0)->origin() == a);
  assert(jlm::rvsdg::is<jlm::rvsdg::theta_op>(
      jlm::rvsdg::node_output::node(storenode->input(1)->origin())));
  assert(jlm::rvsdg::is<jlm::rvsdg::theta_op>(
      jlm::rvsdg::node_output::node(storenode->input(2)->origin())));
}

static int
verify()
{
  test_gamma();
  test_theta();
  test_push_theta_bottom();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-push", verify)
