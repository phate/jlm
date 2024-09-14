/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Phi.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/cne.hpp>
#include <jlm/util/Statistics.hpp>

static jlm::util::StatisticsCollector statisticsCollector;

static inline void
test_simple()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, vt, "z");

  auto n1 = jlm::tests::create_testop(graph.root(), {}, { vt })[0];
  auto n2 = jlm::tests::create_testop(graph.root(), {}, { vt })[0];

  auto u1 = jlm::tests::create_testop(graph.root(), { z }, { vt })[0];

  auto b1 = jlm::tests::create_testop(graph.root(), { x, y }, { vt })[0];
  auto b2 = jlm::tests::create_testop(graph.root(), { x, y }, { vt })[0];
  auto b3 = jlm::tests::create_testop(graph.root(), { n1, z }, { vt })[0];
  auto b4 = jlm::tests::create_testop(graph.root(), { n2, z }, { vt })[0];

  GraphExport::Create(*n1, "n1");
  GraphExport::Create(*n2, "n2");
  GraphExport::Create(*u1, "u1");
  GraphExport::Create(*b1, "b1");
  GraphExport::Create(*b2, "b2");
  GraphExport::Create(*b3, "b3");
  GraphExport::Create(*b4, "b4");

  //	jlm::rvsdg::view(graph.root(), stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());
  assert(graph.root()->result(3)->origin() == graph.root()->result(4)->origin());
  assert(graph.root()->result(5)->origin() == graph.root()->result(6)->origin());
}

static inline void
test_gamma()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, vt, "z");

  auto u1 = jlm::tests::create_testop(graph.root(), { x }, { vt })[0];
  auto u2 = jlm::tests::create_testop(graph.root(), { x }, { vt })[0];

  auto gamma = jlm::rvsdg::GammaNode::create(c, 2);

  auto ev1 = gamma->add_entryvar(u1);
  auto ev2 = gamma->add_entryvar(u2);
  auto ev3 = gamma->add_entryvar(y);
  auto ev4 = gamma->add_entryvar(z);
  auto ev5 = gamma->add_entryvar(z);

  auto n1 = jlm::tests::create_testop(gamma->subregion(0), {}, { vt })[0];
  auto n2 = jlm::tests::create_testop(gamma->subregion(0), {}, { vt })[0];
  auto n3 = jlm::tests::create_testop(gamma->subregion(0), {}, { vt })[0];

  gamma->add_exitvar({ ev1->argument(0), ev2->argument(1) });
  gamma->add_exitvar({ ev2->argument(0), ev2->argument(1) });
  gamma->add_exitvar({ ev3->argument(0), ev3->argument(1) });
  gamma->add_exitvar({ n1, ev3->argument(1) });
  gamma->add_exitvar({ n2, ev3->argument(1) });
  gamma->add_exitvar({ n3, ev3->argument(1) });
  gamma->add_exitvar({ ev5->argument(0), ev4->argument(1) });

  GraphExport::Create(*gamma->output(0), "x1");
  GraphExport::Create(*gamma->output(1), "x2");
  GraphExport::Create(*gamma->output(2), "y");

  //	jlm::rvsdg::view(graph.root(), stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  auto subregion0 = gamma->subregion(0);
  auto subregion1 = gamma->subregion(1);
  assert(gamma->input(1)->origin() == gamma->input(2)->origin());
  assert(subregion0->result(0)->origin() == subregion0->result(1)->origin());
  assert(subregion0->result(3)->origin() == subregion0->result(4)->origin());
  assert(subregion0->result(3)->origin() == subregion0->result(5)->origin());
  assert(subregion1->result(0)->origin() == subregion1->result(1)->origin());
  assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());

  auto argument0 =
      dynamic_cast<const jlm::rvsdg::RegionArgument *>(subregion0->result(6)->origin());
  auto argument1 =
      dynamic_cast<const jlm::rvsdg::RegionArgument *>(subregion1->result(6)->origin());
  assert(argument0->input() == argument1->input());
}

static inline void
test_theta()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(graph.root());
  auto region = theta->subregion();

  auto lv1 = theta->add_loopvar(c);
  auto lv2 = theta->add_loopvar(x);
  auto lv3 = theta->add_loopvar(x);
  auto lv4 = theta->add_loopvar(x);

  auto u1 = jlm::tests::create_testop(region, { lv2->argument() }, { vt })[0];
  auto u2 = jlm::tests::create_testop(region, { lv3->argument() }, { vt })[0];
  auto b1 = jlm::tests::create_testop(region, { lv3->argument(), lv4->argument() }, { vt })[0];

  lv2->result()->divert_to(u1);
  lv3->result()->divert_to(u2);
  lv4->result()->divert_to(b1);

  theta->set_predicate(lv1->argument());

  GraphExport::Create(*theta->output(1), "lv2");
  GraphExport::Create(*theta->output(2), "lv3");
  GraphExport::Create(*theta->output(3), "lv4");

  //	jlm::rvsdg::view(graph.root(), stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  auto un1 = jlm::rvsdg::node_output::node(u1);
  auto un2 = jlm::rvsdg::node_output::node(u2);
  auto bn1 = jlm::rvsdg::node_output::node(b1);
  assert(un1->input(0)->origin() == un2->input(0)->origin());
  assert(bn1->input(0)->origin() == un1->input(0)->origin());
  assert(bn1->input(1)->origin() == region->argument(3));
  assert(region->result(2)->origin() == region->result(3)->origin());
  assert(graph.root()->result(0)->origin() == graph.root()->result(1)->origin());
}

static inline void
test_theta2()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto theta = jlm::rvsdg::ThetaNode::create(graph.root());
  auto region = theta->subregion();

  auto lv1 = theta->add_loopvar(c);
  auto lv2 = theta->add_loopvar(x);
  auto lv3 = theta->add_loopvar(x);

  auto u1 = jlm::tests::create_testop(region, { lv2->argument() }, { vt })[0];
  auto u2 = jlm::tests::create_testop(region, { lv3->argument() }, { vt })[0];
  auto b1 = jlm::tests::create_testop(region, { u2, u2 }, { vt })[0];

  lv2->result()->divert_to(u1);
  lv3->result()->divert_to(b1);

  theta->set_predicate(lv1->argument());

  GraphExport::Create(*theta->output(1), "lv2");
  GraphExport::Create(*theta->output(2), "lv3");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(lv2->result()->origin() == u1);
  assert(lv2->argument()->nusers() != 0 && lv3->argument()->nusers() != 0);
}

static inline void
test_theta3()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto theta1 = jlm::rvsdg::ThetaNode::create(graph.root());
  auto r1 = theta1->subregion();

  auto lv1 = theta1->add_loopvar(c);
  auto lv2 = theta1->add_loopvar(x);
  auto lv3 = theta1->add_loopvar(x);
  auto lv4 = theta1->add_loopvar(x);

  auto theta2 = jlm::rvsdg::ThetaNode::create(r1);
  auto r2 = theta2->subregion();
  auto p = theta2->add_loopvar(lv1->argument());
  theta2->add_loopvar(lv2->argument());
  theta2->add_loopvar(lv3->argument());
  theta2->add_loopvar(lv4->argument());
  theta2->set_predicate(p->argument());

  auto u1 = jlm::tests::test_op::create(r1, { theta2->output(1) }, { vt });
  auto b1 = jlm::tests::test_op::create(r1, { theta2->output(2), theta2->output(2) }, { vt });
  auto u2 = jlm::tests::test_op::create(r1, { theta2->output(3) }, { vt });

  lv2->result()->divert_to(u1->output(0));
  lv3->result()->divert_to(b1->output(0));
  lv4->result()->divert_to(u1->output(0));

  theta1->set_predicate(lv1->argument());

  GraphExport::Create(*theta1->output(1), "lv2");
  GraphExport::Create(*theta1->output(2), "lv3");
  GraphExport::Create(*theta1->output(3), "lv4");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(r1->result(2)->origin() == r1->result(4)->origin());
  assert(u1->input(0)->origin() == u2->input(0)->origin());
  assert(r2->result(2)->origin() == r2->result(4)->origin());
  assert(theta2->input(1)->origin() == theta2->input(3)->origin());
  assert(r1->result(3)->origin() != r1->result(4)->origin());
  assert(r2->result(3)->origin() != r2->result(4)->origin());
}

static inline void
test_theta4()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");

  auto theta = jlm::rvsdg::ThetaNode::create(graph.root());
  auto region = theta->subregion();

  auto lv1 = theta->add_loopvar(c);
  auto lv2 = theta->add_loopvar(x);
  auto lv3 = theta->add_loopvar(x);
  auto lv4 = theta->add_loopvar(y);
  auto lv5 = theta->add_loopvar(y);
  auto lv6 = theta->add_loopvar(x);
  auto lv7 = theta->add_loopvar(x);

  auto u1 = jlm::tests::test_op::create(region, { lv2->argument() }, { vt });
  auto b1 = jlm::tests::test_op::create(region, { lv3->argument(), lv3->argument() }, { vt });

  lv2->result()->divert_to(lv4->argument());
  lv3->result()->divert_to(lv5->argument());
  lv4->result()->divert_to(u1->output(0));
  lv5->result()->divert_to(b1->output(0));

  theta->set_predicate(lv1->argument());

  auto & ex1 = GraphExport::Create(*theta->output(1), "lv2");
  auto & ex2 = GraphExport::Create(*theta->output(2), "lv3");
  GraphExport::Create(*theta->output(3), "lv4");
  GraphExport::Create(*theta->output(4), "lv5");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(ex1.origin() != ex2.origin());
  assert(lv2->argument()->nusers() != 0 && lv3->argument()->nusers() != 0);
  assert(lv6->result()->origin() == lv7->result()->origin());
}

static inline void
test_theta5()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ct = jlm::rvsdg::ctltype::Create(2);

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto c = &jlm::tests::GraphImport::Create(graph, ct, "c");
  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, vt, "y");

  auto theta = jlm::rvsdg::ThetaNode::create(graph.root());
  auto region = theta->subregion();

  auto lv0 = theta->add_loopvar(c);
  auto lv1 = theta->add_loopvar(x);
  auto lv2 = theta->add_loopvar(x);
  auto lv3 = theta->add_loopvar(y);
  auto lv4 = theta->add_loopvar(y);

  lv1->result()->divert_to(lv3->argument());
  lv2->result()->divert_to(lv4->argument());

  theta->set_predicate(lv0->argument());

  auto & ex1 = GraphExport::Create(*theta->output(1), "lv1");
  auto & ex2 = GraphExport::Create(*theta->output(2), "lv2");
  auto & ex3 = GraphExport::Create(*theta->output(3), "lv3");
  auto & ex4 = GraphExport::Create(*theta->output(4), "lv4");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  assert(ex1.origin() == ex2.origin());
  assert(ex3.origin() == ex4.origin());
  assert(region->result(4)->origin() == region->result(5)->origin());
  assert(region->result(2)->origin() == region->result(3)->origin());
}

static inline void
test_lambda()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ft = FunctionType::Create({ vt, vt }, { vt });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  auto lambda = lambda::node::create(graph.root(), ft, "f", linkage::external_linkage);

  auto d1 = lambda->add_ctxvar(x);
  auto d2 = lambda->add_ctxvar(x);

  auto b1 = jlm::tests::create_testop(lambda->subregion(), { d1, d2 }, { vt })[0];

  auto output = lambda->finalize({ b1 });

  GraphExport::Create(*output, "f");

  //	jlm::rvsdg::view(graph.root(), stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  auto bn1 = jlm::rvsdg::node_output::node(b1);
  assert(bn1->input(0)->origin() == bn1->input(1)->origin());
}

static inline void
test_phi()
{
  using namespace jlm::llvm;

  auto vt = jlm::tests::valuetype::Create();
  auto ft = FunctionType::Create({ vt, vt }, { vt });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, vt, "x");

  phi::builder pb;
  pb.begin(graph.root());
  auto region = pb.subregion();

  auto d1 = pb.add_ctxvar(x);
  auto d2 = pb.add_ctxvar(x);

  auto r1 = pb.add_recvar(PointerType::Create());
  auto r2 = pb.add_recvar(PointerType::Create());

  auto lambda1 = lambda::node::create(region, ft, "f", linkage::external_linkage);
  auto cv1 = lambda1->add_ctxvar(d1);
  auto f1 = lambda1->finalize({ cv1 });

  auto lambda2 = lambda::node::create(region, ft, "f", linkage::external_linkage);
  auto cv2 = lambda2->add_ctxvar(d2);
  auto f2 = lambda2->finalize({ cv2 });

  r1->set_rvorigin(f1);
  r2->set_rvorigin(f2);

  auto phi = pb.end();

  GraphExport::Create(*phi->output(0), "f1");
  GraphExport::Create(*phi->output(1), "f2");

  //	jlm::rvsdg::view(graph.root(), stdout);
  jlm::llvm::cne cne;
  cne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph.root(), stdout);

  assert(f1->node()->input(0)->origin() == f2->node()->input(0)->origin());
}

static int
verify()
{
  test_simple();
  test_gamma();
  test_theta();
  test_theta2();
  test_theta3();
  test_theta4();
  test_theta5();
  test_lambda();
  test_phi();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-cne", verify)
