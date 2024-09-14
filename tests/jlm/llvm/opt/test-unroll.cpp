/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/util/Statistics.hpp>

static jlm::util::StatisticsCollector statisticsCollector;

static size_t
nthetas(jlm::rvsdg::region * region)
{
  size_t n = 0;
  for (const auto & node : region->nodes)
  {
    if (jlm::rvsdg::is<jlm::rvsdg::ThetaOperation>(&node))
      n++;
  }

  return n;
}

static jlm::rvsdg::ThetaNode *
create_theta(
    const jlm::rvsdg::bitcompare_op & cop,
    const jlm::rvsdg::bitbinary_op & aop,
    jlm::rvsdg::output * init,
    jlm::rvsdg::output * step,
    jlm::rvsdg::output * end)
{
  using namespace jlm::rvsdg;

  auto graph = init->region()->graph();

  auto theta = ThetaNode::create(graph->root());
  auto subregion = theta->subregion();
  auto idv = theta->add_loopvar(init);
  auto lvs = theta->add_loopvar(step);
  auto lve = theta->add_loopvar(end);

  auto arm =
      simple_node::create_normalized(subregion, aop, { idv->argument(), lvs->argument() })[0];
  auto cmp = simple_node::create_normalized(subregion, cop, { arm, lve->argument() })[0];
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  idv->result()->divert_to(arm);
  theta->set_predicate(match);

  return theta;
}

static inline void
test_unrollinfo()
{
  auto bt32 = jlm::rvsdg::bittype::Create(32);
  jlm::rvsdg::bitslt_op slt(32);
  jlm::rvsdg::bitult_op ult(32);
  jlm::rvsdg::bitule_op ule(32);
  jlm::rvsdg::bitugt_op ugt(32);
  jlm::rvsdg::bitsge_op sge(32);
  jlm::rvsdg::biteq_op eq(32);

  jlm::rvsdg::bitadd_op add(32);
  jlm::rvsdg::bitsub_op sub(32);

  {
    jlm::rvsdg::graph graph;
    auto x = &jlm::tests::GraphImport::Create(graph, bt32, "x");
    auto theta = create_theta(slt, add, x, x, x);
    auto ui = jlm::llvm::unrollinfo::create(theta);

    assert(ui);
    assert(ui->is_additive());
    assert(!ui->is_subtractive());
    assert(!ui->is_known());
    assert(!ui->niterations());
    assert(ui->theta() == theta);
    assert(ui->idv()->input()->origin() == x);
  }

  {
    jlm::rvsdg::graph graph;
    auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
    nf->set_mutable(false);

    auto init0 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0);
    auto init1 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 1);
    auto initm1 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0xFFFFFFFF);

    auto step1 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 1);
    auto step0 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0);
    auto stepm1 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0xFFFFFFFF);
    auto step2 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 2);

    auto end100 = jlm::rvsdg::create_bitconstant(graph.root(), 32, 100);

    auto theta = create_theta(ult, add, init0, step1, end100);
    auto ui = jlm::llvm::unrollinfo::create(theta);
    assert(ui && *ui->niterations() == 100);

    theta = create_theta(ule, add, init0, step1, end100);
    ui = jlm::llvm::unrollinfo::create(theta);
    assert(ui && *ui->niterations() == 101);

    theta = create_theta(ugt, sub, end100, stepm1, init0);
    ui = jlm::llvm::unrollinfo::create(theta);
    assert(ui && *ui->niterations() == 100);

    theta = create_theta(sge, sub, end100, stepm1, init0);
    ui = jlm::llvm::unrollinfo::create(theta);
    assert(ui && *ui->niterations() == 101);

    theta = create_theta(ult, add, init0, step0, end100);
    ui = jlm::llvm::unrollinfo::create(theta);
    assert(ui && !ui->niterations());

    theta = create_theta(eq, add, initm1, step1, end100);
    ui = jlm::llvm::unrollinfo::create(theta);
    assert(ui && *ui->niterations() == 101);

    theta = create_theta(eq, add, init1, step2, end100);
    ui = jlm::llvm::unrollinfo::create(theta);
    assert(ui && !ui->niterations());
  }
}

static inline void
test_known_boundaries()
{
  jlm::rvsdg::bitult_op ult(32);
  jlm::rvsdg::bitsgt_op sgt(32);
  jlm::rvsdg::bitadd_op add(32);
  jlm::rvsdg::bitsub_op sub(32);

  {
    jlm::rvsdg::graph graph;
    auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
    nf->set_mutable(false);

    auto init = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0);
    auto step = jlm::rvsdg::create_bitconstant(graph.root(), 32, 1);
    auto end = jlm::rvsdg::create_bitconstant(graph.root(), 32, 4);

    auto theta = create_theta(ult, add, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 4);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is greater than or equal the number of iterations.
      The loop should be fully unrolled and the theta removed.
    */
    assert(nthetas(graph.root()) == 0);
  }

  {
    jlm::rvsdg::graph graph;
    auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
    nf->set_mutable(false);

    auto init = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0);
    auto step = jlm::rvsdg::create_bitconstant(graph.root(), 32, 1);
    auto end = jlm::rvsdg::create_bitconstant(graph.root(), 32, 100);

    auto theta = create_theta(ult, add, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 2);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is a multiple of the number of iterations.
      We should only find one (unrolled) theta.
    */
    assert(nthetas(graph.root()) == 1);
  }

  {
    jlm::rvsdg::graph graph;
    auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
    nf->set_mutable(false);

    auto init = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0);
    auto step = jlm::rvsdg::create_bitconstant(graph.root(), 32, 1);
    auto end = jlm::rvsdg::create_bitconstant(graph.root(), 32, 100);

    auto theta = create_theta(ult, add, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 3);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is NOT a multiple of the number of iterations
      and we have one remaining iteration. We should find only the
      unrolled theta and the body of the old theta as epilogue.
    */
    assert(nthetas(graph.root()) == 1);
  }

  {
    jlm::rvsdg::graph graph;
    auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
    nf->set_mutable(false);

    auto init = jlm::rvsdg::create_bitconstant(graph.root(), 32, 100);
    auto step = jlm::rvsdg::create_bitconstant(graph.root(), 32, -1);
    auto end = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0);

    auto theta = create_theta(sgt, sub, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 6);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is NOT a multiple of the number of iterations
      and we have four remaining iterations. We should find two thetas:
      one unrolled theta and one theta for the residual iterations.
    */
    assert(nthetas(graph.root()) == 2);
  }
}

static inline void
test_unknown_boundaries()
{
  using namespace jlm::llvm;

  auto bt = jlm::rvsdg::bittype::Create(32);
  jlm::tests::test_op op({ bt }, { bt });

  RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::tests::GraphImport::Create(graph, bt, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bt, "y");

  auto theta = jlm::rvsdg::ThetaNode::create(graph.root());
  auto lv1 = theta->add_loopvar(x);
  auto lv2 = theta->add_loopvar(y);

  auto one = jlm::rvsdg::create_bitconstant(theta->subregion(), 32, 1);
  auto add = jlm::rvsdg::bitadd_op::create(32, lv1->argument(), one);
  auto cmp = jlm::rvsdg::bitult_op::create(32, add, lv2->argument());
  auto match = jlm::rvsdg::match(1, { { 1, 0 } }, 1, 2, cmp);

  lv1->result()->divert_to(add);

  theta->set_predicate(match);

  auto & ex1 = GraphExport::Create(*lv1, "x");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::loopunroll loopunroll(2);
  loopunroll.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  auto node = jlm::rvsdg::node_output::node(ex1.origin());
  assert(jlm::rvsdg::is<jlm::rvsdg::GammaOperation>(node));
  node = jlm::rvsdg::node_output::node(node->input(1)->origin());
  assert(jlm::rvsdg::is<jlm::rvsdg::GammaOperation>(node));

  /* Create cleaner output */
  DeadNodeElimination dne;
  dne.run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);
}

static std::vector<jlm::rvsdg::ThetaNode *>
find_thetas(jlm::rvsdg::region * region)
{
  std::vector<jlm::rvsdg::ThetaNode *> thetas;
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto theta = dynamic_cast<jlm::rvsdg::ThetaNode *>(node))
      thetas.push_back(theta);
  }

  return thetas;
}

static inline void
test_nested_theta()
{
  jlm::llvm::RvsdgModule rm(jlm::util::filepath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  auto init = jlm::rvsdg::create_bitconstant(graph.root(), 32, 0);
  auto step = jlm::rvsdg::create_bitconstant(graph.root(), 32, 1);
  auto end = jlm::rvsdg::create_bitconstant(graph.root(), 32, 97);

  /* Outer loop */
  auto otheta = jlm::rvsdg::ThetaNode::create(graph.root());

  auto lvo_init = otheta->add_loopvar(init);
  auto lvo_step = otheta->add_loopvar(step);
  auto lvo_end = otheta->add_loopvar(end);

  auto add = jlm::rvsdg::bitadd_op::create(32, lvo_init->argument(), lvo_step->argument());
  auto compare = jlm::rvsdg::bitult_op::create(32, add, lvo_end->argument());
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, compare);
  otheta->set_predicate(match);
  lvo_init->result()->divert_to(add);

  /* First inner loop in the original loop */
  auto inner_theta = jlm::rvsdg::ThetaNode::create(otheta->subregion());

  auto inner_init = jlm::rvsdg::create_bitconstant(otheta->subregion(), 32, 0);
  auto lvi_init = inner_theta->add_loopvar(inner_init);
  auto lvi_step = inner_theta->add_loopvar(lvo_step->argument());
  auto lvi_end = inner_theta->add_loopvar(lvo_end->argument());

  auto inner_add = jlm::rvsdg::bitadd_op::create(32, lvi_init->argument(), lvi_step->argument());
  auto inner_compare = jlm::rvsdg::bitult_op::create(32, inner_add, lvi_end->argument());
  auto inner_match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, inner_compare);
  inner_theta->set_predicate(inner_match);
  lvi_init->result()->divert_to(inner_add);

  /* Nested inner loop */
  auto inner_nested_theta = jlm::rvsdg::ThetaNode::create(inner_theta->subregion());

  auto inner_nested_init = jlm::rvsdg::create_bitconstant(inner_theta->subregion(), 32, 0);
  auto lvi_nested_init = inner_nested_theta->add_loopvar(inner_nested_init);
  auto lvi_nested_step = inner_nested_theta->add_loopvar(lvi_step->argument());
  auto lvi_nested_end = inner_nested_theta->add_loopvar(lvi_end->argument());

  auto inner_nested_add =
      jlm::rvsdg::bitadd_op::create(32, lvi_nested_init->argument(), lvi_nested_step->argument());
  auto inner_nested_compare =
      jlm::rvsdg::bitult_op::create(32, inner_nested_add, lvi_nested_end->argument());
  auto inner_nested_match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, inner_nested_compare);
  inner_nested_theta->set_predicate(inner_nested_match);
  lvi_nested_init->result()->divert_to(inner_nested_add);

  /* Second inner loop in the original loop */
  auto inner2_theta = jlm::rvsdg::ThetaNode::create(otheta->subregion());

  auto inner2_init = jlm::rvsdg::create_bitconstant(otheta->subregion(), 32, 0);
  auto lvi2_init = inner2_theta->add_loopvar(inner2_init);
  auto lvi2_step = inner2_theta->add_loopvar(lvo_step->argument());
  auto lvi2_end = inner2_theta->add_loopvar(lvo_end->argument());

  auto inner2_add = jlm::rvsdg::bitadd_op::create(32, lvi2_init->argument(), lvi2_step->argument());
  auto inner2_compare = jlm::rvsdg::bitult_op::create(32, inner2_add, lvi2_end->argument());
  auto inner2_match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, inner2_compare);
  inner2_theta->set_predicate(inner2_match);
  lvi2_init->result()->divert_to(inner2_add);

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::loopunroll loopunroll(4);
  loopunroll.run(rm, statisticsCollector);
  /*
    The outher theta should contain two inner thetas
  */
  assert(nthetas(otheta->subregion()) == 2);
  /*
    The outer theta should not be unrolled and since the
    original graph contains 7 nodes and the unroll factor
    is 4 an unrolled theta should have around 28 nodes. So
    we check for less than 20 nodes in case an updated
    unroll algorithm would hoist code from the innner
    thetas.
  */
  assert(otheta->subregion()->nnodes() <= 20);
  /*
    The inner theta should not be unrolled and since the
    original graph contains 5 nodes and the unroll factor
    is 4 an unrolled theta should have around 20 nodes. So
    we check for less than 15 nodes in case an updated
    unroll algorithm would hoist code from the innner
    thetas.
  */
  assert(inner_theta->subregion()->nnodes() <= 15);
  /*
    The innermost theta should be unrolled and since the
    original graph contains 3 nodes and the unroll factor
    is 4 an unrolled theta should have around 12 nodes. So
    we check for more than 7 nodes in case an updated
    unroll algorithm would hoist code from the innner
    thetas.
  */
  auto thetas = find_thetas(inner_theta->subregion());
  assert(thetas.size() == 1 && thetas[0]->subregion()->nnodes() >= 7);
  /*
    The second inner theta should be unrolled and since
    the original graph contains 3 nodes and the unroll
    factor is 4 an unrolled theta should have around 12
    nodes. So we check for less than 7 nodes in case an
    updated unroll algorithm would hoist code from the
    innner thetas.
  */
  thetas = find_thetas(otheta->subregion());
  assert(thetas.size() == 2 && thetas[1]->subregion()->nnodes() >= 7);
  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::unroll(otheta, 4);
  //	jlm::rvsdg::view(graph, stdout);
  /*
    After unrolling the outher theta four times it should
    now contain 8 thetas.
  */
  thetas = find_thetas(graph.root());
  assert(thetas.size() == 3 && nthetas(thetas[0]->subregion()) == 8);
}

static int
verify()
{
  test_unrollinfo();

  test_nested_theta();
  test_known_boundaries();
  test_unknown_boundaries();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/test-unroll", verify)
