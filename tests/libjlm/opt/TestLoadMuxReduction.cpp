/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/operators/load.hpp>
#include <jlm/ir/operators/operators.hpp>

#include <jive/view.hpp>

static inline void
test_load_mux_reduction()
{
  using namespace jlm;

  jlm::valuetype vt;
  jlm::ptrtype pt(vt);
  jive::memtype mt;

  jive::graph graph;
  auto nf = jlm::load_op::normal_form(&graph);
  nf->set_mutable(false);
  nf->set_load_mux_reducible(false);

  auto a = graph.add_import({pt, "a"});
  auto s1 = graph.add_import({mt, "s1"});
  auto s2 = graph.add_import({mt, "s2"});
  auto s3 = graph.add_import({mt, "s3"});

  auto mux = MemStateMergeOperator::Create({s1, s2, s3});
  auto value = load_op::create(a, {mux}, 4)[0];

  auto ex = graph.add_export(value, {value->type(), "v"});

  // jive::view(graph.root(), stdout);

  nf->set_mutable(true);
  nf->set_load_mux_reducible(true);
  graph.normalize();
  graph.prune();

  // jive::view(graph.root(), stdout);

  auto load = jive::node_output::node(ex->origin());
  assert(jive::is<jlm::load_op>(load));
  assert(load->ninputs() == 4);
  assert(load->input(1)->origin() == s1);
  assert(load->input(2)->origin() == s2);
  assert(load->input(3)->origin() == s3);
}

static int
test()
{
  test_load_mux_reduction();

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/TestLoadMuxReduction", test)