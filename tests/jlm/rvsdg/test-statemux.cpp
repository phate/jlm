/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/statemux.hpp>
#include <jlm/rvsdg/view.hpp>

static void
test_mux_mux_reduction()
{
  using namespace jlm::rvsdg;

  auto st = jlm::tests::statetype::Create();

  Graph graph;
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::mux_op));
  auto mnf = static_cast<jlm::rvsdg::mux_normal_form *>(nf);
  mnf->set_mutable(false);
  mnf->set_mux_mux_reducible(false);

  auto x = &jlm::tests::GraphImport::Create(graph, st, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, st, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, st, "z");

  auto mux1 = jlm::rvsdg::create_state_merge(st, { x, y });
  auto mux2 = jlm::rvsdg::create_state_split(st, z, 2);
  auto mux3 = jlm::rvsdg::create_state_merge(st, { mux1, mux2[0], mux2[1], z });

  auto & ex = jlm::tests::GraphExport::Create(*mux3, "m");

  //	jlm::rvsdg::view(graph.root(), stdout);

  mnf->set_mutable(true);
  mnf->set_mux_mux_reducible(true);
  graph.normalize();
  graph.prune();

  //	jlm::rvsdg::view(graph.root(), stdout);

  auto node = output::GetNode(*ex.origin());
  assert(node->ninputs() == 4);
  assert(node->input(0)->origin() == x);
  assert(node->input(1)->origin() == y);
  assert(node->input(2)->origin() == z);
  assert(node->input(3)->origin() == z);
}

static void
test_multiple_origin_reduction()
{
  using namespace jlm::rvsdg;

  auto st = jlm::tests::statetype::Create();

  Graph graph;
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::mux_op));
  auto mnf = static_cast<jlm::rvsdg::mux_normal_form *>(nf);
  mnf->set_mutable(false);
  mnf->set_multiple_origin_reducible(false);

  auto x = &jlm::tests::GraphImport::Create(graph, st, "x");
  auto mux1 = jlm::rvsdg::create_state_merge(st, { x, x });
  auto & ex = jlm::tests::GraphExport::Create(*mux1, "m");

  view(graph.root(), stdout);

  mnf->set_mutable(true);
  mnf->set_multiple_origin_reducible(true);
  graph.normalize();
  graph.prune();

  view(graph.root(), stdout);

  assert(output::GetNode(*ex.origin())->ninputs() == 1);
}

static int
test_main()
{
  test_mux_mux_reduction();
  test_multiple_origin_reduction();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-statemux", test_main)
