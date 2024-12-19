/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/statemux.hpp>
#include <jlm/rvsdg/view.hpp>

#include <cassert>

static int
MuxMuxReduction()
{
  using namespace jlm::rvsdg;

  auto stateType = jlm::tests::statetype::Create();

  // Arrange
  Graph graph;
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::mux_op));
  auto mnf = static_cast<jlm::rvsdg::mux_normal_form *>(nf);
  mnf->set_mutable(false);
  mnf->set_mux_mux_reducible(false);

  auto x = &jlm::tests::GraphImport::Create(graph, stateType, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, stateType, "y");
  auto z = &jlm::tests::GraphImport::Create(graph, stateType, "z");

  auto mux1 = jlm::rvsdg::create_state_merge(stateType, { x, y });
  auto mux2 = jlm::rvsdg::create_state_split(stateType, z, 2);
  auto mux3 = jlm::rvsdg::create_state_merge(stateType, { mux1, mux2[0], mux2[1], z });

  auto & ex = jlm::tests::GraphExport::Create(*mux3, "m");

  view(graph.root(), stdout);

  // Act
  bool success = false;
  do
  {
    auto muxNode = output::GetNode(*ex.origin());
    success = ReduceNode<mux_op>(NormalizeMuxMux, *muxNode);
  } while (success);

  view(graph.root(), stdout);

  // Assert
  auto node = output::GetNode(*ex.origin());
  assert(node->ninputs() == 4);
  assert(node->input(0)->origin() == x);
  assert(node->input(1)->origin() == y);
  assert(node->input(2)->origin() == z);
  assert(node->input(3)->origin() == z);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-statemux-MuxMuxReduction", MuxMuxReduction)

static int
DuplicateOperandReduction()
{
  using namespace jlm::rvsdg;

  auto stateType = jlm::tests::statetype::Create();

  // Arrange
  Graph graph;
  auto nf = graph.node_normal_form(typeid(jlm::rvsdg::mux_op));
  auto mnf = static_cast<jlm::rvsdg::mux_normal_form *>(nf);
  mnf->set_mutable(false);
  mnf->set_multiple_origin_reducible(false);

  auto x = &jlm::tests::GraphImport::Create(graph, stateType, "x");
  auto mux1 = jlm::rvsdg::create_state_merge(stateType, { x, x });
  auto & ex = jlm::tests::GraphExport::Create(*mux1, "m");

  view(graph.root(), stdout);

  // Act
  auto muxNode = output::GetNode(*ex.origin());
  auto success = ReduceNode<mux_op>(NormalizeMuxDuplicateOperands, *muxNode);
  graph.prune();

  view(graph.root(), stdout);

  // Assert
  assert(success);
  assert(output::GetNode(*ex.origin())->ninputs() == 1);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/test-statemux-DuplicateOperandReduction",
    DuplicateOperandReduction)
