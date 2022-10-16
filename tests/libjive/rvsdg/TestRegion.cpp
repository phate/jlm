/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include "test-operation.hpp"
#include "test-types.hpp"

#include <cassert>

/**
 * Test check for adding argument to input of wrong structural node.
 */
static void
TestArgumentNodeMismatch()
{
  using namespace jive;

  jlm::valuetype vt;

  jive::graph graph;
  auto import = graph.add_import({vt, "import"});

  auto structuralNode1 = jlm::structural_node::create(graph.root(), 1);
  auto structuralNode2 = jlm::structural_node::create(graph.root(), 2);

  auto structuralInput = structural_input::create(structuralNode1, import, vt);

  bool inputErrorHandlerCalled = false;
  try {
    argument::create(structuralNode2->subregion(0), structuralInput, vt);
  } catch (jive::compiler_error & e) {
    inputErrorHandlerCalled = true;
  }

  assert(inputErrorHandlerCalled);
}

/**
 * Test check for adding result to output of wrong structural node.
 */
static void
TestResultNodeMismatch()
{
  using namespace jive;

  jlm::valuetype vt;

  jive::graph graph;
  auto import = graph.add_import({vt, "import"});

  auto structuralNode1 = jlm::structural_node::create(graph.root(), 1);
  auto structuralNode2 = jlm::structural_node::create(graph.root(), 2);

  auto structuralInput = structural_input::create(structuralNode1, import, vt);

  auto argument = argument::create(structuralNode1->subregion(0), structuralInput, vt);
  auto structuralOutput = structural_output::create(structuralNode1, vt);

  bool outputErrorHandlerCalled = false;
  try {
    result::create(structuralNode2->subregion(0), argument, structuralOutput, vt);
  } catch (jive::compiler_error & e) {
    outputErrorHandlerCalled = true;
  }

  assert(outputErrorHandlerCalled);
}

static int
Test()
{
  TestArgumentNodeMismatch();
  TestResultNodeMismatch();

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjive/rvsdg/TestRegion", Test)
