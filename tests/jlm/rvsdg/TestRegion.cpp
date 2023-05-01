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

/**
 * Test region::Contains().
 */
static void
TestContainsMethod()
{
  using namespace jlm;

  valuetype vt;

  jive::graph graph;
  auto import = graph.add_import({vt, "import"});

  auto structuralNode1 = structural_node::create(graph.root(), 1);
  auto structuralInput1 = jive::structural_input::create(structuralNode1, import, vt);
  auto regionArgument1 = jive::argument::create(structuralNode1->subregion(0), structuralInput1, vt);
  unary_op::create(structuralNode1->subregion(0), {vt}, regionArgument1, {vt});

  auto structuralNode2 = jlm::structural_node::create(graph.root(), 1);
  auto structuralInput2 = jive::structural_input::create(structuralNode2, import, vt);
  auto regionArgument2 = jive::argument::create(structuralNode2->subregion(0), structuralInput2, vt);
  binary_op::create({vt}, {vt}, regionArgument2, regionArgument2);

  assert(jive::region::Contains<structural_op>(*graph.root(), false));
  assert(jive::region::Contains<unary_op>(*graph.root(), true));
  assert(jive::region::Contains<binary_op>(*graph.root(), true));
  assert(!jive::region::Contains<test_op>(*graph.root(), true));
}

/**
 * Test region::IsRootRegion().
 */
static void
TestIsRootRegion()
{
  jive::graph graph;

  auto structuralNode = jlm::structural_node::create(graph.root(), 1);

  assert(graph.root()->IsRootRegion());
  assert(!structuralNode->subregion(0)->IsRootRegion());
}

/**
 * Test region::NumRegions()
 */
static void
TestNumRegions()
{
  using namespace jive;

  {
    jive::graph graph;

    assert(region::NumRegions(*graph.root()) == 1);
  }

  {
    jive::graph graph;
    auto structuralNode = jlm::structural_node::create(graph.root(), 4);
    jlm::structural_node::create(structuralNode->subregion(0), 2);
    jlm::structural_node::create(structuralNode->subregion(3), 5);

    assert(region::NumRegions(*graph.root()) == 1+4+2+5);
  }
}

static int
Test()
{
  TestArgumentNodeMismatch();
  TestResultNodeMismatch();

  TestContainsMethod();
  TestIsRootRegion();
  TestNumRegions();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TestRegion", Test)
