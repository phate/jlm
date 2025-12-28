/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

static void
test_main()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto type = jlm::rvsdg::TestType::createStateType();
  auto value_type = TestType::createValueType();

  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), {}, { type });

  bool error_handler_called = false;
  try
  {
    TestOperation::createNode(&graph.GetRootRegion(), { value_type }, { n1->output(0) }, {});
  }
  catch (jlm::util::TypeError & e)
  {
    error_handler_called = true;
  }

  assert(error_handler_called);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-typemismatch", test_main)
