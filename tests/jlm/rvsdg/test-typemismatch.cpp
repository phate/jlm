/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/TestType.hpp>

TEST(TypeMismatchTests, test_main)
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto type = jlm::rvsdg::TestType::createStateType();
  auto value_type = TestType::createValueType();

  auto n1 = TestOperation::createNode(&graph.GetRootRegion(), {}, { type });

  // Act & Assert
  EXPECT_THROW(
      TestOperation::createNode(&graph.GetRootRegion(), { value_type }, { n1->output(0) }, {}),
      jlm::util::TypeError);
}
