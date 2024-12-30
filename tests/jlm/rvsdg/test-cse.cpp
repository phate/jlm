/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

static int
test_main()
{
  using namespace jlm::rvsdg;

  auto t = jlm::tests::valuetype::Create();

  Graph graph;
  auto i = &jlm::tests::GraphImport::Create(graph, t, "i");

  auto o1 = jlm::tests::test_op::create(&graph.GetRootRegion(), {}, { t })->output(0);
  auto o2 = jlm::tests::test_op::create(&graph.GetRootRegion(), { i }, { t })->output(0);

  auto & e1 = jlm::tests::GraphExport::Create(*o1, "o1");
  auto & e2 = jlm::tests::GraphExport::Create(*o2, "o2");

  auto nf = dynamic_cast<jlm::rvsdg::simple_normal_form *>(
      graph.GetNodeNormalForm(typeid(jlm::tests::test_op)));
  nf->set_mutable(false);

  auto o3 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { t })[0];
  auto o4 = jlm::tests::create_testop(&graph.GetRootRegion(), { i }, { t })[0];

  auto & e3 = jlm::tests::GraphExport::Create(*o3, "o3");
  auto & e4 = jlm::tests::GraphExport::Create(*o4, "o4");

  nf->set_mutable(true);
  graph.Normalize();
  assert(e1.origin() == e3.origin());
  assert(e2.origin() == e4.origin());

  auto o5 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { t })[0];
  assert(o5 == e1.origin());

  auto o6 = jlm::tests::create_testop(&graph.GetRootRegion(), { i }, { t })[0];
  assert(o6 == e2.origin());

  nf->set_cse(false);

  auto o7 = jlm::tests::create_testop(&graph.GetRootRegion(), {}, { t })[0];
  assert(o7 != e1.origin());

  graph.Normalize();
  assert(o7 != e1.origin());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-cse", test_main)
