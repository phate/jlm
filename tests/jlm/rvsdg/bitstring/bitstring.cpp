/*
 * Copyright 2010 2011 2012 2013 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include <test-operation.hpp>

#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/view.hpp>

static int
types_bitstring_arithmetic_test_bitand()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto and0 = bitand_op::create(32, s0, s1);
  auto and1 = bitand_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*and0, "dummy");
  jlm::tests::GraphExport::Create(*and1, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*and0)->GetOperation() == bitand_op(32));
  assert(output::GetNode(*and1)->GetOperation() == int_constant_op(32, +1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitashr()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, -16);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto ashr0 = bitashr_op::create(32, s0, s1);
  auto ashr1 = bitashr_op::create(32, c0, c2);
  auto ashr2 = bitashr_op::create(32, c0, c3);
  auto ashr3 = bitashr_op::create(32, c1, c2);
  auto ashr4 = bitashr_op::create(32, c1, c3);

  jlm::tests::GraphExport::Create(*ashr0, "dummy");
  jlm::tests::GraphExport::Create(*ashr1, "dummy");
  jlm::tests::GraphExport::Create(*ashr2, "dummy");
  jlm::tests::GraphExport::Create(*ashr3, "dummy");
  jlm::tests::GraphExport::Create(*ashr4, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*ashr0)->GetOperation() == bitashr_op(32));
  assert(output::GetNode(*ashr1)->GetOperation() == int_constant_op(32, 4));
  assert(output::GetNode(*ashr2)->GetOperation() == int_constant_op(32, 0));
  assert(output::GetNode(*ashr3)->GetOperation() == int_constant_op(32, -4));
  assert(output::GetNode(*ashr4)->GetOperation() == int_constant_op(32, -1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitdifference()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto diff = bitsub_op::create(32, s0, s1);

  jlm::tests::GraphExport::Create(*diff, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*diff)->GetOperation() == bitsub_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitnegate()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto neg0 = bitneg_op::create(32, s0);
  auto neg1 = bitneg_op::create(32, c0);
  auto neg2 = bitneg_op::create(32, neg1);

  jlm::tests::GraphExport::Create(*neg0, "dummy");
  jlm::tests::GraphExport::Create(*neg1, "dummy");
  jlm::tests::GraphExport::Create(*neg2, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*neg0)->GetOperation() == bitneg_op(32));
  assert(output::GetNode(*neg1)->GetOperation() == int_constant_op(32, -3));
  assert(output::GetNode(*neg2)->GetOperation() == int_constant_op(32, 3));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitnot()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto not0 = bitnot_op::create(32, s0);
  auto not1 = bitnot_op::create(32, c0);
  auto not2 = bitnot_op::create(32, not1);

  jlm::tests::GraphExport::Create(*not0, "dummy");
  jlm::tests::GraphExport::Create(*not1, "dummy");
  jlm::tests::GraphExport::Create(*not2, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*not0)->GetOperation() == bitnot_op(32));
  assert(output::GetNode(*not1)->GetOperation() == int_constant_op(32, -4));
  assert(output::GetNode(*not2)->GetOperation() == int_constant_op(32, 3));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitor()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto or0 = bitor_op::create(32, s0, s1);
  auto or1 = bitor_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*or0, "dummy");
  jlm::tests::GraphExport::Create(*or1, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*or0)->GetOperation() == bitor_op(32));
  assert(output::GetNode(*or1)->GetOperation() == uint_constant_op(32, 7));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitproduct()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto product0 = bitmul_op::create(32, s0, s1);
  auto product1 = bitmul_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*product0, "dummy");
  jlm::tests::GraphExport::Create(*product1, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*product0)->GetOperation() == bitmul_op(32));
  assert(output::GetNode(*product1)->GetOperation() == uint_constant_op(32, 15));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshiproduct()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto shiproduct = bitsmulh_op::create(32, s0, s1);

  jlm::tests::GraphExport::Create(*shiproduct, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*shiproduct)->GetOperation() == bitsmulh_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshl()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto shl0 = bitshl_op::create(32, s0, s1);
  auto shl1 = bitshl_op::create(32, c0, c1);
  auto shl2 = bitshl_op::create(32, c0, c2);

  jlm::tests::GraphExport::Create(*shl0, "dummy");
  jlm::tests::GraphExport::Create(*shl1, "dummy");
  jlm::tests::GraphExport::Create(*shl2, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*shl0)->GetOperation() == bitshl_op(32));
  assert(output::GetNode(*shl1)->GetOperation() == uint_constant_op(32, 64));
  assert(output::GetNode(*shl2)->GetOperation() == uint_constant_op(32, 0));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshr()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto shr0 = bitshr_op::create(32, s0, s1);
  auto shr1 = bitshr_op::create(32, c0, c1);
  auto shr2 = bitshr_op::create(32, c0, c2);

  jlm::tests::GraphExport::Create(*shr0, "dummy");
  jlm::tests::GraphExport::Create(*shr1, "dummy");
  jlm::tests::GraphExport::Create(*shr2, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*shr0)->GetOperation() == bitshr_op(32));
  assert(output::GetNode(*shr1)->GetOperation() == uint_constant_op(32, 4));
  assert(output::GetNode(*shr2)->GetOperation() == uint_constant_op(32, 0));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsmod()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, -7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto smod0 = bitsmod_op::create(32, s0, s1);
  auto smod1 = bitsmod_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*smod0, "dummy");
  jlm::tests::GraphExport::Create(*smod1, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*smod0)->GetOperation() == bitsmod_op(32));
  assert(output::GetNode(*smod1)->GetOperation() == int_constant_op(32, -1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsquotient()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, -3);

  auto squot0 = bitsdiv_op::create(32, s0, s1);
  auto squot1 = bitsdiv_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*squot0, "dummy");
  jlm::tests::GraphExport::Create(*squot1, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*squot0)->GetOperation() == bitsdiv_op(32));
  assert(output::GetNode(*squot1)->GetOperation() == int_constant_op(32, -2));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsum()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto sum0 = bitadd_op::create(32, s0, s1);
  auto sum1 = bitadd_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*sum0, "dummy");
  jlm::tests::GraphExport::Create(*sum1, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*sum0)->GetOperation() == bitadd_op(32));
  assert(output::GetNode(*sum1)->GetOperation() == int_constant_op(32, 8));

  return 0;
}

static int
types_bitstring_arithmetic_test_bituhiproduct()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto uhiproduct = bitumulh_op::create(32, s0, s1);

  jlm::tests::GraphExport::Create(*uhiproduct, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*uhiproduct)->GetOperation() == bitumulh_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitumod()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto umod0 = bitumod_op::create(32, s0, s1);
  auto umod1 = bitumod_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*umod0, "dummy");
  jlm::tests::GraphExport::Create(*umod1, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*umod0)->GetOperation() == bitumod_op(32));
  assert(output::GetNode(*umod1)->GetOperation() == int_constant_op(32, 1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bituquotient()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto uquot0 = bitudiv_op::create(32, s0, s1);
  auto uquot1 = bitudiv_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*uquot0, "dummy");
  jlm::tests::GraphExport::Create(*uquot1, "dummy");

  graph.Normalize();
  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*uquot0)->GetOperation() == bitudiv_op(32));
  assert(output::GetNode(*uquot1)->GetOperation() == int_constant_op(32, 2));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitxor()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto xor0 = bitxor_op::create(32, s0, s1);
  auto xor1 = bitxor_op::create(32, c0, c1);

  jlm::tests::GraphExport::Create(*xor0, "dummy");
  jlm::tests::GraphExport::Create(*xor1, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*xor0)->GetOperation() == bitxor_op(32));
  assert(output::GetNode(*xor1)->GetOperation() == int_constant_op(32, 6));

  return 0;
}

static inline void
expect_static_true(jlm::rvsdg::output * port)
{
  auto node = jlm::rvsdg::output::GetNode(*port);
  auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&node->GetOperation());
  assert(op && op->value().nbits() == 1 && op->value().str() == "1");
}

static inline void
expect_static_false(jlm::rvsdg::output * port)
{
  auto node = jlm::rvsdg::output::GetNode(*port);
  auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&node->GetOperation());
  assert(op && op->value().nbits() == 1 && op->value().str() == "0");
}

static int
types_bitstring_comparison_test_bitequal()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant_undefined(&graph.GetRootRegion(), 32);

  auto equal0 = biteq_op::create(32, s0, s1);
  auto equal1 = biteq_op::create(32, c0, c0);
  auto equal2 = biteq_op::create(32, c0, c1);
  auto equal3 = biteq_op::create(32, c0, c2);

  jlm::tests::GraphExport::Create(*equal0, "dummy");
  jlm::tests::GraphExport::Create(*equal1, "dummy");
  jlm::tests::GraphExport::Create(*equal2, "dummy");
  jlm::tests::GraphExport::Create(*equal3, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*equal0)->GetOperation() == biteq_op(32));
  expect_static_true(equal1);
  expect_static_false(equal2);
  assert(output::GetNode(*equal3)->GetOperation() == biteq_op(32));

  return 0;
}

static int
types_bitstring_comparison_test_bitnotequal()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant_undefined(&graph.GetRootRegion(), 32);

  auto nequal0 = bitne_op::create(32, s0, s1);
  auto nequal1 = bitne_op::create(32, c0, c0);
  auto nequal2 = bitne_op::create(32, c0, c1);
  auto nequal3 = bitne_op::create(32, c0, c2);

  jlm::tests::GraphExport::Create(*nequal0, "dummy");
  jlm::tests::GraphExport::Create(*nequal1, "dummy");
  jlm::tests::GraphExport::Create(*nequal2, "dummy");
  jlm::tests::GraphExport::Create(*nequal3, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*nequal0)->GetOperation() == bitne_op(32));
  expect_static_false(nequal1);
  expect_static_true(nequal2);
  assert(output::GetNode(*nequal3)->GetOperation() == bitne_op(32));

  return 0;
}

static int
types_bitstring_comparison_test_bitsgreater()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto sgreater0 = bitsgt_op::create(32, s0, s1);
  auto sgreater1 = bitsgt_op::create(32, c0, c1);
  auto sgreater2 = bitsgt_op::create(32, c1, c0);
  auto sgreater3 = bitsgt_op::create(32, s0, c2);
  auto sgreater4 = bitsgt_op::create(32, c3, s1);

  jlm::tests::GraphExport::Create(*sgreater0, "dummy");
  jlm::tests::GraphExport::Create(*sgreater1, "dummy");
  jlm::tests::GraphExport::Create(*sgreater2, "dummy");
  jlm::tests::GraphExport::Create(*sgreater3, "dummy");
  jlm::tests::GraphExport::Create(*sgreater4, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*sgreater0)->GetOperation() == bitsgt_op(32));
  expect_static_false(sgreater1);
  expect_static_true(sgreater2);
  expect_static_false(sgreater3);
  expect_static_false(sgreater4);

  return 0;
}

static int
types_bitstring_comparison_test_bitsgreatereq()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto sgreatereq0 = bitsge_op::create(32, s0, s1);
  auto sgreatereq1 = bitsge_op::create(32, c0, c1);
  auto sgreatereq2 = bitsge_op::create(32, c1, c0);
  auto sgreatereq3 = bitsge_op::create(32, c0, c0);
  auto sgreatereq4 = bitsge_op::create(32, c2, s0);
  auto sgreatereq5 = bitsge_op::create(32, s1, c3);

  jlm::tests::GraphExport::Create(*sgreatereq0, "dummy");
  jlm::tests::GraphExport::Create(*sgreatereq1, "dummy");
  jlm::tests::GraphExport::Create(*sgreatereq2, "dummy");
  jlm::tests::GraphExport::Create(*sgreatereq3, "dummy");
  jlm::tests::GraphExport::Create(*sgreatereq4, "dummy");
  jlm::tests::GraphExport::Create(*sgreatereq5, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*sgreatereq0)->GetOperation() == bitsge_op(32));
  expect_static_false(sgreatereq1);
  expect_static_true(sgreatereq2);
  expect_static_true(sgreatereq3);
  expect_static_true(sgreatereq4);
  expect_static_true(sgreatereq5);

  return 0;
}

static int
types_bitstring_comparison_test_bitsless()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto sless0 = bitslt_op::create(32, s0, s1);
  auto sless1 = bitslt_op::create(32, c0, c1);
  auto sless2 = bitslt_op::create(32, c1, c0);
  auto sless3 = bitslt_op::create(32, c2, s0);
  auto sless4 = bitslt_op::create(32, s1, c3);

  jlm::tests::GraphExport::Create(*sless0, "dummy");
  jlm::tests::GraphExport::Create(*sless1, "dummy");
  jlm::tests::GraphExport::Create(*sless2, "dummy");
  jlm::tests::GraphExport::Create(*sless3, "dummy");
  jlm::tests::GraphExport::Create(*sless4, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*sless0)->GetOperation() == bitslt_op(32));
  expect_static_true(sless1);
  expect_static_false(sless2);
  expect_static_false(sless3);
  expect_static_false(sless4);

  return 0;
}

static int
types_bitstring_comparison_test_bitslesseq()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto slesseq0 = bitsle_op::create(32, s0, s1);
  auto slesseq1 = bitsle_op::create(32, c0, c1);
  auto slesseq2 = bitsle_op::create(32, c0, c0);
  auto slesseq3 = bitsle_op::create(32, c1, c0);
  auto slesseq4 = bitsle_op::create(32, s0, c2);
  auto slesseq5 = bitsle_op::create(32, c3, s1);

  jlm::tests::GraphExport::Create(*slesseq0, "dummy");
  jlm::tests::GraphExport::Create(*slesseq1, "dummy");
  jlm::tests::GraphExport::Create(*slesseq2, "dummy");
  jlm::tests::GraphExport::Create(*slesseq3, "dummy");
  jlm::tests::GraphExport::Create(*slesseq4, "dummy");
  jlm::tests::GraphExport::Create(*slesseq5, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*slesseq0)->GetOperation() == bitsle_op(32));
  expect_static_true(slesseq1);
  expect_static_true(slesseq2);
  expect_static_false(slesseq3);
  expect_static_true(slesseq4);
  expect_static_true(slesseq5);

  return 0;
}

static int
types_bitstring_comparison_test_bitugreater()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, (0xffffffffUL));
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 0);

  auto ugreater0 = bitugt_op::create(32, s0, s1);
  auto ugreater1 = bitugt_op::create(32, c0, c1);
  auto ugreater2 = bitugt_op::create(32, c1, c0);
  auto ugreater3 = bitugt_op::create(32, s0, c2);
  auto ugreater4 = bitugt_op::create(32, c3, s1);

  jlm::tests::GraphExport::Create(*ugreater0, "dummy");
  jlm::tests::GraphExport::Create(*ugreater1, "dummy");
  jlm::tests::GraphExport::Create(*ugreater2, "dummy");
  jlm::tests::GraphExport::Create(*ugreater3, "dummy");
  jlm::tests::GraphExport::Create(*ugreater4, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*ugreater0)->GetOperation() == bitugt_op(32));
  expect_static_false(ugreater1);
  expect_static_true(ugreater2);
  expect_static_false(ugreater3);
  expect_static_false(ugreater4);

  return 0;
}

static int
types_bitstring_comparison_test_bitugreatereq()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, (0xffffffffUL));
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 0);

  auto ugreatereq0 = bituge_op::create(32, s0, s1);
  auto ugreatereq1 = bituge_op::create(32, c0, c1);
  auto ugreatereq2 = bituge_op::create(32, c1, c0);
  auto ugreatereq3 = bituge_op::create(32, c0, c0);
  auto ugreatereq4 = bituge_op::create(32, c2, s0);
  auto ugreatereq5 = bituge_op::create(32, s1, c3);

  jlm::tests::GraphExport::Create(*ugreatereq0, "dummy");
  jlm::tests::GraphExport::Create(*ugreatereq1, "dummy");
  jlm::tests::GraphExport::Create(*ugreatereq2, "dummy");
  jlm::tests::GraphExport::Create(*ugreatereq3, "dummy");
  jlm::tests::GraphExport::Create(*ugreatereq4, "dummy");
  jlm::tests::GraphExport::Create(*ugreatereq5, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*ugreatereq0)->GetOperation() == bituge_op(32));
  expect_static_false(ugreatereq1);
  expect_static_true(ugreatereq2);
  expect_static_true(ugreatereq3);
  expect_static_true(ugreatereq4);
  expect_static_true(ugreatereq5);

  return 0;
}

static int
types_bitstring_comparison_test_bituless()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, (0xffffffffUL));
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 0);

  auto uless0 = bitult_op::create(32, s0, s1);
  auto uless1 = bitult_op::create(32, c0, c1);
  auto uless2 = bitult_op::create(32, c1, c0);
  auto uless3 = bitult_op::create(32, c2, s0);
  auto uless4 = bitult_op::create(32, s1, c3);

  jlm::tests::GraphExport::Create(*uless0, "dummy");
  jlm::tests::GraphExport::Create(*uless1, "dummy");
  jlm::tests::GraphExport::Create(*uless2, "dummy");
  jlm::tests::GraphExport::Create(*uless3, "dummy");
  jlm::tests::GraphExport::Create(*uless4, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*uless0)->GetOperation() == bitult_op(32));
  expect_static_true(uless1);
  expect_static_false(uless2);
  expect_static_false(uless3);
  expect_static_false(uless4);

  return 0;
}

static int
types_bitstring_comparison_test_bitulesseq()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, (0xffffffffUL));
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 0);

  auto ulesseq0 = bitule_op::create(32, s0, s1);
  auto ulesseq1 = bitule_op::create(32, c0, c1);
  auto ulesseq2 = bitule_op::create(32, c0, c0);
  auto ulesseq3 = bitule_op::create(32, c1, c0);
  auto ulesseq4 = bitule_op::create(32, s0, c2);
  auto ulesseq5 = bitule_op::create(32, c3, s1);

  jlm::tests::GraphExport::Create(*ulesseq0, "dummy");
  jlm::tests::GraphExport::Create(*ulesseq1, "dummy");
  jlm::tests::GraphExport::Create(*ulesseq2, "dummy");
  jlm::tests::GraphExport::Create(*ulesseq3, "dummy");
  jlm::tests::GraphExport::Create(*ulesseq4, "dummy");
  jlm::tests::GraphExport::Create(*ulesseq5, "dummy");

  graph.PruneNodes();
  view(&graph.GetRootRegion(), stdout);

  assert(output::GetNode(*ulesseq0)->GetOperation() == bitule_op(32));
  expect_static_true(ulesseq1);
  expect_static_true(ulesseq2);
  expect_static_false(ulesseq3);
  expect_static_true(ulesseq4);
  expect_static_true(ulesseq5);

  return 0;
}

#define ZERO_64 \
  "00000000"    \
  "00000000"    \
  "00000000"    \
  "00000000"    \
  "00000000"    \
  "00000000"    \
  "00000000"    \
  "00000000"
#define ONE_64 \
  "10000000"   \
  "00000000"   \
  "00000000"   \
  "00000000"   \
  "00000000"   \
  "00000000"   \
  "00000000"   \
  "00000000"
#define MONE_64 \
  "11111111"    \
  "11111111"    \
  "11111111"    \
  "11111111"    \
  "11111111"    \
  "11111111"    \
  "11111111"    \
  "11111111"

static int
types_bitstring_test_constant()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto b1 = output::GetNode(*create_bitconstant(&graph.GetRootRegion(), "00110011"));
  auto b2 = output::GetNode(*create_bitconstant(&graph.GetRootRegion(), 8, 204));
  auto b3 = output::GetNode(*create_bitconstant(&graph.GetRootRegion(), 8, 204));
  auto b4 = output::GetNode(*create_bitconstant(&graph.GetRootRegion(), "001100110"));

  assert(b1->GetOperation() == uint_constant_op(8, 204));
  assert(b1->GetOperation() == int_constant_op(8, -52));

  assert(b1 == b2);
  assert(b1 == b3);

  assert(b1->GetOperation() == uint_constant_op(8, 204));
  assert(b1->GetOperation() == int_constant_op(8, -52));

  assert(b4->GetOperation() == uint_constant_op(9, 204));
  assert(b4->GetOperation() == int_constant_op(9, 204));

  auto plus_one_128 = output::GetNode(*create_bitconstant(&graph.GetRootRegion(), ONE_64 ZERO_64));
  assert(plus_one_128->GetOperation() == uint_constant_op(128, 1));
  assert(plus_one_128->GetOperation() == int_constant_op(128, 1));

  auto minus_one_128 =
      output::GetNode(*create_bitconstant(&graph.GetRootRegion(), MONE_64 MONE_64));
  assert(minus_one_128->GetOperation() == int_constant_op(128, -1));

  view(&graph.GetRootRegion(), stdout);

  return 0;
}

static int
types_bitstring_test_normalize()
{
  using namespace jlm::rvsdg;

  Graph graph;

  bittype bits32(32);
  auto imp = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "imp");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 4);

  auto sum_nf = graph.GetNodeNormalForm(typeid(bitadd_op));
  assert(sum_nf);
  sum_nf->set_mutable(false);

  auto sum0 = output::GetNode(*bitadd_op::create(32, imp, c0));
  assert(sum0->GetOperation() == bitadd_op(32));
  assert(sum0->ninputs() == 2);

  auto sum1 = output::GetNode(*bitadd_op::create(32, sum0->output(0), c1));
  assert(sum1->GetOperation() == bitadd_op(32));
  assert(sum1->ninputs() == 2);

  auto & exp = jlm::tests::GraphExport::Create(*sum1->output(0), "dummy");

  sum_nf->set_mutable(true);
  graph.Normalize();
  graph.PruneNodes();

  auto origin = dynamic_cast<node_output *>(exp.origin());
  assert(origin->node()->GetOperation() == bitadd_op(32));
  assert(origin->node()->ninputs() == 2);
  auto op1 = origin->node()->input(0)->origin();
  auto op2 = origin->node()->input(1)->origin();
  if (!is<node_output>(op1))
  {
    auto tmp = op1;
    op1 = op2;
    op2 = tmp;
  }
  /* FIXME: the graph traversers are currently broken, that is why it won't normalize */
  assert(output::GetNode(*op1)->GetOperation() == int_constant_op(32, 3 + 4));
  assert(op2 == imp);

  view(&graph.GetRootRegion(), stdout);

  return 0;
}

static void
assert_constant(jlm::rvsdg::output * bitstr, size_t nbits, const char bits[])
{
  auto node = jlm::rvsdg::output::GetNode(*bitstr);
  auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op &>(node->GetOperation());
  assert(op.value() == jlm::rvsdg::bitvalue_repr(std::string(bits, nbits).c_str()));
}

static int
types_bitstring_test_reduction()
{
  using namespace jlm::rvsdg;

  Graph graph;

  auto a = create_bitconstant(&graph.GetRootRegion(), "1100");
  auto b = create_bitconstant(&graph.GetRootRegion(), "1010");

  assert_constant(bitand_op::create(4, a, b), 4, "1000");
  assert_constant(bitor_op::create(4, a, b), 4, "1110");
  assert_constant(bitxor_op::create(4, a, b), 4, "0110");
  assert_constant(bitadd_op::create(4, a, b), 4, "0001");
  assert_constant(bitmul_op::create(4, a, b), 4, "1111");
  assert_constant(jlm::rvsdg::bitconcat({ a, b }), 8, "11001010");
  assert_constant(bitneg_op::create(4, a), 4, "1011");
  assert_constant(bitneg_op::create(4, b), 4, "1101");

  graph.PruneNodes();

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(16), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bittype::Create(16), "y");

  {
    auto concat = jlm::rvsdg::bitconcat({ x, y });
    auto node = output::GetNode(*jlm::rvsdg::bitslice(concat, 8, 24));
    auto o0 = dynamic_cast<node_output *>(node->input(0)->origin());
    auto o1 = dynamic_cast<node_output *>(node->input(1)->origin());
    assert(dynamic_cast<const bitconcat_op *>(&node->GetOperation()));
    assert(node->ninputs() == 2);
    assert(dynamic_cast<const bitslice_op *>(&o0->node()->GetOperation()));
    assert(dynamic_cast<const bitslice_op *>(&o1->node()->GetOperation()));

    const bitslice_op * attrs;
    attrs = dynamic_cast<const bitslice_op *>(&o0->node()->GetOperation());
    assert((attrs->low() == 8) && (attrs->high() == 16));
    attrs = dynamic_cast<const bitslice_op *>(&o1->node()->GetOperation());
    assert((attrs->low() == 0) && (attrs->high() == 8));

    assert(o0->node()->input(0)->origin() == x);
    assert(o1->node()->input(0)->origin() == y);
  }

  {
    auto slice1 = jlm::rvsdg::bitslice(x, 0, 8);
    auto slice2 = jlm::rvsdg::bitslice(x, 8, 16);
    auto concat = jlm::rvsdg::bitconcat({ slice1, slice2 });
    assert(concat == x);
  }

  return 0;
}

static int
SliceOfConstant()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  const Graph graph;
  const auto constant = create_bitconstant(&graph.GetRootRegion(), "00110111");
  const auto slice = bitslice(constant, 2, 6);
  auto & ex = jlm::tests::GraphExport::Create(*slice, "dummy");

  view(graph, stdout);

  // Assert
  const auto node = output::GetNode(*ex.origin());
  auto & operation = dynamic_cast<const bitconstant_op &>(node->GetOperation());
  assert(operation.value() == bitvalue_repr("1101"));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfConstant", SliceOfConstant);

static int
SliceOfSlice()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;
  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  auto slice1 = bitslice(x, 2, 6);
  auto slice2 = bitslice(slice1, 1, 3);

  auto & ex = jlm::tests::GraphExport::Create(*slice2, "dummy");
  view(graph, stdout);

  // Assert
  const auto node = output::GetNode(*ex.origin());
  const auto operation = dynamic_cast<const bitslice_op *>(&node->GetOperation());
  assert(operation->low() == 3 && operation->high() == 5);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfSlice", SliceOfSlice);

static int
SliceOfFullNode()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;
  const auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  auto sliceResult = bitslice(x, 0, 8);

  auto & ex = jlm::tests::GraphExport::Create(*sliceResult, "dummy");
  view(graph, stdout);

  // Assert
  assert(ex.origin() == x);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfFullNode", SliceOfFullNode);

static int
SliceOfConcat()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "y");

  auto concatResult = bitconcat({ x, y });
  auto sliceResult = bitslice(concatResult, 0, 8);

  auto & ex = jlm::tests::GraphExport::Create(*sliceResult, "dummy");
  view(graph, stdout);

  // Assert
  const auto bitType = dynamic_cast<const bittype *>(&ex.origin()->type());
  assert(bitType && bitType->nbits() == 8);
  assert(ex.origin() == x);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfConcat", SliceOfConcat);

static int
ConcatFlattening()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "y");
  auto z = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "z");

  auto concatResult1 = bitconcat({ x, y });
  auto concatResult2 = bitconcat({ concatResult1, z });

  auto & ex = jlm::tests::GraphExport::Create(*concatResult2, "dummy");
  view(graph, stdout);

  // Assert
  auto node = output::GetNode(*ex.origin());
  assert(dynamic_cast<const bitconcat_op *>(&node->GetOperation()));
  assert(node->ninputs() == 3);
  assert(node->input(0)->origin() == x);
  assert(node->input(1)->origin() == y);
  assert(node->input(2)->origin() == z);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatFlattening", ConcatFlattening);

static int
ConcatWithSingleOperand()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  const auto concatResult = bitconcat({ x });

  auto & ex = jlm::tests::GraphExport::Create(*concatResult, "dummy");
  view(graph, stdout);

  // Assert
  assert(ex.origin() == x);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/bitstring/bitstring-ConcatWithSingleOperand",
    ConcatWithSingleOperand);

static int
ConcatOfSlices()
{
  using namespace jlm::rvsdg;

  // Assert & Act
  Graph graph;

  const auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  auto sliceResult1 = bitslice(x, 0, 4);
  auto sliceResult2 = bitslice(x, 4, 8);
  const auto concatResult = bitconcat({ sliceResult1, sliceResult2 });

  auto & ex = jlm::tests::GraphExport::Create(*concatResult, "dummy");
  view(graph, stdout);

  // Assert
  assert(ex.origin() == x);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatWithOfSlices", ConcatOfSlices);

static int
ConcatOfConstants()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;

  auto c1 = create_bitconstant(&graph.GetRootRegion(), "00110111");
  auto c2 = create_bitconstant(&graph.GetRootRegion(), "11001000");

  auto concatResult = bitconcat({ c1, c2 });

  auto & ex = jlm::tests::GraphExport::Create(*concatResult, "dummy");
  view(graph, stdout);

  // Assert
  auto node = output::GetNode(*ex.origin());
  auto operation = dynamic_cast<const bitconstant_op &>(node->GetOperation());
  assert(operation.value() == bitvalue_repr("0011011111001000"));

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatOfConstants", ConcatOfConstants);

static int
ConcatCne()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "y");

  auto slice1 = bitslice(x, 2, 6);
  auto slice2 = bitslice(x, 2, 6);
  assert(slice1 == slice2);

  auto concat1 = bitconcat({ x, y });
  auto concat2 = bitconcat({ x, y });
  assert(concat1 == concat2);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatCne", ConcatCne);

static int
SliceCne()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  auto slice1 = bitslice(x, 2, 6);
  auto slice2 = bitslice(x, 2, 6);

  auto & ex1 = jlm::tests::GraphExport::Create(*slice1, "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*slice2, "dummy");
  view(graph, stdout);

  // Assert
  assert(ex1.origin() == ex2.origin());

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceCne", SliceCne);

static const char * bs[] = { "00000000", "11111111", "10000000", "01111111", "00001111",
                             "XXXX0011", "XD001100", "XXXXDDDD", "10XDDX01", "0DDDDDD1" };

static std::string bitstring_not[] = { "11111111", "00000000", "01111111", "10000000", "11110000",
                                       "XXXX1100", "XD110011", "XXXXDDDD", "01XDDX10", "1DDDDDD0" };

static std::string bitstring_xor[10][10] = { { "00000000",
                                               "11111111",
                                               "10000000",
                                               "01111111",
                                               "00001111",
                                               "XXXX0011",
                                               "XD001100",
                                               "XXXXDDDD",
                                               "10XDDX01",
                                               "0DDDDDD1" },
                                             { "11111111",
                                               "00000000",
                                               "01111111",
                                               "10000000",
                                               "11110000",
                                               "XXXX1100",
                                               "XD110011",
                                               "XXXXDDDD",
                                               "01XDDX10",
                                               "1DDDDDD0" },
                                             { "10000000",
                                               "01111111",
                                               "00000000",
                                               "11111111",
                                               "10001111",
                                               "XXXX0011",
                                               "XD001100",
                                               "XXXXDDDD",
                                               "00XDDX01",
                                               "1DDDDDD1" },
                                             { "01111111",
                                               "10000000",
                                               "11111111",
                                               "00000000",
                                               "01110000",
                                               "XXXX1100",
                                               "XD110011",
                                               "XXXXDDDD",
                                               "11XDDX10",
                                               "0DDDDDD0" },
                                             { "00001111",
                                               "11110000",
                                               "10001111",
                                               "01110000",
                                               "00000000",
                                               "XXXX1100",
                                               "XD000011",
                                               "XXXXDDDD",
                                               "10XDDX10",
                                               "0DDDDDD0" },
                                             { "XXXX0011",
                                               "XXXX1100",
                                               "XXXX0011",
                                               "XXXX1100",
                                               "XXXX1100",
                                               "XXXX0000",
                                               "XXXX1111",
                                               "XXXXDDDD",
                                               "XXXXDX10",
                                               "XXXXDDD0" },
                                             { "XD001100",
                                               "XD110011",
                                               "XD001100",
                                               "XD110011",
                                               "XD000011",
                                               "XXXX1111",
                                               "XD000000",
                                               "XXXXDDDD",
                                               "XDXDDX01",
                                               "XDDDDDD1" },
                                             { "XXXXDDDD",
                                               "XXXXDDDD",
                                               "XXXXDDDD",
                                               "XXXXDDDD",
                                               "XXXXDDDD",
                                               "XXXXDDDD",
                                               "XXXXDDDD",
                                               "XXXXDDDD",
                                               "XXXXDXDD",
                                               "XXXXDDDD" },
                                             { "10XDDX01",
                                               "01XDDX10",
                                               "00XDDX01",
                                               "11XDDX10",
                                               "10XDDX10",
                                               "XXXXDX10",
                                               "XDXDDX01",
                                               "XXXXDXDD",
                                               "00XDDX00",
                                               "1DXDDXD0" },
                                             { "0DDDDDD1",
                                               "1DDDDDD0",
                                               "1DDDDDD1",
                                               "0DDDDDD0",
                                               "0DDDDDD0",
                                               "XXXXDDD0",
                                               "XDDDDDD1",
                                               "XXXXDDDD",
                                               "1DXDDXD0",
                                               "0DDDDDD0" } };

static std::string bitstring_or[10][10] = { { "00000000",
                                              "11111111",
                                              "10000000",
                                              "01111111",
                                              "00001111",
                                              "XXXX0011",
                                              "XD001100",
                                              "XXXXDDDD",
                                              "10XDDX01",
                                              "0DDDDDD1" },
                                            { "11111111",
                                              "11111111",
                                              "11111111",
                                              "11111111",
                                              "11111111",
                                              "11111111",
                                              "11111111",
                                              "11111111",
                                              "11111111",
                                              "11111111" },
                                            { "10000000",
                                              "11111111",
                                              "10000000",
                                              "11111111",
                                              "10001111",
                                              "1XXX0011",
                                              "1D001100",
                                              "1XXXDDDD",
                                              "10XDDX01",
                                              "1DDDDDD1" },
                                            { "01111111",
                                              "11111111",
                                              "11111111",
                                              "01111111",
                                              "01111111",
                                              "X1111111",
                                              "X1111111",
                                              "X1111111",
                                              "11111111",
                                              "01111111" },
                                            { "00001111",
                                              "11111111",
                                              "10001111",
                                              "01111111",
                                              "00001111",
                                              "XXXX1111",
                                              "XD001111",
                                              "XXXX1111",
                                              "10XD1111",
                                              "0DDD1111" },
                                            { "XXXX0011",
                                              "11111111",
                                              "1XXX0011",
                                              "X1111111",
                                              "XXXX1111",
                                              "XXXX0011",
                                              "XXXX1111",
                                              "XXXXDD11",
                                              "1XXXDX11",
                                              "XXXXDD11" },
                                            { "XD001100",
                                              "11111111",
                                              "1D001100",
                                              "X1111111",
                                              "XD001111",
                                              "XXXX1111",
                                              "XD001100",
                                              "XXXX11DD",
                                              "1DXD1101",
                                              "XDDD11D1" },
                                            { "XXXXDDDD",
                                              "11111111",
                                              "1XXXDDDD",
                                              "X1111111",
                                              "XXXX1111",
                                              "XXXXDD11",
                                              "XXXX11DD",
                                              "XXXXDDDD",
                                              "1XXXDXD1",
                                              "XXXXDDD1" },
                                            { "10XDDX01",
                                              "11111111",
                                              "10XDDX01",
                                              "11111111",
                                              "10XD1111",
                                              "1XXXDX11",
                                              "1DXD1101",
                                              "1XXXDXD1",
                                              "10XDDX01",
                                              "1DXDDXD1" },
                                            { "0DDDDDD1",
                                              "11111111",
                                              "1DDDDDD1",
                                              "01111111",
                                              "0DDD1111",
                                              "XXXXDD11",
                                              "XDDD11D1",
                                              "XXXXDDD1",
                                              "1DXDDXD1",
                                              "0DDDDDD1" } };

static std::string bitstring_and[10][10] = { { "00000000",
                                               "00000000",
                                               "00000000",
                                               "00000000",
                                               "00000000",
                                               "00000000",
                                               "00000000",
                                               "00000000",
                                               "00000000",
                                               "00000000" },
                                             { "00000000",
                                               "11111111",
                                               "10000000",
                                               "01111111",
                                               "00001111",
                                               "XXXX0011",
                                               "XD001100",
                                               "XXXXDDDD",
                                               "10XDDX01",
                                               "0DDDDDD1" },
                                             { "00000000",
                                               "10000000",
                                               "10000000",
                                               "00000000",
                                               "00000000",
                                               "X0000000",
                                               "X0000000",
                                               "X0000000",
                                               "10000000",
                                               "00000000" },
                                             { "00000000",
                                               "01111111",
                                               "00000000",
                                               "01111111",
                                               "00001111",
                                               "0XXX0011",
                                               "0D001100",
                                               "0XXXDDDD",
                                               "00XDDX01",
                                               "0DDDDDD1" },
                                             { "00000000",
                                               "00001111",
                                               "00000000",
                                               "00001111",
                                               "00001111",
                                               "00000011",
                                               "00001100",
                                               "0000DDDD",
                                               "0000DX01",
                                               "0000DDD1" },
                                             { "00000000",
                                               "XXXX0011",
                                               "X0000000",
                                               "0XXX0011",
                                               "00000011",
                                               "XXXX0011",
                                               "XX000000",
                                               "XXXX00DD",
                                               "X0XX0001",
                                               "0XXX00D1" },
                                             { "00000000",
                                               "XD001100",
                                               "X0000000",
                                               "0D001100",
                                               "00001100",
                                               "XX000000",
                                               "XD001100",
                                               "XX00DD00",
                                               "X000DX00",
                                               "0D00DD00" },
                                             { "00000000",
                                               "XXXXDDDD",
                                               "X0000000",
                                               "0XXXDDDD",
                                               "0000DDDD",
                                               "XXXX00DD",
                                               "XX00DD00",
                                               "XXXXDDDD",
                                               "X0XXDX0D",
                                               "0XXXDDDD" },
                                             { "00000000",
                                               "10XDDX01",
                                               "10000000",
                                               "00XDDX01",
                                               "0000DX01",
                                               "X0XX0001",
                                               "X000DX00",
                                               "X0XXDX0D",
                                               "10XDDX01",
                                               "00XDDX01" },
                                             { "00000000",
                                               "0DDDDDD1",
                                               "00000000",
                                               "0DDDDDD1",
                                               "0000DDD1",
                                               "0XXX00D1",
                                               "0D00DD00",
                                               "0XXXDDDD",
                                               "00XDDX01",
                                               "0DDDDDD1" } };

static char equal[10][10] = { { '1', '0', '0', '0', '0', '0', '0', 'X', '0', '0' },
                              { '0', '1', '0', '0', '0', '0', '0', 'X', '0', '0' },
                              { '0', '0', '1', '0', '0', '0', '0', 'X', '0', '0' },
                              { '0', '0', '0', '1', '0', '0', '0', 'X', '0', 'D' },
                              { '0', '0', '0', '0', '1', '0', '0', 'X', '0', 'D' },
                              { '0', '0', '0', '0', '0', 'X', '0', 'X', '0', 'X' },
                              { '0', '0', '0', '0', '0', '0', 'X', 'X', '0', '0' },
                              { 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X' },
                              { '0', '0', '0', '0', '0', '0', '0', 'X', 'X', '0' },
                              { '0', '0', '0', 'D', 'D', 'X', '0', 'X', '0', 'D' } };

static char notequal[10][10] = { { '0', '1', '1', '1', '1', '1', '1', 'X', '1', '1' },
                                 { '1', '0', '1', '1', '1', '1', '1', 'X', '1', '1' },
                                 { '1', '1', '0', '1', '1', '1', '1', 'X', '1', '1' },
                                 { '1', '1', '1', '0', '1', '1', '1', 'X', '1', 'D' },
                                 { '1', '1', '1', '1', '0', '1', '1', 'X', '1', 'D' },
                                 { '1', '1', '1', '1', '1', 'X', '1', 'X', '1', 'X' },
                                 { '1', '1', '1', '1', '1', '1', 'X', 'X', '1', '1' },
                                 { 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X' },
                                 { '1', '1', '1', '1', '1', '1', '1', 'X', 'X', '1' },
                                 { '1', '1', '1', 'D', 'D', 'X', '1', 'X', '1', 'D' } };

static char sgreatereq[10][10] = { { '1', '1', '0', '1', '1', '1', '0', 'X', '1', '1' },
                                   { '0', '1', '0', '1', '1', '1', '0', 'D', '1', '1' },
                                   { '1', '1', '1', '1', '1', '1', '0', 'X', '1', '1' },
                                   { '0', '0', '0', '1', '1', '1', '0', 'X', '1', '1' },
                                   { '0', '0', '0', '0', '1', '1', '0', 'X', '1', 'D' },
                                   { '0', '0', '0', '0', '0', 'X', '0', 'X', '1', 'X' },
                                   { '1', '1', '1', '1', '1', '1', 'X', 'X', '1', '1' },
                                   { 'D', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X' },
                                   { '0', '0', '0', '0', '0', '0', '0', 'X', 'X', 'X' },
                                   { '0', '0', '0', 'D', 'D', 'X', '0', 'X', 'X', 'D' } };

static char sgreater[10][10] = { { '0', '1', '0', '1', '1', '1', '0', 'D', '1', '1' },
                                 { '0', '0', '0', '1', '1', '1', '0', 'X', '1', '1' },
                                 { '1', '1', '0', '1', '1', '1', '0', 'X', '1', '1' },
                                 { '0', '0', '0', '0', '1', '1', '0', 'X', '1', 'D' },
                                 { '0', '0', '0', '0', '0', '1', '0', 'D', '1', 'D' },
                                 { '0', '0', '0', '0', '0', 'X', '0', 'X', '1', 'X' },
                                 { '1', '1', '1', '1', '1', '1', 'X', 'X', '1', '1' },
                                 { 'X', 'D', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X' },
                                 { '0', '0', '0', '0', '0', '0', '0', 'X', 'X', 'X' },
                                 { '0', '0', '0', '0', 'D', 'X', '0', 'X', 'X', 'D' } };

static char slesseq[10][10] = { { '1', '0', '1', '0', '0', '0', '1', 'D', '0', '0' },
                                { '1', '1', '1', '0', '0', '0', '1', 'X', '0', '0' },
                                { '0', '0', '1', '0', '0', '0', '1', 'X', '0', '0' },
                                { '1', '1', '1', '1', '0', '0', '1', 'X', '0', 'D' },
                                { '1', '1', '1', '1', '1', '0', '1', 'D', '0', 'D' },
                                { '1', '1', '1', '1', '1', 'X', '1', 'X', '0', 'X' },
                                { '0', '0', '0', '0', '0', '0', 'X', 'X', '0', '0' },
                                { 'X', 'D', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X' },
                                { '1', '1', '1', '1', '1', '1', '1', 'X', 'X', 'X' },
                                { '1', '1', '1', '1', 'D', 'X', '1', 'X', 'X', 'D' } };

static char sless[10][10] = { { '0', '0', '1', '0', '0', '0', '1', 'X', '0', '0' },
                              { '1', '0', '1', '0', '0', '0', '1', 'D', '0', '0' },
                              { '0', '0', '0', '0', '0', '0', '1', 'X', '0', '0' },
                              { '1', '1', '1', '0', '0', '0', '1', 'X', '0', '0' },
                              { '1', '1', '1', '1', '0', '0', '1', 'X', '0', 'D' },
                              { '1', '1', '1', '1', '1', 'X', '1', 'X', '0', 'X' },
                              { '0', '0', '0', '0', '0', '0', 'X', 'X', '0', '0' },
                              { 'D', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X' },
                              { '1', '1', '1', '1', '1', '1', '1', 'X', 'X', 'X' },
                              { '1', '1', '1', 'D', 'D', 'X', '1', 'X', 'X', 'D' } };

static char ugreatereq[10][10] = { { '1', '0', '0', '0', '0', '0', '0', 'X', '0', '0' },
                                   { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                                   { '1', '0', '1', '0', '0', '0', '0', 'X', '0', '0' },
                                   { '1', '0', '1', '1', '1', '1', '1', 'X', '1', '1' },
                                   { '1', '0', '1', '0', '1', '1', '1', 'X', '1', 'D' },
                                   { '1', '0', '1', '0', '0', 'X', '1', 'X', '1', 'X' },
                                   { '1', '0', '1', '0', '0', '0', 'X', 'X', '0', '0' },
                                   { '1', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X' },
                                   { '1', '0', '1', '0', '0', '0', '1', 'X', 'X', 'X' },
                                   { '1', '0', '1', 'D', 'D', 'X', '1', 'X', 'X', 'D' } };

static char ugreater[10][10] = { { '0', '0', '0', '0', '0', '0', '0', '0', '0', '0' },
                                 { '1', '0', '1', '1', '1', '1', '1', 'X', '1', '1' },
                                 { '1', '0', '0', '0', '0', '0', '0', 'X', '0', '0' },
                                 { '1', '0', '1', '0', '1', '1', '1', 'X', '1', 'D' },
                                 { '1', '0', '1', '0', '0', '1', '1', 'D', '1', 'D' },
                                 { '1', '0', '1', '0', '0', 'X', '1', 'X', '1', 'X' },
                                 { '1', '0', '1', '0', '0', '0', 'X', 'X', '0', '0' },
                                 { 'X', '0', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X' },
                                 { '1', '0', '1', '0', '0', '0', '1', 'X', 'X', 'X' },
                                 { '1', '0', '1', '0', 'D', 'X', '1', 'X', 'X', 'D' } };

static char ulesseq[10][10] = { { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
                                { '0', '1', '0', '0', '0', '0', '0', 'X', '0', '0' },
                                { '0', '1', '1', '1', '1', '1', '1', 'X', '1', '1' },
                                { '0', '1', '0', '1', '0', '0', '0', 'X', '0', 'D' },
                                { '0', '1', '0', '1', '1', '0', '0', 'D', '0', 'D' },
                                { '0', '1', '0', '1', '1', 'X', '0', 'X', '0', 'X' },
                                { '0', '1', '0', '1', '1', '1', 'X', 'X', '1', '1' },
                                { 'X', '1', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X' },
                                { '0', '1', '0', '1', '1', '1', '0', 'X', 'X', 'X' },
                                { '0', '1', '0', '1', 'D', 'X', '0', 'X', 'X', 'D' } };

static char uless[10][10] = { { '0', '1', '1', '1', '1', '1', '1', 'X', '1', '1' },
                              { '0', '0', '0', '0', '0', '0', '0', '0', '0', '0' },
                              { '0', '1', '0', '1', '1', '1', '1', 'X', '1', '1' },
                              { '0', '1', '0', '0', '0', '0', '0', 'X', '0', '0' },
                              { '0', '1', '0', '1', '0', '0', '0', 'X', '0', 'D' },
                              { '0', '1', '0', '1', '1', 'X', '0', 'X', '0', 'X' },
                              { '0', '1', '0', '1', '1', '1', 'X', 'X', '1', '1' },
                              { '0', 'X', 'X', 'X', 'D', 'X', 'X', 'X', 'X', 'X' },
                              { '0', '1', '0', '1', '1', '1', '0', 'X', 'X', 'X' },
                              { '0', '1', '0', 'D', 'D', 'X', '0', 'X', 'X', 'D' } };

static int
types_bitstring_test_value_representation()
{
  using namespace jlm::rvsdg;

  for (size_t r = 0; r < 10; r++)
  {
    assert(bitvalue_repr(bs[r]).lnot() == bitstring_not[r]);
    for (size_t c = 0; c < 10; c++)
    {
      assert(bitvalue_repr(bs[r]).land(bs[c]) == bitstring_and[r][c]);
      assert(bitvalue_repr(bs[r]).lor(bs[c]) == bitstring_or[r][c]);
      assert(bitvalue_repr(bs[r]).lxor(bs[c]) == bitstring_xor[r][c]);

      assert(bitvalue_repr(bs[r]).ult(bs[c]) == uless[r][c]);
      assert(bitvalue_repr(bs[r]).slt(bs[c]) == sless[r][c]);

      assert(bitvalue_repr(bs[r]).ule(bs[c]) == ulesseq[r][c]);
      assert(bitvalue_repr(bs[r]).sle(bs[c]) == slesseq[r][c]);

      assert(bitvalue_repr(bs[r]).eq(bs[c]) == equal[r][c]);
      assert(bitvalue_repr(bs[r]).ne(bs[c]) == notequal[r][c]);

      assert(bitvalue_repr(bs[r]).uge(bs[c]) == ugreatereq[r][c]);
      assert(bitvalue_repr(bs[r]).sge(bs[c]) == sgreatereq[r][c]);

      assert(bitvalue_repr(bs[r]).ugt(bs[c]) == ugreater[r][c]);
      assert(bitvalue_repr(bs[r]).sgt(bs[c]) == sgreater[r][c]);
    }
  }

  assert(bitvalue_repr("000110").to_uint() == 24);
  assert(bitvalue_repr("00011").to_int() == -8);

  for (ssize_t r = -4; r < 5; r++)
  {
    bitvalue_repr rbits(32, r);

    assert(rbits.neg() == -r);
    assert(rbits.shl(1) == r << 1);
    assert(rbits.shl(32) == 0);
    assert(rbits.ashr(1) == r >> 1);
    assert(rbits.ashr(34) == (r < 0 ? -1 : 0));

    if (r >= 0)
    {
      assert(rbits.shr(1) == r >> 1);
      assert(rbits.shr(34) == 0);
    }

    for (ssize_t c = -4; c < 5; c++)
    {
      bitvalue_repr cbits(32, c);

      assert(rbits.add(cbits) == r + c);
      assert(rbits.sub(cbits) == r - c);
      assert(rbits.mul(cbits) == r * c);

      if (r >= 0 && c > 0)
      {
        assert(rbits.udiv(cbits) == r / c);
        assert(rbits.umod(cbits) == r % c);
      }

      if (c != 0)
      {
        assert(rbits.sdiv(cbits) == r / c);
        assert(rbits.smod(cbits) == r % c);
      }
    }
  }

  return 0;
}

static int
RunTests()
{
  types_bitstring_arithmetic_test_bitand();
  types_bitstring_arithmetic_test_bitashr();
  types_bitstring_arithmetic_test_bitdifference();
  types_bitstring_arithmetic_test_bitnegate();
  types_bitstring_arithmetic_test_bitnot();
  types_bitstring_arithmetic_test_bitor();
  types_bitstring_arithmetic_test_bitproduct();
  types_bitstring_arithmetic_test_bitshiproduct();
  types_bitstring_arithmetic_test_bitshl();
  types_bitstring_arithmetic_test_bitshr();
  types_bitstring_arithmetic_test_bitsmod();
  types_bitstring_arithmetic_test_bitsquotient();
  types_bitstring_arithmetic_test_bitsum();
  types_bitstring_arithmetic_test_bituhiproduct();
  types_bitstring_arithmetic_test_bitumod();
  types_bitstring_arithmetic_test_bituquotient();
  types_bitstring_arithmetic_test_bitxor();
  types_bitstring_comparison_test_bitequal();
  types_bitstring_comparison_test_bitnotequal();
  types_bitstring_comparison_test_bitsgreater();
  types_bitstring_comparison_test_bitsgreatereq();
  types_bitstring_comparison_test_bitsless();
  types_bitstring_comparison_test_bitslesseq();
  types_bitstring_comparison_test_bitugreater();
  types_bitstring_comparison_test_bitugreatereq();
  types_bitstring_comparison_test_bituless();
  types_bitstring_comparison_test_bitulesseq();
  types_bitstring_test_constant();
  types_bitstring_test_normalize();
  types_bitstring_test_reduction();
  types_bitstring_test_value_representation();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring", RunTests);
