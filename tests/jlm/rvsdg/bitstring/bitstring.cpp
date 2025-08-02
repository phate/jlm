/*
 * Copyright 2010 2011 2012 2013 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"
#include <test-operation.hpp>

#include <jlm/rvsdg/bitstring.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/view.hpp>

static int
types_bitstring_arithmetic_test_bitand()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  const auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  const auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  const auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & and0 = CreateOpNode<bitand_op>({ s0, s1 }, 32);
  auto & and1 = CreateOpNode<bitand_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*and0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*and1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitand_op>(NormalizeBinaryOperation, and0);
  ReduceNode<bitand_op>(NormalizeBinaryOperation, and1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitand_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, +1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitashr()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, -16);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto & ashr0 = CreateOpNode<bitashr_op>({ s0, s1 }, 32);
  auto & ashr1 = CreateOpNode<bitashr_op>({ c0, c2 }, 32);
  auto & ashr2 = CreateOpNode<bitashr_op>({ c0, c3 }, 32);
  auto & ashr3 = CreateOpNode<bitashr_op>({ c1, c2 }, 32);
  auto & ashr4 = CreateOpNode<bitashr_op>({ c1, c3 }, 32);

  auto & ex0 = GraphExport::Create(*ashr0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*ashr1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*ashr2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*ashr3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*ashr4.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitashr_op>(NormalizeBinaryOperation, ashr0);
  ReduceNode<bitashr_op>(NormalizeBinaryOperation, ashr1);
  ReduceNode<bitashr_op>(NormalizeBinaryOperation, ashr2);
  ReduceNode<bitashr_op>(NormalizeBinaryOperation, ashr3);
  ReduceNode<bitashr_op>(NormalizeBinaryOperation, ashr4);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitashr_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, 4));
  assert(TryGetOwnerNode<Node>(*ex2.origin())->GetOperation() == int_constant_op(32, 0));
  assert(TryGetOwnerNode<Node>(*ex3.origin())->GetOperation() == int_constant_op(32, -4));
  assert(TryGetOwnerNode<Node>(*ex4.origin())->GetOperation() == int_constant_op(32, -1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitdifference()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto & diff = CreateOpNode<bitsub_op>({ s0, s1 }, 32);

  auto & ex0 = GraphExport::Create(*diff.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsub_op>(NormalizeBinaryOperation, diff);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Act
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitsub_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitnegate()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & neg0 = CreateOpNode<bitneg_op>({ s0 }, 32);
  auto & neg1 = CreateOpNode<bitneg_op>({ c0 }, 32);
  auto & neg2 = CreateOpNode<bitneg_op>({ neg1.output(0) }, 32);

  auto & ex0 = GraphExport::Create(*neg0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*neg1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*neg2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, neg0);
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, neg1);
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, neg2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitneg_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, -3));
  assert(TryGetOwnerNode<Node>(*ex2.origin())->GetOperation() == int_constant_op(32, 3));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitnot()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & not0 = CreateOpNode<bitnot_op>({ s0 }, 32);
  auto & not1 = CreateOpNode<bitnot_op>({ c0 }, 32);
  auto & not2 = CreateOpNode<bitnot_op>({ not1.output(0) }, 32);

  auto & ex0 = GraphExport::Create(*not0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*not1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*not2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitnot_op>(NormalizeUnaryOperation, not0);
  ReduceNode<bitnot_op>(NormalizeUnaryOperation, not1);
  ReduceNode<bitnot_op>(NormalizeUnaryOperation, not2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitnot_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, -4));
  assert(TryGetOwnerNode<Node>(*ex2.origin())->GetOperation() == int_constant_op(32, 3));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitor()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & or0 = CreateOpNode<bitor_op>({ s0, s1 }, 32);
  auto & or1 = CreateOpNode<bitor_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*or0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*or1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitor_op>(NormalizeBinaryOperation, or0);
  ReduceNode<bitor_op>(NormalizeBinaryOperation, or1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitor_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == uint_constant_op(32, 7));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitproduct()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & product0 = CreateOpNode<bitmul_op>({ s0, s1 }, 32);
  auto & product1 = CreateOpNode<bitmul_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*product0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*product1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitmul_op>(NormalizeBinaryOperation, product0);
  ReduceNode<bitmul_op>(NormalizeBinaryOperation, product1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitmul_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == uint_constant_op(32, 15));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshiproduct()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto & shiproduct = CreateOpNode<bitsmulh_op>({ s0, s1 }, 32);

  auto & ex0 = GraphExport::Create(*shiproduct.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsmulh_op>(NormalizeBinaryOperation, shiproduct);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitsmulh_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshl()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto & shl0 = CreateOpNode<bitshl_op>({ s0, s1 }, 32);
  auto & shl1 = CreateOpNode<bitshl_op>({ c0, c1 }, 32);
  auto & shl2 = CreateOpNode<bitshl_op>({ c0, c2 }, 32);

  auto & ex0 = GraphExport::Create(*shl0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*shl1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*shl2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitshl_op>(NormalizeBinaryOperation, shl0);
  ReduceNode<bitshl_op>(NormalizeBinaryOperation, shl1);
  ReduceNode<bitshl_op>(NormalizeBinaryOperation, shl2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitshl_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == uint_constant_op(32, 64));
  assert(TryGetOwnerNode<Node>(*ex2.origin())->GetOperation() == uint_constant_op(32, 0));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshr()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto & shr0 = CreateOpNode<bitshr_op>({ s0, s1 }, 32);
  auto & shr1 = CreateOpNode<bitshr_op>({ c0, c1 }, 32);
  auto & shr2 = CreateOpNode<bitshr_op>({ c0, c2 }, 32);

  auto & ex0 = GraphExport::Create(*shr0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*shr1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*shr2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitshr_op>(NormalizeBinaryOperation, shr0);
  ReduceNode<bitshr_op>(NormalizeBinaryOperation, shr1);
  ReduceNode<bitshr_op>(NormalizeBinaryOperation, shr2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitshr_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == uint_constant_op(32, 4));
  assert(TryGetOwnerNode<Node>(*ex2.origin())->GetOperation() == uint_constant_op(32, 0));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsmod()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, -7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & smod0 = CreateOpNode<bitsmod_op>({ s0, s1 }, 32);
  auto & smod1 = CreateOpNode<bitsmod_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*smod0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*smod1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsmod_op>(NormalizeBinaryOperation, smod0);
  ReduceNode<bitsmod_op>(NormalizeBinaryOperation, smod1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitsmod_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, -1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsquotient()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, -3);

  auto & squot0 = CreateOpNode<bitsdiv_op>({ s0, s1 }, 32);
  auto & squot1 = CreateOpNode<bitsdiv_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*squot0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*squot1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsdiv_op>(NormalizeBinaryOperation, squot0);
  ReduceNode<bitsdiv_op>(NormalizeBinaryOperation, squot1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitsdiv_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, -2));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsum()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & sum0 = CreateOpNode<bitadd_op>({ s0, s1 }, 32);
  auto & sum1 = CreateOpNode<bitadd_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*sum0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*sum1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitadd_op>(NormalizeBinaryOperation, sum0);
  ReduceNode<bitadd_op>(NormalizeBinaryOperation, sum1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitadd_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, 8));

  return 0;
}

static int
types_bitstring_arithmetic_test_bituhiproduct()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto & uhiproduct = CreateOpNode<bitumulh_op>({ s0, s1 }, 32);

  auto & ex0 = GraphExport::Create(*uhiproduct.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitumulh_op>(NormalizeBinaryOperation, uhiproduct);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitumulh_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitumod()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & umod0 = CreateOpNode<bitumod_op>({ s0, s1 }, 32);
  auto & umod1 = CreateOpNode<bitumod_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*umod0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*umod1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitumod_op>(NormalizeBinaryOperation, umod0);
  ReduceNode<bitumod_op>(NormalizeBinaryOperation, umod1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitumod_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, 1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bituquotient()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & uquot0 = CreateOpNode<bitudiv_op>({ s0, s1 }, 32);
  auto & uquot1 = CreateOpNode<bitudiv_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*uquot0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*uquot1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitudiv_op>(NormalizeBinaryOperation, uquot0);
  ReduceNode<bitudiv_op>(NormalizeBinaryOperation, uquot1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitudiv_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, 2));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitxor()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & xor0 = CreateOpNode<bitxor_op>({ s0, s1 }, 32);
  auto & xor1 = CreateOpNode<bitxor_op>({ c0, c1 }, 32);

  auto & ex0 = GraphExport::Create(*xor0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*xor1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitxor_op>(NormalizeBinaryOperation, xor0);
  ReduceNode<bitxor_op>(NormalizeBinaryOperation, xor1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Arrange
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitxor_op(32));
  assert(TryGetOwnerNode<Node>(*ex1.origin())->GetOperation() == int_constant_op(32, 6));

  return 0;
}

static inline void
expect_static_true(jlm::rvsdg::Output * port)
{
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*port);
  auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&node->GetOperation());
  assert(op && op->value().nbits() == 1 && op->value().str() == "1");
}

static inline void
expect_static_false(jlm::rvsdg::Output * port)
{
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*port);
  auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op *>(&node->GetOperation());
  assert(op && op->value().nbits() == 1 && op->value().str() == "0");
}

static int
types_bitstring_comparison_test_bitequal()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant_undefined(&graph.GetRootRegion(), 32);

  auto & equal0 = CreateOpNode<biteq_op>({ s0, s1 }, 32);
  auto & equal1 = CreateOpNode<biteq_op>({ c0, c0 }, 32);
  auto & equal2 = CreateOpNode<biteq_op>({ c0, c1 }, 32);
  auto & equal3 = CreateOpNode<biteq_op>({ c0, c2 }, 32);

  auto & ex0 = GraphExport::Create(*equal0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*equal1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*equal2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*equal3.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal0);
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal1);
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal2);
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal3);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == biteq_op(32));
  expect_static_true(ex1.origin());
  expect_static_false(ex2.origin());
  assert(TryGetOwnerNode<Node>(*ex3.origin())->GetOperation() == biteq_op(32));

  return 0;
}

static int
types_bitstring_comparison_test_bitnotequal()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant_undefined(&graph.GetRootRegion(), 32);

  auto & nequal0 = CreateOpNode<bitne_op>({ s0, s1 }, 32);
  auto & nequal1 = CreateOpNode<bitne_op>({ c0, c0 }, 32);
  auto & nequal2 = CreateOpNode<bitne_op>({ c0, c1 }, 32);
  auto & nequal3 = CreateOpNode<bitne_op>({ c0, c2 }, 32);

  auto & ex0 = GraphExport::Create(*nequal0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*nequal1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*nequal2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*nequal3.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal0);
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal1);
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal2);
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal3);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitne_op(32));
  expect_static_false(ex1.origin());
  expect_static_true(ex2.origin());
  assert(TryGetOwnerNode<Node>(*ex3.origin())->GetOperation() == bitne_op(32));

  return 0;
}

static int
types_bitstring_comparison_test_bitsgreater()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto & sgreater0 = CreateOpNode<bitsgt_op>({ s0, s1 }, 32);
  auto & sgreater1 = CreateOpNode<bitsgt_op>({ c0, c1 }, 32);
  auto & sgreater2 = CreateOpNode<bitsgt_op>({ c1, c0 }, 32);
  auto & sgreater3 = CreateOpNode<bitsgt_op>({ s0, c2 }, 32);
  auto & sgreater4 = CreateOpNode<bitsgt_op>({ c3, s1 }, 32);

  auto & ex0 = GraphExport::Create(*sgreater0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*sgreater1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*sgreater2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*sgreater3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*sgreater4.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsgt_op>(NormalizeBinaryOperation, sgreater0);
  ReduceNode<bitsgt_op>(NormalizeBinaryOperation, sgreater1);
  ReduceNode<bitsgt_op>(NormalizeBinaryOperation, sgreater2);
  ReduceNode<bitsgt_op>(NormalizeBinaryOperation, sgreater3);
  ReduceNode<bitsgt_op>(NormalizeBinaryOperation, sgreater4);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitsgt_op(32));
  expect_static_false(ex1.origin());
  expect_static_true(ex2.origin());
  expect_static_false(ex3.origin());
  expect_static_false(ex4.origin());

  return 0;
}

static int
types_bitstring_comparison_test_bitsgreatereq()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto & sgreatereq0 = CreateOpNode<bitsge_op>({ s0, s1 }, 32);
  auto & sgreatereq1 = CreateOpNode<bitsge_op>({ c0, c1 }, 32);
  auto & sgreatereq2 = CreateOpNode<bitsge_op>({ c1, c0 }, 32);
  auto & sgreatereq3 = CreateOpNode<bitsge_op>({ c0, c0 }, 32);
  auto & sgreatereq4 = CreateOpNode<bitsge_op>({ c2, s0 }, 32);
  auto & sgreatereq5 = CreateOpNode<bitsge_op>({ s1, c3 }, 32);

  auto & ex0 = GraphExport::Create(*sgreatereq0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*sgreatereq1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*sgreatereq2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*sgreatereq3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*sgreatereq4.output(0), "dummy");
  auto & ex5 = GraphExport::Create(*sgreatereq5.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsge_op>(NormalizeBinaryOperation, sgreatereq0);
  ReduceNode<bitsge_op>(NormalizeBinaryOperation, sgreatereq1);
  ReduceNode<bitsge_op>(NormalizeBinaryOperation, sgreatereq2);
  ReduceNode<bitsge_op>(NormalizeBinaryOperation, sgreatereq3);
  ReduceNode<bitsge_op>(NormalizeBinaryOperation, sgreatereq4);
  ReduceNode<bitsge_op>(NormalizeBinaryOperation, sgreatereq5);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Arrange
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitsge_op(32));
  expect_static_false(ex1.origin());
  expect_static_true(ex2.origin());
  expect_static_true(ex3.origin());
  expect_static_true(ex4.origin());
  expect_static_true(ex5.origin());

  return 0;
}

static int
types_bitstring_comparison_test_bitsless()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto & sless0 = CreateOpNode<bitslt_op>({ s0, s1 }, 32);
  auto & sless1 = CreateOpNode<bitslt_op>({ c0, c1 }, 32);
  auto & sless2 = CreateOpNode<bitslt_op>({ c1, c0 }, 32);
  auto & sless3 = CreateOpNode<bitslt_op>({ c2, s0 }, 32);
  auto & sless4 = CreateOpNode<bitslt_op>({ s1, c3 }, 32);

  auto & ex0 = GraphExport::Create(*sless0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*sless1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*sless2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*sless3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*sless4.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitslt_op>(NormalizeBinaryOperation, sless0);
  ReduceNode<bitslt_op>(NormalizeBinaryOperation, sless1);
  ReduceNode<bitslt_op>(NormalizeBinaryOperation, sless2);
  ReduceNode<bitslt_op>(NormalizeBinaryOperation, sless3);
  ReduceNode<bitslt_op>(NormalizeBinaryOperation, sless4);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Arrange
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitslt_op(32));
  expect_static_true(ex1.origin());
  expect_static_false(ex2.origin());
  expect_static_false(ex3.origin());
  expect_static_false(ex4.origin());

  return 0;
}

static int
types_bitstring_comparison_test_bitslesseq()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 0x7fffffffL);
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, (-0x7fffffffL - 1));

  auto & slesseq0 = CreateOpNode<bitsle_op>({ s0, s1 }, 32);
  auto & slesseq1 = CreateOpNode<bitsle_op>({ c0, c1 }, 32);
  auto & slesseq2 = CreateOpNode<bitsle_op>({ c0, c0 }, 32);
  auto & slesseq3 = CreateOpNode<bitsle_op>({ c1, c0 }, 32);
  auto & slesseq4 = CreateOpNode<bitsle_op>({ s0, c2 }, 32);
  auto & slesseq5 = CreateOpNode<bitsle_op>({ c3, s1 }, 32);

  auto & ex0 = GraphExport::Create(*slesseq0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*slesseq1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*slesseq2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*slesseq3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*slesseq4.output(0), "dummy");
  auto & ex5 = GraphExport::Create(*slesseq5.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsle_op>(NormalizeBinaryOperation, slesseq0);
  ReduceNode<bitsle_op>(NormalizeBinaryOperation, slesseq1);
  ReduceNode<bitsle_op>(NormalizeBinaryOperation, slesseq2);
  ReduceNode<bitsle_op>(NormalizeBinaryOperation, slesseq3);
  ReduceNode<bitsle_op>(NormalizeBinaryOperation, slesseq4);
  ReduceNode<bitsle_op>(NormalizeBinaryOperation, slesseq5);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitsle_op(32));
  expect_static_true(ex1.origin());
  expect_static_true(ex2.origin());
  expect_static_false(ex3.origin());
  expect_static_true(ex4.origin());
  expect_static_true(ex5.origin());

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

  auto & ugreater0 = CreateOpNode<bitugt_op>({ s0, s1 }, 32);
  auto & ugreater1 = CreateOpNode<bitugt_op>({ c0, c1 }, 32);
  auto & ugreater2 = CreateOpNode<bitugt_op>({ c1, c0 }, 32);
  auto & ugreater3 = CreateOpNode<bitugt_op>({ s0, c2 }, 32);
  auto & ugreater4 = CreateOpNode<bitugt_op>({ c3, s1 }, 32);

  auto & ex0 = GraphExport::Create(*ugreater0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*ugreater1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*ugreater2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*ugreater3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*ugreater4.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Assert
  ReduceNode<bitugt_op>(NormalizeBinaryOperation, ugreater0);
  ReduceNode<bitugt_op>(NormalizeBinaryOperation, ugreater1);
  ReduceNode<bitugt_op>(NormalizeBinaryOperation, ugreater2);
  ReduceNode<bitugt_op>(NormalizeBinaryOperation, ugreater3);
  ReduceNode<bitugt_op>(NormalizeBinaryOperation, ugreater4);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitugt_op(32));
  expect_static_false(ex1.origin());
  expect_static_true(ex2.origin());
  expect_static_false(ex3.origin());
  expect_static_false(ex4.origin());

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

  auto & ugreatereq0 = CreateOpNode<bituge_op>({ s0, s1 }, 32);
  auto & ugreatereq1 = CreateOpNode<bituge_op>({ c0, c1 }, 32);
  auto & ugreatereq2 = CreateOpNode<bituge_op>({ c1, c0 }, 32);
  auto & ugreatereq3 = CreateOpNode<bituge_op>({ c0, c0 }, 32);
  auto & ugreatereq4 = CreateOpNode<bituge_op>({ c2, s0 }, 32);
  auto & ugreatereq5 = CreateOpNode<bituge_op>({ s1, c3 }, 32);

  auto & ex0 = GraphExport::Create(*ugreatereq0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*ugreatereq1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*ugreatereq2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*ugreatereq3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*ugreatereq4.output(0), "dummy");
  auto & ex5 = GraphExport::Create(*ugreatereq5.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bituge_op>(NormalizeBinaryOperation, ugreatereq0);
  ReduceNode<bituge_op>(NormalizeBinaryOperation, ugreatereq1);
  ReduceNode<bituge_op>(NormalizeBinaryOperation, ugreatereq2);
  ReduceNode<bituge_op>(NormalizeBinaryOperation, ugreatereq3);
  ReduceNode<bituge_op>(NormalizeBinaryOperation, ugreatereq4);
  ReduceNode<bituge_op>(NormalizeBinaryOperation, ugreatereq5);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bituge_op(32));
  expect_static_false(ex1.origin());
  expect_static_true(ex2.origin());
  expect_static_true(ex3.origin());
  expect_static_true(ex4.origin());
  expect_static_true(ex5.origin());

  return 0;
}

static int
types_bitstring_comparison_test_bituless()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, (0xffffffffUL));
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 0);

  auto & uless0 = CreateOpNode<bitult_op>({ s0, s1 }, 32);
  auto & uless1 = CreateOpNode<bitult_op>({ c0, c1 }, 32);
  auto & uless2 = CreateOpNode<bitult_op>({ c1, c0 }, 32);
  auto & uless3 = CreateOpNode<bitult_op>({ c2, s0 }, 32);
  auto & uless4 = CreateOpNode<bitult_op>({ s1, c3 }, 32);

  auto & ex0 = GraphExport::Create(*uless0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*uless1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*uless2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*uless3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*uless4.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitult_op>(NormalizeBinaryOperation, uless0);
  ReduceNode<bitult_op>(NormalizeBinaryOperation, uless1);
  ReduceNode<bitult_op>(NormalizeBinaryOperation, uless2);
  ReduceNode<bitult_op>(NormalizeBinaryOperation, uless3);
  ReduceNode<bitult_op>(NormalizeBinaryOperation, uless4);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitult_op(32));
  expect_static_true(ex1.origin());
  expect_static_false(ex2.origin());
  expect_static_false(ex3.origin());
  expect_static_false(ex4.origin());

  return 0;
}

static int
types_bitstring_comparison_test_bitulesseq()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, (0xffffffffUL));
  auto c3 = create_bitconstant(&graph.GetRootRegion(), 32, 0);

  auto & ulesseq0 = CreateOpNode<bitule_op>({ s0, s1 }, 32);
  auto & ulesseq1 = CreateOpNode<bitule_op>({ c0, c1 }, 32);
  auto & ulesseq2 = CreateOpNode<bitule_op>({ c0, c0 }, 32);
  auto & ulesseq3 = CreateOpNode<bitule_op>({ c1, c0 }, 32);
  auto & ulesseq4 = CreateOpNode<bitule_op>({ s0, c2 }, 32);
  auto & ulesseq5 = CreateOpNode<bitule_op>({ c3, s1 }, 32);

  auto & ex0 = GraphExport::Create(*ulesseq0.output(0), "dummy");
  auto & ex1 = GraphExport::Create(*ulesseq1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*ulesseq2.output(0), "dummy");
  auto & ex3 = GraphExport::Create(*ulesseq3.output(0), "dummy");
  auto & ex4 = GraphExport::Create(*ulesseq4.output(0), "dummy");
  auto & ex5 = GraphExport::Create(*ulesseq5.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitule_op>(NormalizeBinaryOperation, ulesseq0);
  ReduceNode<bitule_op>(NormalizeBinaryOperation, ulesseq1);
  ReduceNode<bitule_op>(NormalizeBinaryOperation, ulesseq2);
  ReduceNode<bitule_op>(NormalizeBinaryOperation, ulesseq3);
  ReduceNode<bitule_op>(NormalizeBinaryOperation, ulesseq4);
  ReduceNode<bitule_op>(NormalizeBinaryOperation, ulesseq5);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(TryGetOwnerNode<Node>(*ex0.origin())->GetOperation() == bitule_op(32));
  expect_static_true(ex1.origin());
  expect_static_true(ex2.origin());
  expect_static_false(ex3.origin());
  expect_static_true(ex4.origin());
  expect_static_true(ex5.origin());

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

  // Arrange
  Graph graph;

  auto NormalizeCne =
      [&](const SimpleOperation & operation, const std::vector<jlm::rvsdg::Output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  auto & b1 = CreateOpNode<bitconstant_op>(graph.GetRootRegion(), "00110011");
  auto & b2 = *TryGetOwnerNode<Node>(*create_bitconstant(&graph.GetRootRegion(), 8, 204));
  auto & b3 = *TryGetOwnerNode<Node>(*create_bitconstant(&graph.GetRootRegion(), 8, 204));
  auto & b4 = CreateOpNode<bitconstant_op>(graph.GetRootRegion(), "001100110");

  auto & ex1 = GraphExport::Create(*b1.output(0), "b1");
  auto & ex2 = GraphExport::Create(*b2.output(0), "b2");
  auto & ex3 = GraphExport::Create(*b3.output(0), "b3");
  auto & ex4 = GraphExport::Create(*b4.output(0), "b4");

  view(graph, stdout);

  // Act & Assert
  assert(b1.GetOperation() == uint_constant_op(8, 204));
  assert(b1.GetOperation() == int_constant_op(8, -52));

  ReduceNode<bitconstant_op>(NormalizeCne, *TryGetOwnerNode<Node>(*ex1.origin()));
  ReduceNode<bitconstant_op>(NormalizeCne, *TryGetOwnerNode<Node>(*ex2.origin()));
  ReduceNode<bitconstant_op>(NormalizeCne, *TryGetOwnerNode<Node>(*ex3.origin()));
  ReduceNode<bitconstant_op>(NormalizeCne, *TryGetOwnerNode<Node>(*ex4.origin()));

  assert(ex1.origin() == ex2.origin());
  assert(ex1.origin() == ex3.origin());

  const auto node1 = TryGetOwnerNode<Node>(*ex1.origin());
  assert(node1->GetOperation() == uint_constant_op(8, 204));
  assert(node1->GetOperation() == int_constant_op(8, -52));

  const auto node4 = TryGetOwnerNode<Node>(*ex4.origin());
  assert(node4->GetOperation() == uint_constant_op(9, 204));
  assert(node4->GetOperation() == int_constant_op(9, 204));

  const auto & plus_one_128 = CreateOpNode<bitconstant_op>(graph.GetRootRegion(), ONE_64 ZERO_64);
  assert(plus_one_128.GetOperation() == uint_constant_op(128, 1));
  assert(plus_one_128.GetOperation() == int_constant_op(128, 1));

  const auto & minus_one_128 = CreateOpNode<bitconstant_op>(graph.GetRootRegion(), MONE_64 MONE_64);
  assert(minus_one_128.GetOperation() == int_constant_op(128, -1));

  view(&graph.GetRootRegion(), stdout);

  return 0;
}

static int
types_bitstring_test_normalize()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  bittype bits32(32);
  auto imp = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "imp");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 4);

  auto & sum0 = CreateOpNode<bitadd_op>({ imp, c0 }, 32);
  auto & sum1 = CreateOpNode<bitadd_op>({ sum0.output(0), c1 }, 32);

  auto & ex = GraphExport::Create(*sum1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitadd_op>(FlattenAssociativeBinaryOperation, sum1);
  auto & flattenedBinaryNode = *TryGetOwnerNode<Node>(*ex.origin());
  ReduceNode<FlattenedBinaryOperation>(NormalizeFlattenedBinaryOperation, flattenedBinaryNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  auto node = TryGetOwnerNode<Node>(*ex.origin());
  assert(node->GetOperation() == bitadd_op(32));
  assert(node->ninputs() == 2);
  auto op1 = node->input(0)->origin();
  auto op2 = node->input(1)->origin();
  if (!is<node_output>(op1))
  {
    auto tmp = op1;
    op1 = op2;
    op2 = tmp;
  }
  /* FIXME: the graph traversers are currently broken, that is why it won't normalize */
  assert(TryGetOwnerNode<Node>(*op1)->GetOperation() == int_constant_op(32, 3 + 4));
  assert(op2 == imp);

  view(&graph.GetRootRegion(), stdout);

  return 0;
}

static void
assert_constant(jlm::rvsdg::Output * bitstr, size_t nbits, const char bits[])
{
  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*bitstr);
  auto op = dynamic_cast<const jlm::rvsdg::bitconstant_op &>(node->GetOperation());
  assert(op.value() == jlm::rvsdg::bitvalue_repr(std::string(bits, nbits).c_str()));
}

static void
types_bitstring_test_reduction()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit4Type = bittype::Create(4);
  std::vector types({ bit4Type, bit4Type });

  auto a = create_bitconstant(&graph.GetRootRegion(), "1100");
  auto b = create_bitconstant(&graph.GetRootRegion(), "1010");

  auto & bitAndNode = CreateOpNode<bitand_op>({ a, b }, 4);
  auto & bitOrNode = CreateOpNode<bitor_op>({ a, b }, 4);
  auto & bitXorNode = CreateOpNode<bitxor_op>({ a, b }, 4);
  auto & bitAddNode = CreateOpNode<bitadd_op>({ a, b }, 4);
  auto & bitMulNode = CreateOpNode<bitmul_op>({ a, b }, 4);
  auto & bitConcatNode = CreateOpNode<BitConcatOperation>({ a, b }, types);
  auto & bitNegNode1 = CreateOpNode<bitneg_op>({ a }, 4);
  auto & bitNegNode2 = CreateOpNode<bitneg_op>({ b }, 4);

  auto & exBitAnd = GraphExport::Create(*bitAndNode.output(0), "bitAnd");
  auto & exBitOr = GraphExport::Create(*bitOrNode.output(0), "bitOr");
  auto & exBitXor = GraphExport::Create(*bitXorNode.output(0), "bitXor");
  auto & exBitAdd = GraphExport::Create(*bitAddNode.output(0), "bitAdd");
  auto & exBitMul = GraphExport::Create(*bitMulNode.output(0), "bitMul");
  auto & exBitConcat = GraphExport::Create(*bitConcatNode.output(0), "bitConcat");
  auto & exBitNeg1 = GraphExport::Create(*bitNegNode1.output(0), "bitNeg1");
  auto & exBitNeg2 = GraphExport::Create(*bitNegNode2.output(0), "bitNeg2");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitand_op>(NormalizeBinaryOperation, bitAndNode);
  ReduceNode<bitor_op>(NormalizeBinaryOperation, bitOrNode);
  ReduceNode<bitxor_op>(NormalizeBinaryOperation, bitXorNode);
  ReduceNode<bitadd_op>(NormalizeBinaryOperation, bitAddNode);
  ReduceNode<bitmul_op>(NormalizeBinaryOperation, bitMulNode);
  ReduceNode<BitConcatOperation>(NormalizeBinaryOperation, bitConcatNode);
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, bitNegNode1);
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, bitNegNode2);

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert_constant(exBitAnd.origin(), 4, "1000");
  assert_constant(exBitOr.origin(), 4, "1110");
  assert_constant(exBitXor.origin(), 4, "0110");
  assert_constant(exBitAdd.origin(), 4, "0001");
  assert_constant(exBitMul.origin(), 4, "1111");
  assert_constant(exBitConcat.origin(), 8, "11001010");
  assert_constant(exBitNeg1.origin(), 4, "1011");
  assert_constant(exBitNeg2.origin(), 4, "1101");
}

static void
SliceOfConcatReduction()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit16Type = bittype::Create(16);
  auto bit32Type = bittype::Create(32);
  std::vector types({ bit16Type, bit16Type });

  auto x = &jlm::tests::GraphImport::Create(graph, bit16Type, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bit16Type, "y");

  auto & concatNode = CreateOpNode<BitConcatOperation>({ x, y }, types);
  auto & sliceNode = CreateOpNode<BitSliceOperation>({ concatNode.output(0) }, bit32Type, 8, 24);

  auto & ex = GraphExport::Create(*sliceNode.output(0), "bitAnd");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<BitSliceOperation>(NormalizeUnaryOperation, sliceNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  const auto node = TryGetOwnerNode<SimpleNode>(*ex.origin());
  const auto o0_node = TryGetOwnerNode<SimpleNode>(*node->input(0)->origin());
  const auto o1_node = TryGetOwnerNode<SimpleNode>(*node->input(1)->origin());
  assert(is<BitConcatOperation>(node->GetOperation()));
  assert(node->ninputs() == 2);
  assert(is<BitSliceOperation>(o0_node->GetOperation()));
  assert(is<BitSliceOperation>(o1_node->GetOperation()));

  auto attrs = dynamic_cast<const BitSliceOperation *>(&o0_node->GetOperation());
  assert((attrs->low() == 8) && (attrs->high() == 16));
  attrs = dynamic_cast<const BitSliceOperation *>(&o1_node->GetOperation());
  assert((attrs->low() == 0) && (attrs->high() == 8));

  assert(o0_node->input(0)->origin() == x);
  assert(o1_node->input(0)->origin() == y);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/bitstring/bitstring-SliceOfConcatReduction",
    SliceOfConcatReduction);

static void
ConcatOfSliceReduction()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit8Type = bittype::Create(8);
  auto bit16Type = bittype::Create(16);
  std::vector types({ bit8Type, bit8Type });

  auto x = &jlm::tests::GraphImport::Create(graph, bit16Type, "x");

  auto slice1 = bitslice(x, 0, 8);
  auto slice2 = bitslice(x, 8, 16);
  auto & concatNode = CreateOpNode<BitConcatOperation>({ slice1, slice2 }, types);

  auto & ex = GraphExport::Create(*concatNode.output(0), "bitAnd");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<BitConcatOperation>(NormalizeBinaryOperation, concatNode);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  const auto sliceNode = TryGetOwnerNode<Node>(*ex.origin());
  assert(sliceNode->GetOperation() == BitSliceOperation(bit16Type, 0, 16));
  assert(sliceNode->input(0)->origin() == x);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/bitstring/bitstring-ConcatOfSliceReduction",
    ConcatOfSliceReduction);

static void
SliceOfConstant()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit8Type = bittype::Create(8);

  const auto constant = create_bitconstant(&graph.GetRootRegion(), "00110111");
  auto & sliceNode = CreateOpNode<BitSliceOperation>({ constant }, bit8Type, 2, 6);
  auto & ex = GraphExport::Create(*sliceNode.output(0), "dummy");

  view(graph, stdout);

  // Act
  ReduceNode<BitSliceOperation>(NormalizeUnaryOperation, sliceNode);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  const auto node = TryGetOwnerNode<Node>(*ex.origin());
  auto & operation = dynamic_cast<const bitconstant_op &>(node->GetOperation());
  assert(operation.value() == bitvalue_repr("1101"));
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfConstant", SliceOfConstant);

static void
SliceOfSlice()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit4Type = bittype::Create(4);

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  auto slice1 = bitslice(x, 2, 6);
  auto & sliceNode2 = CreateOpNode<BitSliceOperation>({ slice1 }, bit4Type, 1, 3);

  auto & ex = GraphExport::Create(*sliceNode2.output(0), "dummy");
  view(graph, stdout);

  // Act
  ReduceNode<BitSliceOperation>(NormalizeUnaryOperation, sliceNode2);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  const auto node = TryGetOwnerNode<Node>(*ex.origin());
  const auto operation = dynamic_cast<const BitSliceOperation *>(&node->GetOperation());
  assert(operation->low() == 3 && operation->high() == 5);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfSlice", SliceOfSlice);

static void
SliceOfFullNode()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit8Type = bittype::Create(8);

  const auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  auto & sliceNode = CreateOpNode<BitSliceOperation>({ x }, bit8Type, 0, 8);

  auto & ex = GraphExport::Create(*sliceNode.output(0), "dummy");
  view(graph, stdout);

  // Act
  ReduceNode<BitSliceOperation>(NormalizeUnaryOperation, sliceNode);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(ex.origin() == x);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfFullNode", SliceOfFullNode);

static void
SliceOfConcat()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit16Type = bittype::Create(16);

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "y");

  auto concatResult = bitconcat({ x, y });
  auto & sliceNode = CreateOpNode<BitSliceOperation>({ concatResult }, bit16Type, 0, 8);

  auto & ex = GraphExport::Create(*sliceNode.output(0), "dummy");
  view(graph, stdout);

  // Act
  ReduceNode<BitSliceOperation>(NormalizeUnaryOperation, sliceNode);
  auto concatNode = TryGetOwnerNode<Node>(*ex.origin());
  ReduceNode<BitSliceOperation>(
      NormalizeUnaryOperation,
      *TryGetOwnerNode<Node>(*concatNode->input(0)->origin()));
  concatNode = TryGetOwnerNode<Node>(*ex.origin());
  ReduceNode<BitConcatOperation>(NormalizeBinaryOperation, *concatNode);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  const auto bitType = std::dynamic_pointer_cast<const bittype>(ex.origin()->Type());
  assert(bitType && bitType->nbits() == 8);
  assert(ex.origin() == x);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-SliceOfConcat", SliceOfConcat);

static void
ConcatFlattening()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "y");
  auto z = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "z");

  auto concatResult1 = bitconcat({ x, y });
  auto concatResult2 = bitconcat({ concatResult1, z });

  auto & ex = GraphExport::Create(*concatResult2, "dummy");
  view(graph, stdout);

  // Act
  const auto concatNode = TryGetOwnerNode<Node>(*ex.origin());
  ReduceNode<BitConcatOperation>(FlattenBitConcatOperation, *concatNode);

  view(graph, stdout);

  // Assert
  auto node = TryGetOwnerNode<Node>(*ex.origin());
  assert(is<BitConcatOperation>(node->GetOperation()));
  assert(node->ninputs() == 3);
  assert(node->input(0)->origin() == x);
  assert(node->input(1)->origin() == y);
  assert(node->input(2)->origin() == z);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatFlattening", ConcatFlattening);

static void
ConcatWithSingleOperand()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto bit8Type = bittype::Create(8);
  std::vector bit8Types({ bit8Type });

  auto x = &jlm::tests::GraphImport::Create(graph, bit8Type, "x");

  auto & concatNode = CreateOpNode<BitConcatOperation>({ x }, bit8Types);

  auto & ex = GraphExport::Create(*concatNode.output(0), "dummy");
  view(graph, stdout);

  // Act
  ReduceNode<BitConcatOperation>(NormalizeBinaryOperation, concatNode);
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(ex.origin() == x);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/rvsdg/bitstring/bitstring-ConcatWithSingleOperand",
    ConcatWithSingleOperand);

static void
ConcatOfSlices()
{
  using namespace jlm::rvsdg;

  // Assert
  Graph graph;
  auto bit4Type = bittype::Create(4);
  std::vector bit4Types({ bit4Type, bit4Type });

  const auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");

  auto sliceResult1 = bitslice(x, 0, 4);
  auto sliceResult2 = bitslice(x, 4, 8);
  auto & concatNode = CreateOpNode<BitConcatOperation>({ sliceResult1, sliceResult2 }, bit4Types);

  auto & ex = GraphExport::Create(*concatNode.output(0), "dummy");
  view(graph, stdout);

  // Act
  ReduceNode<BitConcatOperation>(NormalizeBinaryOperation, concatNode);
  ReduceNode<BitSliceOperation>(NormalizeUnaryOperation, *TryGetOwnerNode<Node>(*ex.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(ex.origin() == x);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatWithOfSlices", ConcatOfSlices);

static void
ConcatOfConstants()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto c1 = create_bitconstant(&graph.GetRootRegion(), "00110111");
  auto c2 = create_bitconstant(&graph.GetRootRegion(), "11001000");

  auto concatResult = bitconcat({ c1, c2 });

  auto & ex = GraphExport::Create(*concatResult, "dummy");
  view(graph, stdout);

  // Act
  ReduceNode<BitConcatOperation>(NormalizeBinaryOperation, *TryGetOwnerNode<Node>(*ex.origin()));

  // Assert
  auto node = TryGetOwnerNode<Node>(*ex.origin());
  auto operation = dynamic_cast<const bitconstant_op &>(node->GetOperation());
  assert(operation.value() == bitvalue_repr("0011011111001000"));
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatOfConstants", ConcatOfConstants);

static void
ConcatCne()
{
  using namespace jlm::rvsdg;

  // Arrange & Act
  Graph graph;
  auto NormalizeCne =
      [&](const SimpleOperation & operation, const std::vector<jlm::rvsdg::Output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  auto bitType8 = bittype::Create(8);
  std::vector bitTypes({ bitType8, bitType8 });

  auto x = &jlm::tests::GraphImport::Create(graph, bitType8, "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bitType8, "y");

  auto & concatNode1 = CreateOpNode<BitConcatOperation>({ x, y }, bitTypes);
  auto & concatNode2 = CreateOpNode<BitConcatOperation>({ x, y }, bitTypes);

  auto & ex1 = GraphExport::Create(*concatNode1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*concatNode2.output(0), "dummy");

  view(graph, stdout);

  // Act
  ReduceNode<BitConcatOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*ex1.origin()));
  ReduceNode<BitConcatOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*ex2.origin()));
  graph.PruneNodes();

  view(graph, stdout);

  // Assert
  assert(ex1.origin() == ex2.origin());
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring-ConcatCne", ConcatCne);

static void
SliceCne()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto NormalizeCne =
      [&](const SimpleOperation & operation, const std::vector<jlm::rvsdg::Output *> & operands)
  {
    return NormalizeSimpleOperationCommonNodeElimination(
        graph.GetRootRegion(),
        operation,
        operands);
  };

  auto bitType8 = bittype::Create(8);

  auto x = &jlm::tests::GraphImport::Create(graph, bitType8, "x");

  auto & sliceNode1 = CreateOpNode<BitSliceOperation>({ x }, bitType8, 2, 6);
  auto & sliceNode2 = CreateOpNode<BitSliceOperation>({ x }, bitType8, 2, 6);

  auto & ex1 = GraphExport::Create(*sliceNode1.output(0), "dummy");
  auto & ex2 = GraphExport::Create(*sliceNode2.output(0), "dummy");

  view(graph, stdout);

  // Act
  ReduceNode<BitSliceOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*ex1.origin()));
  ReduceNode<BitSliceOperation>(NormalizeCne, *TryGetOwnerNode<Node>(*ex2.origin()));
  graph.PruneNodes();
  view(graph, stdout);

  // Assert
  assert(ex1.origin() == ex2.origin());
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

static void
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
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/bitstring/bitstring", RunTests);
