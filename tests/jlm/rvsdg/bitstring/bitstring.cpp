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
  const auto nf = graph.GetNodeNormalForm(typeid(bitand_op));
  nf->set_mutable(false);

  const auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  const auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  const auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  const auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & and0 = CreateOpNode<bitand_op>({ s0, s1 }, 32);
  auto & and1 = CreateOpNode<bitand_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*and0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*and1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitand_op>(NormalizeBinaryOperation, and0);
  ReduceNode<bitand_op>(NormalizeBinaryOperation, and1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitand_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, +1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitashr()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitashr_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*ashr0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*ashr1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*ashr2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*ashr3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*ashr4.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitashr_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, 4));
  assert(output::GetNode(*ex2.origin())->GetOperation() == int_constant_op(32, 0));
  assert(output::GetNode(*ex3.origin())->GetOperation() == int_constant_op(32, -4));
  assert(output::GetNode(*ex4.origin())->GetOperation() == int_constant_op(32, -1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitdifference()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitsub_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto & diff = CreateOpNode<bitsub_op>({ s0, s1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*diff.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsub_op>(NormalizeBinaryOperation, diff);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Act
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitsub_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitnegate()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitneg_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & neg0 = CreateOpNode<bitneg_op>({ s0 }, 32);
  auto & neg1 = CreateOpNode<bitneg_op>({ c0 }, 32);
  auto & neg2 = CreateOpNode<bitneg_op>({ neg1.output(0) }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*neg0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*neg1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*neg2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, neg0);
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, neg1);
  ReduceNode<bitneg_op>(NormalizeUnaryOperation, neg2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitneg_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, -3));
  assert(output::GetNode(*ex2.origin())->GetOperation() == int_constant_op(32, 3));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitnot()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitnot_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & not0 = CreateOpNode<bitnot_op>({ s0 }, 32);
  auto & not1 = CreateOpNode<bitnot_op>({ c0 }, 32);
  auto & not2 = CreateOpNode<bitnot_op>({ not1.output(0) }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*not0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*not1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*not2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitnot_op>(NormalizeUnaryOperation, not0);
  ReduceNode<bitnot_op>(NormalizeUnaryOperation, not1);
  ReduceNode<bitnot_op>(NormalizeUnaryOperation, not2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitnot_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, -4));
  assert(output::GetNode(*ex2.origin())->GetOperation() == int_constant_op(32, 3));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitor()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitor_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & or0 = CreateOpNode<bitor_op>({ s0, s1 }, 32);
  auto & or1 = CreateOpNode<bitor_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*or0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*or1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitor_op>(NormalizeBinaryOperation, or0);
  ReduceNode<bitor_op>(NormalizeBinaryOperation, or1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitor_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == uint_constant_op(32, 7));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitproduct()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitashr_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & product0 = CreateOpNode<bitmul_op>({ s0, s1 }, 32);
  auto & product1 = CreateOpNode<bitmul_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*product0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*product1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitmul_op>(NormalizeBinaryOperation, product0);
  ReduceNode<bitmul_op>(NormalizeBinaryOperation, product1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitmul_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == uint_constant_op(32, 15));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshiproduct()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitsmulh_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto & shiproduct = CreateOpNode<bitsmulh_op>({ s0, s1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*shiproduct.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsmulh_op>(NormalizeBinaryOperation, shiproduct);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitsmulh_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshl()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitshl_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto & shl0 = CreateOpNode<bitshl_op>({ s0, s1 }, 32);
  auto & shl1 = CreateOpNode<bitshl_op>({ c0, c1 }, 32);
  auto & shl2 = CreateOpNode<bitshl_op>({ c0, c2 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*shl0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*shl1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*shl2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitshl_op>(NormalizeBinaryOperation, shl0);
  ReduceNode<bitshl_op>(NormalizeBinaryOperation, shl1);
  ReduceNode<bitshl_op>(NormalizeBinaryOperation, shl2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitshl_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == uint_constant_op(32, 64));
  assert(output::GetNode(*ex2.origin())->GetOperation() == uint_constant_op(32, 0));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitshr()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitshr_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 16);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 2);
  auto c2 = create_bitconstant(&graph.GetRootRegion(), 32, 32);

  auto & shr0 = CreateOpNode<bitshr_op>({ s0, s1 }, 32);
  auto & shr1 = CreateOpNode<bitshr_op>({ c0, c1 }, 32);
  auto & shr2 = CreateOpNode<bitshr_op>({ c0, c2 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*shr0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*shr1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*shr2.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitshr_op>(NormalizeBinaryOperation, shr0);
  ReduceNode<bitshr_op>(NormalizeBinaryOperation, shr1);
  ReduceNode<bitshr_op>(NormalizeBinaryOperation, shr2);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitshr_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == uint_constant_op(32, 4));
  assert(output::GetNode(*ex2.origin())->GetOperation() == uint_constant_op(32, 0));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsmod()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitsmod_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, -7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & smod0 = CreateOpNode<bitsmod_op>({ s0, s1 }, 32);
  auto & smod1 = CreateOpNode<bitsmod_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*smod0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*smod1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsmod_op>(NormalizeBinaryOperation, smod0);
  ReduceNode<bitsmod_op>(NormalizeBinaryOperation, smod1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitsmod_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, -1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsquotient()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitsdiv_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, -3);

  auto & squot0 = CreateOpNode<bitsdiv_op>({ s0, s1 }, 32);
  auto & squot1 = CreateOpNode<bitsdiv_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*squot0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*squot1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitsdiv_op>(NormalizeBinaryOperation, squot0);
  ReduceNode<bitsdiv_op>(NormalizeBinaryOperation, squot1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitsdiv_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, -2));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitsum()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitadd_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & sum0 = CreateOpNode<bitadd_op>({ s0, s1 }, 32);
  auto & sum1 = CreateOpNode<bitadd_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*sum0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*sum1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitadd_op>(NormalizeBinaryOperation, sum0);
  ReduceNode<bitadd_op>(NormalizeBinaryOperation, sum1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitadd_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, 8));

  return 0;
}

static int
types_bitstring_arithmetic_test_bituhiproduct()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitumulh_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto & uhiproduct = CreateOpNode<bitumulh_op>({ s0, s1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*uhiproduct.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitumulh_op>(NormalizeBinaryOperation, uhiproduct);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitumulh_op(32));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitumod()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitumod_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & umod0 = CreateOpNode<bitumod_op>({ s0, s1 }, 32);
  auto & umod1 = CreateOpNode<bitumod_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*umod0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*umod1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitumod_op>(NormalizeBinaryOperation, umod0);
  ReduceNode<bitumod_op>(NormalizeBinaryOperation, umod1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitumod_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, 1));

  return 0;
}

static int
types_bitstring_arithmetic_test_bituquotient()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitudiv_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 7);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 3);

  auto & uquot0 = CreateOpNode<bitudiv_op>({ s0, s1 }, 32);
  auto & uquot1 = CreateOpNode<bitudiv_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*uquot0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*uquot1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitudiv_op>(NormalizeBinaryOperation, uquot0);
  ReduceNode<bitudiv_op>(NormalizeBinaryOperation, uquot1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitudiv_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, 2));

  return 0;
}

static int
types_bitstring_arithmetic_test_bitxor()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitxor_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");

  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 3);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);

  auto & xor0 = CreateOpNode<bitxor_op>({ s0, s1 }, 32);
  auto & xor1 = CreateOpNode<bitxor_op>({ c0, c1 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*xor0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*xor1.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitxor_op>(NormalizeBinaryOperation, xor0);
  ReduceNode<bitxor_op>(NormalizeBinaryOperation, xor1);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Arrange
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitxor_op(32));
  assert(output::GetNode(*ex1.origin())->GetOperation() == int_constant_op(32, 6));

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

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(biteq_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant_undefined(&graph.GetRootRegion(), 32);

  auto & equal0 = CreateOpNode<biteq_op>({ s0, s1 }, 32);
  auto & equal1 = CreateOpNode<biteq_op>({ c0, c0 }, 32);
  auto & equal2 = CreateOpNode<biteq_op>({ c0, c1 }, 32);
  auto & equal3 = CreateOpNode<biteq_op>({ c0, c2 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*equal0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*equal1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*equal2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*equal3.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal0);
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal1);
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal2);
  ReduceNode<biteq_op>(NormalizeBinaryOperation, equal3);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == biteq_op(32));
  expect_static_true(ex1.origin());
  expect_static_false(ex2.origin());
  assert(output::GetNode(*ex3.origin())->GetOperation() == biteq_op(32));

  return 0;
}

static int
types_bitstring_comparison_test_bitnotequal()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitne_op));
  nf->set_mutable(false);

  auto s0 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s0");
  auto s1 = &jlm::tests::GraphImport::Create(graph, bittype::Create(32), "s1");
  auto c0 = create_bitconstant(&graph.GetRootRegion(), 32, 4);
  auto c1 = create_bitconstant(&graph.GetRootRegion(), 32, 5);
  auto c2 = create_bitconstant_undefined(&graph.GetRootRegion(), 32);

  auto & nequal0 = CreateOpNode<bitne_op>({ s0, s1 }, 32);
  auto & nequal1 = CreateOpNode<bitne_op>({ c0, c0 }, 32);
  auto & nequal2 = CreateOpNode<bitne_op>({ c0, c1 }, 32);
  auto & nequal3 = CreateOpNode<bitne_op>({ c0, c2 }, 32);

  auto & ex0 = jlm::tests::GraphExport::Create(*nequal0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*nequal1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*nequal2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*nequal3.output(0), "dummy");

  view(&graph.GetRootRegion(), stdout);

  // Act
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal0);
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal1);
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal2);
  ReduceNode<bitne_op>(NormalizeBinaryOperation, nequal3);
  graph.PruneNodes();

  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitne_op(32));
  expect_static_false(ex1.origin());
  expect_static_true(ex2.origin());
  assert(output::GetNode(*ex3.origin())->GetOperation() == bitne_op(32));

  return 0;
}

static int
types_bitstring_comparison_test_bitsgreater()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  auto nf = graph.GetNodeNormalForm(typeid(bitsgt_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*sgreater0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*sgreater1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*sgreater2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*sgreater3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*sgreater4.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitsgt_op(32));
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
  auto nf = graph.GetNodeNormalForm(typeid(bitsge_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*sgreatereq0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*sgreatereq1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*sgreatereq2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*sgreatereq3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*sgreatereq4.output(0), "dummy");
  auto & ex5 = jlm::tests::GraphExport::Create(*sgreatereq5.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitsge_op(32));
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
  auto nf = graph.GetNodeNormalForm(typeid(bitslt_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*sless0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*sless1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*sless2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*sless3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*sless4.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitslt_op(32));
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
  auto nf = graph.GetNodeNormalForm(typeid(bitsle_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*slesseq0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*slesseq1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*slesseq2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*slesseq3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*slesseq4.output(0), "dummy");
  auto & ex5 = jlm::tests::GraphExport::Create(*slesseq5.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitsle_op(32));
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
  auto nf = graph.GetNodeNormalForm(typeid(bitugt_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*ugreater0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*ugreater1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*ugreater2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*ugreater3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*ugreater4.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitugt_op(32));
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
  auto nf = graph.GetNodeNormalForm(typeid(bituge_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*ugreatereq0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*ugreatereq1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*ugreatereq2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*ugreatereq3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*ugreatereq4.output(0), "dummy");
  auto & ex5 = jlm::tests::GraphExport::Create(*ugreatereq5.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bituge_op(32));
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
  auto nf = graph.GetNodeNormalForm(typeid(bitult_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*uless0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*uless1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*uless2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*uless3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*uless4.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitult_op(32));
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
  auto nf = graph.GetNodeNormalForm(typeid(bitule_op));
  nf->set_mutable(false);

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

  auto & ex0 = jlm::tests::GraphExport::Create(*ulesseq0.output(0), "dummy");
  auto & ex1 = jlm::tests::GraphExport::Create(*ulesseq1.output(0), "dummy");
  auto & ex2 = jlm::tests::GraphExport::Create(*ulesseq2.output(0), "dummy");
  auto & ex3 = jlm::tests::GraphExport::Create(*ulesseq3.output(0), "dummy");
  auto & ex4 = jlm::tests::GraphExport::Create(*ulesseq4.output(0), "dummy");
  auto & ex5 = jlm::tests::GraphExport::Create(*ulesseq5.output(0), "dummy");

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
  assert(output::GetNode(*ex0.origin())->GetOperation() == bitule_op(32));
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

  // Arrange
  Graph graph;
  const auto nf = graph.GetNodeNormalForm(typeid(bitconcat_op));
  nf->set_mutable(false);

  auto x = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "x");
  auto y = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "y");
  auto z = &jlm::tests::GraphImport::Create(graph, bittype::Create(8), "z");

  auto concatResult1 = bitconcat({ x, y });
  auto concatResult2 = bitconcat({ concatResult1, z });

  auto & ex = jlm::tests::GraphExport::Create(*concatResult2, "dummy");
  view(graph, stdout);

  // Act
  const auto concatNode = output::GetNode(*ex.origin());
  ReduceNode<bitconcat_op>(FlattenBitConcatOperation, *concatNode);

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
