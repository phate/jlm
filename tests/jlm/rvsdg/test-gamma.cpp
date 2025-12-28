/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestType.hpp>
#include <jlm/rvsdg/view.hpp>

static void
test_gamma()
{
  using namespace jlm::rvsdg;

  Graph graph;
  auto cmp = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(2), "");
  auto v0 = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(32), "");
  auto v1 = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(32), "");
  auto v2 = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(32), "");
  auto v3 = &jlm::rvsdg::GraphImport::Create(graph, ControlType::Create(2), "");

  auto pred = match(2, { { 0, 0 }, { 1, 1 } }, 2, 3, cmp);

  auto gamma = GammaNode::create(pred, 3);
  auto ev0 = gamma->AddEntryVar(v0);
  auto ev1 = gamma->AddEntryVar(v1);
  auto ev2 = gamma->AddEntryVar(v2);
  gamma->AddExitVar({ ev0.branchArgument[0], ev1.branchArgument[1], ev2.branchArgument[2] });

  GraphExport::Create(*gamma->output(0), "dummy");

  assert(gamma && gamma->GetOperation() == GammaOperation(3));

  /* test gamma copy */

  auto gamma2 =
      static_cast<StructuralNode *>(gamma)->copy(&graph.GetRootRegion(), { pred, v0, v1, v2 });
  view(&graph.GetRootRegion(), stdout);
  assert(dynamic_cast<const GammaNode *>(gamma2));

  /* test entry and exit variable iterators */

  auto gamma3 = GammaNode::create(v3, 2);
  assert(gamma3->GetEntryVars().empty());
  assert(gamma3->GetExitVars().empty());
}

static void
test_predicate_reduction()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  BitType bits2(2);

  auto v0 = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(32), "");
  auto v1 = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(32), "");
  auto v2 = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(32), "");

  auto pred = &ControlConstantOperation::create(graph.GetRootRegion(), 3, 1);

  auto gamma = GammaNode::create(pred, 3);
  auto ev0 = gamma->AddEntryVar(v0);
  auto ev1 = gamma->AddEntryVar(v1);
  auto ev2 = gamma->AddEntryVar(v2);
  gamma->AddExitVar({ ev0.branchArgument[0], ev1.branchArgument[1], ev2.branchArgument[2] });

  auto & r = GraphExport::Create(*gamma->output(0), "");

  view(&graph.GetRootRegion(), stdout);

  // Act
  auto gammaNode = TryGetOwnerNode<GammaNode>(*r.origin());
  ReduceGammaWithStaticallyKnownPredicate(*gammaNode);
  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(r.origin() == v1);

  graph.PruneNodes();
  assert(graph.GetRootRegion().numNodes() == 0);
}

static void
test_invariant_reduction()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;
  const auto valueType = TestType::createValueType();

  const auto predicate = &jlm::rvsdg::GraphImport::Create(graph, ControlType::Create(2), "");
  const auto value = &jlm::rvsdg::GraphImport::Create(graph, valueType, "");

  const auto gammaNode = GammaNode::create(predicate, 2);
  auto [input, branchArgument] = gammaNode->AddEntryVar(value);
  gammaNode->AddExitVar(branchArgument);

  auto & ex = GraphExport::Create(*gammaNode->output(0), "");

  view(&graph.GetRootRegion(), stdout);

  // Act
  const auto success = ReduceGammaInvariantVariables(*gammaNode);
  view(&graph.GetRootRegion(), stdout);

  // Assert
  assert(success);
  assert(ex.origin() == value);

  graph.PruneNodes();
  assert(graph.GetRootRegion().numNodes() == 0);
}

static void
test_control_constant_reduction()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto x = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(1), "x");

  auto c = match(1, { { 0, 0 } }, 1, 2, x);

  auto gamma = GammaNode::create(c, 2);

  auto t = &ControlConstantOperation::createTrue(*gamma->subregion(0));
  auto f = &ControlConstantOperation::createFalse(*gamma->subregion(1));

  auto n0 = &ControlConstantOperation::create(*gamma->subregion(0), 3, 0);
  auto n1 = &ControlConstantOperation::create(*gamma->subregion(1), 3, 1);

  auto xv1 = gamma->AddExitVar({ t, f });
  auto xv2 = gamma->AddExitVar({ n0, n1 });

  auto & ex1 = GraphExport::Create(*xv1.output, "");
  auto & ex2 = GraphExport::Create(*xv2.output, "");

  view(&graph.GetRootRegion(), stdout);

  // Act
  auto gammaNode = TryGetOwnerNode<GammaNode>(*ex1.origin());
  ReduceGammaControlConstant(*gammaNode);
  view(&graph.GetRootRegion(), stdout);

  // Assert
  auto [matchNode, matchOperation] = TryGetSimpleNodeAndOptionalOp<MatchOperation>(*ex1.origin());
  assert(matchNode && matchOperation);
  assert(matchOperation->default_alternative() == 0);

  assert(TryGetOwnerNode<Node>(*ex2.origin()) == gamma);
}

static void
test_control_constant_reduction2()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph graph;

  auto import = &jlm::rvsdg::GraphImport::Create(graph, BitType::Create(2), "import");

  auto c = match(2, { { 3, 2 }, { 2, 1 }, { 1, 0 } }, 3, 4, import);

  auto gamma = GammaNode::create(c, 4);

  auto t1 = &ControlConstantOperation::createTrue(*gamma->subregion(0));
  auto t2 = &ControlConstantOperation::createTrue(*gamma->subregion(1));
  auto t3 = &ControlConstantOperation::createTrue(*gamma->subregion(2));
  auto f = &ControlConstantOperation::createFalse(*gamma->subregion(3));

  auto xv = gamma->AddExitVar({ t1, t2, t3, f });

  auto & ex = GraphExport::Create(*xv.output, "");

  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Act
  auto gammaNode = TryGetOwnerNode<GammaNode>(*ex.origin());
  ReduceGammaControlConstant(*gammaNode);
  jlm::rvsdg::view(&graph.GetRootRegion(), stdout);

  // Assert
  auto match = TryGetOwnerNode<Node>(*ex.origin());
  assert(is<MatchOperation>(match));
}

static void
TestRemoveGammaOutputsWhere()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto vt = TestType::createValueType();
  ControlType ct(2);

  auto predicate = &jlm::rvsdg::GraphImport::Create(rvsdg, ControlType::Create(2), "");
  auto v0 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");
  auto v1 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");
  auto v2 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");
  auto v3 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");

  auto gammaNode = GammaNode::create(predicate, 2);
  auto gammaInput0 = gammaNode->AddEntryVar(v0);
  auto gammaInput1 = gammaNode->AddEntryVar(v1);
  auto gammaInput2 = gammaNode->AddEntryVar(v2);
  auto gammaInput3 = gammaNode->AddEntryVar(v3);

  auto gammaOutput0 = gammaNode->AddExitVar(gammaInput0.branchArgument);
  auto gammaOutput1 = gammaNode->AddExitVar(gammaInput1.branchArgument);
  auto gammaOutput2 = gammaNode->AddExitVar(gammaInput2.branchArgument);
  auto gammaOutput3 = gammaNode->AddExitVar(gammaInput3.branchArgument);

  GraphExport::Create(*gammaOutput0.output, "");
  GraphExport::Create(*gammaOutput2.output, "");

  // Act & Assert
  assert(gammaNode->noutputs() == 4);

  // Remove gammaOutput1
  gammaNode->RemoveGammaOutputsWhere(
      [&](const jlm::rvsdg::Output & output)
      {
        return output.index() == gammaOutput1.output->index();
      });
  assert(gammaNode->noutputs() == 3);
  assert(gammaNode->subregion(0)->nresults() == 3);
  assert(gammaNode->subregion(1)->nresults() == 3);
  assert(gammaOutput2.output->index() == 1);
  assert(gammaOutput3.output->index() == 2);

  // Try to remove gammaOutput2. This should result in no change as gammaOutput2 still has users.
  gammaNode->RemoveGammaOutputsWhere(
      [&](const jlm::rvsdg::Output & output)
      {
        return output.index() == gammaOutput2.output->index();
      });
  assert(gammaNode->noutputs() == 3);
  assert(gammaNode->subregion(0)->nresults() == 3);
  assert(gammaNode->subregion(1)->nresults() == 3);
  assert(gammaOutput2.output->index() == 1);
  assert(gammaOutput3.output->index() == 2);
}

static void
TestPruneOutputs()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto vt = TestType::createValueType();
  ControlType ct(2);

  auto predicate = &jlm::rvsdg::GraphImport::Create(rvsdg, ControlType::Create(2), "");
  auto v0 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");
  auto v1 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");
  auto v2 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");
  auto v3 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");

  auto gammaNode = GammaNode::create(predicate, 2);
  auto gammaInput0 = gammaNode->AddEntryVar(v0);
  auto gammaInput1 = gammaNode->AddEntryVar(v1);
  auto gammaInput2 = gammaNode->AddEntryVar(v2);
  auto gammaInput3 = gammaNode->AddEntryVar(v3);

  auto gammaOutput0 = gammaNode->AddExitVar(gammaInput0.branchArgument);
  gammaNode->AddExitVar(gammaInput1.branchArgument);
  auto gammaOutput2 = gammaNode->AddExitVar(gammaInput2.branchArgument);
  gammaNode->AddExitVar(gammaInput3.branchArgument);

  GraphExport::Create(*gammaOutput0.output, "");
  GraphExport::Create(*gammaOutput2.output, "");

  // Act
  gammaNode->PruneOutputs();

  // Assert
  assert(gammaNode->noutputs() == 2);
  assert(gammaNode->subregion(0)->nresults() == 2);
  assert(gammaNode->subregion(1)->nresults() == 2);

  assert(gammaOutput0.output->index() == 0);
  assert(gammaNode->GetExitVars()[0].output == gammaOutput0.output);

  assert(gammaOutput2.output->index() == 1);
  assert(gammaNode->GetExitVars()[1].output == gammaOutput2.output);
}

static void
TestIsInvariant()
{
  using namespace jlm::rvsdg;

  // Arrange
  Graph rvsdg;
  auto vt = TestType::createValueType();
  ControlType ct(2);

  auto predicate = &jlm::rvsdg::GraphImport::Create(rvsdg, ControlType::Create(2), "");
  auto v0 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");
  auto v1 = &jlm::rvsdg::GraphImport::Create(rvsdg, vt, "");

  auto gammaNode = GammaNode::create(predicate, 2);
  auto gammaInput0 = gammaNode->AddEntryVar(v0);
  auto gammaInput1 = gammaNode->AddEntryVar(v1);
  auto gammaInput2 = gammaNode->AddEntryVar(v1);

  auto gammaOutput0 = gammaNode->AddExitVar(gammaInput0.branchArgument);
  auto gammaOutput1 =
      gammaNode->AddExitVar({ gammaInput1.branchArgument[0], gammaInput2.branchArgument[1] });
  auto gammaOutput2 =
      gammaNode->AddExitVar({ gammaInput0.branchArgument[0], gammaInput2.branchArgument[1] });

  // Act & Assert
  std::optional<jlm::rvsdg::Output *> invariantOrigin;
  invariantOrigin = jlm::rvsdg::GetGammaInvariantOrigin(*gammaNode, gammaOutput0);
  assert(invariantOrigin && *invariantOrigin == v0);

  invariantOrigin = jlm::rvsdg::GetGammaInvariantOrigin(*gammaNode, gammaOutput1);
  assert(invariantOrigin && *invariantOrigin == v1);

  invariantOrigin = jlm::rvsdg::GetGammaInvariantOrigin(*gammaNode, gammaOutput2);
  assert(!invariantOrigin);
}

static void
test_main()
{
  test_gamma();
  TestRemoveGammaOutputsWhere();
  TestPruneOutputs();
  TestIsInvariant();

  test_predicate_reduction();
  test_invariant_reduction();
  test_control_constant_reduction();
  test_control_constant_reduction2();
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/test-gamma", test_main)
