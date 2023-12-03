/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/Statistics.hpp>

#include <cassert>
#include <jlm/rvsdg/view.hpp>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunAndersen(jlm::llvm::RvsdgModule & module)
{
  using namespace jlm::llvm;

  aa::Andersen andersen;
  return andersen.Analyze(module);
}

/**
 * @brief Ensures the given PointsToGraph node points to exactly the given set of target nodes.
 * @param node the source node
 * @param targets a set of nodes that \p node should point to.
 */
[[nodiscard]] static bool
TargetsExactly(
    const jlm::llvm::aa::PointsToGraph::Node & node,
    const std::unordered_set<const jlm::llvm::aa::PointsToGraph::Node *> & targets)
{
  using namespace jlm::llvm::aa;

  std::unordered_set<const PointsToGraph::Node *> nodeTargets;
  for (auto & target : node.Targets())
    nodeTargets.insert(&target);
  return targets == nodeTargets;
}

/**
 * @brief Ensures that the set of Memory Nodes escaping the PointsToGraph is exactly equal
 * to the given set of nodes.
 * @param ptg the PointsToGraph
 * @param nodes the complete set of nodes that should have escaped
 * @return true if the \p ptg's escaped set is identical to \p nodes
 */
[[nodiscard]] static bool
EscapedIsExactly(
    const jlm::llvm::aa::PointsToGraph & ptg,
    const std::unordered_set<const jlm::llvm::aa::PointsToGraph::MemoryNode *> & nodes)
{
  return ptg.GetEscapedMemoryNodes() == jlm::util::HashSet(nodes);
}

static void
TestStore1()
{
  jlm::tests::StoreTest1 test;
  jlm::rvsdg::view(test.graph().root(), stdout);

  auto ptg = RunAndersen(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*ptg) << std::endl;

  assert(ptg->NumAllocaNodes() == 4);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 5);

  auto & alloca_a = ptg->GetAllocaNode(*test.alloca_a);
  auto & alloca_b = ptg->GetAllocaNode(*test.alloca_b);
  auto & alloca_c = ptg->GetAllocaNode(*test.alloca_c);
  auto & alloca_d = ptg->GetAllocaNode(*test.alloca_d);

  auto & palloca_a = ptg->GetRegisterNode(*test.alloca_a->output(0));
  auto & palloca_b = ptg->GetRegisterNode(*test.alloca_b->output(0));
  auto & palloca_c = ptg->GetRegisterNode(*test.alloca_c->output(0));
  auto & palloca_d = ptg->GetRegisterNode(*test.alloca_d->output(0));

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & plambda = ptg->GetRegisterNode(*test.lambda->output());

  assert(TargetsExactly(alloca_a, { &alloca_b }));
  assert(TargetsExactly(alloca_b, { &alloca_c }));
  assert(TargetsExactly(alloca_c, { &alloca_d }));
  assert(TargetsExactly(alloca_d, {}));

  assert(TargetsExactly(palloca_a, { &alloca_a }));
  assert(TargetsExactly(palloca_b, { &alloca_b }));
  assert(TargetsExactly(palloca_c, { &alloca_c }));
  assert(TargetsExactly(palloca_d, { &alloca_d }));

  assert(TargetsExactly(lambda, {}));
  assert(TargetsExactly(plambda, { &lambda }));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static int
TestAndersen()
{
  TestStore1();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen", TestAndersen)