/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

#include <cassert>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunAndersen(jlm::llvm::RvsdgModule & module)
{
  using namespace jlm::llvm;

  aa::Andersen andersen;
  return andersen.Analyze(module);
}

/**
 * @brief Checks that the given PointsToGraph node points to exactly the given set of target nodes.
 * @param node the source node
 * @param targets a set of nodes that \p node should point to.
 * @return false if the check fails, true otherwise
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
 * @brief Checks that the set of Memory Nodes escaping the PointsToGraph is exactly equal
 * to the given set of nodes.
 * @param ptg the PointsToGraph
 * @param nodes the complete set of nodes that should have escaped
 * @return true if the \p ptg's escaped set is identical to \p nodes, false otherwise
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
  const auto ptg = RunAndersen(test.module());

  // std::unordered_map<const jlm::rvsdg::output*, std::string> outputMap;
  // std::cout << jlm::rvsdg::view(test.graph().root(), outputMap) << std::endl;
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*ptg, outputMap) << std::endl;

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

static void
TestStore2()
{
  jlm::tests::StoreTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 5);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 6);

  auto & alloca_a = ptg->GetAllocaNode(*test.alloca_a);
  auto & alloca_b = ptg->GetAllocaNode(*test.alloca_b);
  auto & alloca_x = ptg->GetAllocaNode(*test.alloca_x);
  auto & alloca_y = ptg->GetAllocaNode(*test.alloca_y);
  auto & alloca_p = ptg->GetAllocaNode(*test.alloca_p);

  auto & palloca_a = ptg->GetRegisterNode(*test.alloca_a->output(0));
  auto & palloca_b = ptg->GetRegisterNode(*test.alloca_b->output(0));
  auto & palloca_x = ptg->GetRegisterNode(*test.alloca_x->output(0));
  auto & palloca_y = ptg->GetRegisterNode(*test.alloca_y->output(0));
  auto & palloca_p = ptg->GetRegisterNode(*test.alloca_p->output(0));

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & plambda = ptg->GetRegisterNode(*test.lambda->output());

  assert(TargetsExactly(alloca_a, {}));
  assert(TargetsExactly(alloca_b, {}));
  assert(TargetsExactly(alloca_x, { &alloca_a }));
  assert(TargetsExactly(alloca_y, { &alloca_b }));
  assert(TargetsExactly(alloca_p, { &alloca_x, &alloca_y }));

  assert(TargetsExactly(palloca_a, { &alloca_a }));
  assert(TargetsExactly(palloca_b, { &alloca_b }));
  assert(TargetsExactly(palloca_x, { &alloca_x }));
  assert(TargetsExactly(palloca_y, { &alloca_y }));
  assert(TargetsExactly(palloca_p, { &alloca_p }));

  assert(TargetsExactly(lambda, {}));
  assert(TargetsExactly(plambda, { &lambda }));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestLoad1()
{
  jlm::tests::LoadTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 3);

  auto & loadResult = ptg->GetRegisterNode(*test.load_p->output(0));

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOutput = ptg->GetRegisterNode(*test.lambda->output());
  auto & lambdaArgument0 = ptg->GetRegisterNode(*test.lambda->fctargument(0));

  assert(TargetsExactly(loadResult, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(TargetsExactly(lambdaOutput, { &lambda }));
  assert(TargetsExactly(lambdaArgument0, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestLoad2()
{
  jlm::tests::LoadTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 5);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 8);

  auto & alloca_a = ptg->GetAllocaNode(*test.alloca_a);
  auto & alloca_b = ptg->GetAllocaNode(*test.alloca_b);
  auto & alloca_x = ptg->GetAllocaNode(*test.alloca_x);
  auto & alloca_y = ptg->GetAllocaNode(*test.alloca_y);
  auto & alloca_p = ptg->GetAllocaNode(*test.alloca_p);

  auto & pload_x = ptg->GetRegisterNode(*test.load_x->output(0));
  auto & pload_a = ptg->GetRegisterNode(*test.load_a->output(0));

  auto & lambdaMemoryNode = ptg->GetLambdaNode(*test.lambda);

  assert(TargetsExactly(alloca_x, { &alloca_a }));
  assert(TargetsExactly(alloca_y, { &alloca_a, &alloca_b }));
  assert(TargetsExactly(alloca_p, { &alloca_x }));

  assert(TargetsExactly(pload_x, { &alloca_x }));
  assert(TargetsExactly(pload_a, { &alloca_a }));

  assert(EscapedIsExactly(*ptg, { &lambdaMemoryNode }));
}

static void
TestLoadFromUndef()
{
  jlm::tests::LoadFromUndefTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 2);

  auto & lambdaMemoryNode = ptg->GetLambdaNode(test.Lambda());
  auto & undefValueNode = ptg->GetRegisterNode(*test.UndefValueNode()->output(0));

  assert(TargetsExactly(undefValueNode, {}));
  assert(EscapedIsExactly(*ptg, { &lambdaMemoryNode }));
}

static void
TestGetElementPtr()
{
  jlm::tests::GetElementPtrTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 4);

  /*
    We only care about the getelemenptr's in this test, skipping the validation
    for all other nodes.
  */
  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & gepX = ptg->GetRegisterNode(*test.getElementPtrX->output(0));
  auto & gepY = ptg->GetRegisterNode(*test.getElementPtrY->output(0));

  assert(TargetsExactly(gepX, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(gepY, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestBitCast()
{
  jlm::tests::BitCastTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 3);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOut = ptg->GetRegisterNode(*test.lambda->output());
  auto & lambdaArg = ptg->GetRegisterNode(*test.lambda->fctargument(0));
  auto & bitCast = ptg->GetRegisterNode(*test.bitCast->output(0));

  assert(TargetsExactly(lambdaOut, { &lambda }));
  assert(TargetsExactly(lambdaArg, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(bitCast, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestConstantPointerNull()
{
  jlm::tests::ConstantPointerNullTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterNodes() == 3);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOut = ptg->GetRegisterNode(*test.lambda->output());
  auto & lambdaArg = ptg->GetRegisterNode(*test.lambda->fctargument(0));

  auto & constantPointerNull = ptg->GetRegisterNode(*test.constantPointerNullNode->output(0));

  assert(TargetsExactly(lambdaOut, { &lambda }));
  assert(TargetsExactly(lambdaArg, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(constantPointerNull, {}));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestBits2Ptr()
{
  jlm::tests::Bits2PtrTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumRegisterNodes() == 5);

  auto & lambdaTestMemoryNode = ptg->GetLambdaNode(test.GetLambdaTest());
  auto & externalMemoryNode = ptg->GetExternalMemoryNode();

  auto & callOutput0 = ptg->GetRegisterNode(*test.GetCallNode().output(0));
  auto & bits2ptr = ptg->GetRegisterNode(*test.GetBitsToPtrNode().output(0));

  assert(TargetsExactly(callOutput0, { &lambdaTestMemoryNode, &externalMemoryNode }));
  assert(TargetsExactly(bits2ptr, { &lambdaTestMemoryNode, &externalMemoryNode }));

  assert(EscapedIsExactly(*ptg, { &lambdaTestMemoryNode }));
}

static void
TestCall1()
{
  jlm::tests::CallTest1 test;
  const auto ptg = RunAndersen(test.module());
  
  assert(ptg->NumAllocaNodes() == 3);
  assert(ptg->NumLambdaNodes() == 3);
  assert(ptg->NumRegisterNodes() == 12);

  auto & alloca_x = ptg->GetAllocaNode(*test.alloca_x);
  auto & alloca_y = ptg->GetAllocaNode(*test.alloca_y);
  auto & alloca_z = ptg->GetAllocaNode(*test.alloca_z);

  auto & palloca_x = ptg->GetRegisterNode(*test.alloca_x->output(0));
  auto & palloca_y = ptg->GetRegisterNode(*test.alloca_y->output(0));
  auto & palloca_z = ptg->GetRegisterNode(*test.alloca_z->output(0));

  auto & lambda_f = ptg->GetLambdaNode(*test.lambda_f);
  auto & lambda_g = ptg->GetLambdaNode(*test.lambda_g);
  auto & lambda_h = ptg->GetLambdaNode(*test.lambda_h);

  auto & plambda_f = ptg->GetRegisterNode(*test.lambda_f->output());
  auto & plambda_g = ptg->GetRegisterNode(*test.lambda_g->output());
  auto & plambda_h = ptg->GetRegisterNode(*test.lambda_h->output());

  auto & lambda_f_arg0 = ptg->GetRegisterNode(*test.lambda_f->fctargument(0));
  auto & lambda_f_arg1 = ptg->GetRegisterNode(*test.lambda_f->fctargument(1));

  auto & lambda_g_arg0 = ptg->GetRegisterNode(*test.lambda_g->fctargument(0));
  auto & lambda_g_arg1 = ptg->GetRegisterNode(*test.lambda_g->fctargument(1));

  auto & lambda_h_cv0 = ptg->GetRegisterNode(*test.lambda_h->cvargument(0));
  auto & lambda_h_cv1 = ptg->GetRegisterNode(*test.lambda_h->cvargument(1));

  assert(TargetsExactly(palloca_x, { &alloca_x }));
  assert(TargetsExactly(palloca_y, { &alloca_y }));
  assert(TargetsExactly(palloca_z, { &alloca_z }));

  assert(TargetsExactly(plambda_f, { &lambda_f }));
  assert(TargetsExactly(plambda_g, { &lambda_g }));
  assert(TargetsExactly(plambda_h, { &lambda_h }));

  assert(TargetsExactly(lambda_f_arg0, { &alloca_x }));
  assert(TargetsExactly(lambda_f_arg1, { &alloca_y }));

  assert(TargetsExactly(lambda_g_arg0, { &alloca_z }));
  assert(TargetsExactly(lambda_g_arg1, { &alloca_z }));

  assert(TargetsExactly(lambda_h_cv0, { &lambda_f }));
  assert(TargetsExactly(lambda_h_cv1, { &lambda_g }));

  assert(EscapedIsExactly(*ptg, { &lambda_h }));
}

static int
TestAndersen()
{
  TestStore1();
  TestStore2();
  TestLoad1();
  TestLoad2();
  TestLoadFromUndef();
  TestGetElementPtr();
  TestBitCast();
  TestConstantPointerNull();
  TestBits2Ptr();
  TestCall1();
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen", TestAndersen)