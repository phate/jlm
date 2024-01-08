/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
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
  assert(ptg->NumRegisterSetNodes() == 5);

  auto & alloca_a = ptg->GetAllocaNode(*test.alloca_a);
  auto & alloca_b = ptg->GetAllocaNode(*test.alloca_b);
  auto & alloca_c = ptg->GetAllocaNode(*test.alloca_c);
  auto & alloca_d = ptg->GetAllocaNode(*test.alloca_d);

  auto & palloca_a = ptg->GetRegisterSetNode(*test.alloca_a->output(0));
  auto & palloca_b = ptg->GetRegisterSetNode(*test.alloca_b->output(0));
  auto & palloca_c = ptg->GetRegisterSetNode(*test.alloca_c->output(0));
  auto & palloca_d = ptg->GetRegisterSetNode(*test.alloca_d->output(0));

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & plambda = ptg->GetRegisterSetNode(*test.lambda->output());

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
  assert(ptg->NumRegisterSetNodes() == 6);

  auto & alloca_a = ptg->GetAllocaNode(*test.alloca_a);
  auto & alloca_b = ptg->GetAllocaNode(*test.alloca_b);
  auto & alloca_x = ptg->GetAllocaNode(*test.alloca_x);
  auto & alloca_y = ptg->GetAllocaNode(*test.alloca_y);
  auto & alloca_p = ptg->GetAllocaNode(*test.alloca_p);

  auto & palloca_a = ptg->GetRegisterSetNode(*test.alloca_a->output(0));
  auto & palloca_b = ptg->GetRegisterSetNode(*test.alloca_b->output(0));
  auto & palloca_x = ptg->GetRegisterSetNode(*test.alloca_x->output(0));
  auto & palloca_y = ptg->GetRegisterSetNode(*test.alloca_y->output(0));
  auto & palloca_p = ptg->GetRegisterSetNode(*test.alloca_p->output(0));

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & plambda = ptg->GetRegisterSetNode(*test.lambda->output());

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
  assert(ptg->NumRegisterSetNodes() == 3);

  auto & loadResult = ptg->GetRegisterSetNode(*test.load_p->output(0));

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOutput = ptg->GetRegisterSetNode(*test.lambda->output());
  auto & lambdaArgument0 = ptg->GetRegisterSetNode(*test.lambda->fctargument(0));

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
  assert(ptg->NumRegisterSetNodes() == 8);

  auto & alloca_a = ptg->GetAllocaNode(*test.alloca_a);
  auto & alloca_b = ptg->GetAllocaNode(*test.alloca_b);
  auto & alloca_x = ptg->GetAllocaNode(*test.alloca_x);
  auto & alloca_y = ptg->GetAllocaNode(*test.alloca_y);
  auto & alloca_p = ptg->GetAllocaNode(*test.alloca_p);

  auto & pload_x = ptg->GetRegisterSetNode(*test.load_x->output(0));
  auto & pload_a = ptg->GetRegisterSetNode(*test.load_a->output(0));

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
  assert(ptg->NumRegisterSetNodes() == 2);

  auto & lambdaMemoryNode = ptg->GetLambdaNode(test.Lambda());
  auto & undefValueNode = ptg->GetRegisterSetNode(*test.UndefValueNode()->output(0));

  assert(TargetsExactly(undefValueNode, {}));
  assert(EscapedIsExactly(*ptg, { &lambdaMemoryNode }));
}

static void
TestGetElementPtr()
{
  jlm::tests::GetElementPtrTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 2);

  // We only care about the getelemenptr's in this test, skipping the validation for all other nodes
  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & gepX = ptg->GetRegisterSetNode(*test.getElementPtrX->output(0));
  auto & gepY = ptg->GetRegisterSetNode(*test.getElementPtrY->output(0));

  // The RegisterSetNode is the same
  assert(&gepX == &gepY);

  assert(TargetsExactly(gepX, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestBitCast()
{
  jlm::tests::BitCastTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 2);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOut = ptg->GetRegisterSetNode(*test.lambda->output());
  auto & lambdaArg = ptg->GetRegisterSetNode(*test.lambda->fctargument(0));
  auto & bitCast = ptg->GetRegisterSetNode(*test.bitCast->output(0));

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
  assert(ptg->NumRegisterSetNodes() == 3);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOut = ptg->GetRegisterSetNode(*test.lambda->output());
  auto & lambdaArg = ptg->GetRegisterSetNode(*test.lambda->fctargument(0));

  auto & constantPointerNull = ptg->GetRegisterSetNode(*test.constantPointerNullNode->output(0));

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
  assert(ptg->NumRegisterSetNodes() == 4);

  auto & lambdaTestMemoryNode = ptg->GetLambdaNode(test.GetLambdaTest());
  auto & externalMemoryNode = ptg->GetExternalMemoryNode();

  auto & callOutput0 = ptg->GetRegisterSetNode(*test.GetCallNode().output(0));
  auto & bits2ptr = ptg->GetRegisterSetNode(*test.GetBitsToPtrNode().output(0));

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
  assert(ptg->NumRegisterSetNodes() == 10);

  auto & alloca_x = ptg->GetAllocaNode(*test.alloca_x);
  auto & alloca_y = ptg->GetAllocaNode(*test.alloca_y);
  auto & alloca_z = ptg->GetAllocaNode(*test.alloca_z);

  auto & palloca_x = ptg->GetRegisterSetNode(*test.alloca_x->output(0));
  auto & palloca_y = ptg->GetRegisterSetNode(*test.alloca_y->output(0));
  auto & palloca_z = ptg->GetRegisterSetNode(*test.alloca_z->output(0));

  auto & lambda_f = ptg->GetLambdaNode(*test.lambda_f);
  auto & lambda_g = ptg->GetLambdaNode(*test.lambda_g);
  auto & lambda_h = ptg->GetLambdaNode(*test.lambda_h);

  auto & plambda_f = ptg->GetRegisterSetNode(*test.lambda_f->output());
  auto & plambda_g = ptg->GetRegisterSetNode(*test.lambda_g->output());
  auto & plambda_h = ptg->GetRegisterSetNode(*test.lambda_h->output());

  auto & lambda_f_arg0 = ptg->GetRegisterSetNode(*test.lambda_f->fctargument(0));
  auto & lambda_f_arg1 = ptg->GetRegisterSetNode(*test.lambda_f->fctargument(1));

  auto & lambda_g_arg0 = ptg->GetRegisterSetNode(*test.lambda_g->fctargument(0));
  auto & lambda_g_arg1 = ptg->GetRegisterSetNode(*test.lambda_g->fctargument(1));

  auto & lambda_h_cv0 = ptg->GetRegisterSetNode(*test.lambda_h->cvargument(0));
  auto & lambda_h_cv1 = ptg->GetRegisterSetNode(*test.lambda_h->cvargument(1));

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

static void
TestCall2()
{
  jlm::tests::CallTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 3);
  assert(ptg->NumMallocNodes() == 1);
  assert(ptg->NumImportNodes() == 0);
  assert(ptg->NumRegisterSetNodes() == 7);

  auto & lambda_create = ptg->GetLambdaNode(*test.lambda_create);
  auto & lambda_create_out = ptg->GetRegisterSetNode(*test.lambda_create->output());

  auto & lambda_destroy = ptg->GetLambdaNode(*test.lambda_destroy);
  auto & lambda_destroy_out = ptg->GetRegisterSetNode(*test.lambda_destroy->output());
  auto & lambda_destroy_arg = ptg->GetRegisterSetNode(*test.lambda_destroy->fctargument(0));

  auto & lambda_test = ptg->GetLambdaNode(*test.lambda_test);
  auto & lambda_test_out = ptg->GetRegisterSetNode(*test.lambda_test->output());
  auto & lambda_test_cv1 = ptg->GetRegisterSetNode(*test.lambda_test->cvargument(0));
  auto & lambda_test_cv2 = ptg->GetRegisterSetNode(*test.lambda_test->cvargument(1));

  auto & call_create1_out = ptg->GetRegisterSetNode(*test.CallCreate1().output(0));
  auto & call_create2_out = ptg->GetRegisterSetNode(*test.CallCreate2().output(0));

  auto & malloc = ptg->GetMallocNode(*test.malloc);
  auto & malloc_out = ptg->GetRegisterSetNode(*test.malloc->output(0));

  assert(TargetsExactly(lambda_create_out, { &lambda_create }));

  assert(TargetsExactly(lambda_destroy_out, { &lambda_destroy }));
  assert(TargetsExactly(lambda_destroy_arg, { &malloc }));

  assert(TargetsExactly(lambda_test_out, { &lambda_test }));
  assert(TargetsExactly(lambda_test_cv1, { &lambda_create }));
  assert(TargetsExactly(lambda_test_cv2, { &lambda_destroy }));

  assert(TargetsExactly(call_create1_out, { &malloc }));
  assert(TargetsExactly(call_create2_out, { &malloc }));

  assert(TargetsExactly(malloc_out, { &malloc }));

  assert(EscapedIsExactly(*ptg, { &lambda_test }));
}

static void
TestIndirectCall1()
{
  jlm::tests::IndirectCallTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 4);
  assert(ptg->NumImportNodes() == 0);
  assert(ptg->NumRegisterSetNodes() == 5);

  auto & lambda_three = ptg->GetLambdaNode(test.GetLambdaThree());
  auto & lambda_three_out = ptg->GetRegisterSetNode(*test.GetLambdaThree().output());

  auto & lambda_four = ptg->GetLambdaNode(test.GetLambdaFour());
  auto & lambda_four_out = ptg->GetRegisterSetNode(*test.GetLambdaFour().output());

  auto & lambda_indcall = ptg->GetLambdaNode(test.GetLambdaIndcall());
  auto & lambda_indcall_out = ptg->GetRegisterSetNode(*test.GetLambdaIndcall().output());
  auto & lambda_indcall_arg = ptg->GetRegisterSetNode(*test.GetLambdaIndcall().fctargument(0));

  auto & lambda_test = ptg->GetLambdaNode(test.GetLambdaTest());
  auto & lambda_test_out = ptg->GetRegisterSetNode(*test.GetLambdaTest().output());
  auto & lambda_test_cv0 = ptg->GetRegisterSetNode(*test.GetLambdaTest().cvargument(0));
  auto & lambda_test_cv1 = ptg->GetRegisterSetNode(*test.GetLambdaTest().cvargument(1));
  auto & lambda_test_cv2 = ptg->GetRegisterSetNode(*test.GetLambdaTest().cvargument(2));

  assert(TargetsExactly(lambda_three_out, { &lambda_three }));

  assert(TargetsExactly(lambda_four_out, { &lambda_four }));

  assert(TargetsExactly(lambda_indcall_out, { &lambda_indcall }));
  assert(TargetsExactly(lambda_indcall_arg, { &lambda_three, &lambda_four }));

  assert(TargetsExactly(lambda_test_out, { &lambda_test }));
  assert(TargetsExactly(lambda_test_cv0, { &lambda_indcall }));
  assert(TargetsExactly(lambda_test_cv1, { &lambda_four }));
  assert(TargetsExactly(lambda_test_cv2, { &lambda_three }));

  assert(EscapedIsExactly(*ptg, { &lambda_test }));
}

static void
TestIndirectCall2()
{
  jlm::tests::IndirectCallTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 3);
  assert(ptg->NumLambdaNodes() == 7);
  assert(ptg->NumDeltaNodes() == 2);
  assert(ptg->NumRegisterSetNodes() == 15);

  auto & lambdaThree = ptg->GetLambdaNode(test.GetLambdaThree());
  auto & lambdaThreeOutput = ptg->GetRegisterSetNode(*test.GetLambdaThree().output());

  auto & lambdaFour = ptg->GetLambdaNode(test.GetLambdaFour());
  auto & lambdaFourOutput = ptg->GetRegisterSetNode(*test.GetLambdaFour().output());

  assert(TargetsExactly(lambdaThreeOutput, { &lambdaThree }));
  assert(TargetsExactly(lambdaFourOutput, { &lambdaFour }));
}

static void
TestExternalCall()
{
  jlm::tests::ExternalCallTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 2);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumImportNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 9);

  auto & lambdaF = ptg->GetLambdaNode(test.LambdaF());
  auto & lambdaFArgument0 = ptg->GetRegisterSetNode(*test.LambdaF().fctargument(0));
  auto & lambdaFArgument1 = ptg->GetRegisterSetNode(*test.LambdaF().fctargument(1));
  auto & importG = ptg->GetImportNode(test.ExternalGArgument());

  auto & callResult = ptg->GetRegisterSetNode(*test.CallG().Result(0));

  auto & externalMemory = ptg->GetExternalMemoryNode();

  assert(TargetsExactly(lambdaFArgument0, { &lambdaF, &importG, &externalMemory }));
  assert(TargetsExactly(lambdaFArgument1, { &lambdaF, &importG, &externalMemory }));
  assert(TargetsExactly(callResult, { &lambdaF, &importG, &externalMemory }));
}

static void
TestGamma()
{
  jlm::tests::GammaTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 7);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);

  for (size_t n = 1; n < 5; n++)
  {
    auto & lambdaArgument = ptg->GetRegisterSetNode(*test.lambda->fctargument(n));
    assert(TargetsExactly(lambdaArgument, { &lambda, &ptg->GetExternalMemoryNode() }));
  }

  for (size_t n = 0; n < 4; n++)
  {
    auto & argument0 = ptg->GetRegisterSetNode(*test.gamma->entryvar(n)->argument(0));
    auto & argument1 = ptg->GetRegisterSetNode(*test.gamma->entryvar(n)->argument(1));

    assert(TargetsExactly(argument0, { &lambda, &ptg->GetExternalMemoryNode() }));
    assert(TargetsExactly(argument1, { &lambda, &ptg->GetExternalMemoryNode() }));
  }

  for (size_t n = 0; n < 4; n++)
  {
    auto & gammaOutput = ptg->GetRegisterSetNode(*test.gamma->exitvar(0));
    assert(TargetsExactly(gammaOutput, { &lambda, &ptg->GetExternalMemoryNode() }));
  }

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestTheta()
{
  jlm::tests::ThetaTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 3);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaArgument1 = ptg->GetRegisterSetNode(*test.lambda->fctargument(1));
  auto & lambdaOutput = ptg->GetRegisterSetNode(*test.lambda->output());

  auto & gepOutput = ptg->GetRegisterSetNode(*test.gep->output(0));

  auto & thetaArgument2 = ptg->GetRegisterSetNode(*test.theta->output(2)->argument());
  auto & thetaOutput2 = ptg->GetRegisterSetNode(*test.theta->output(2));

  assert(TargetsExactly(lambdaArgument1, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(lambdaOutput, { &lambda }));

  assert(TargetsExactly(gepOutput, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(TargetsExactly(thetaArgument2, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(thetaOutput2, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));
}

static void
TestDelta1()
{
  jlm::tests::DeltaTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 1);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumRegisterSetNodes() == 4);

  auto & delta_f = ptg->GetDeltaNode(*test.delta_f);
  auto & pdelta_f = ptg->GetRegisterSetNode(*test.delta_f->output());

  auto & lambda_g = ptg->GetLambdaNode(*test.lambda_g);
  auto & plambda_g = ptg->GetRegisterSetNode(*test.lambda_g->output());
  auto & lambda_g_arg0 = ptg->GetRegisterSetNode(*test.lambda_g->fctargument(0));

  auto & lambda_h = ptg->GetLambdaNode(*test.lambda_h);
  auto & plambda_h = ptg->GetRegisterSetNode(*test.lambda_h->output());
  auto & lambda_h_cv0 = ptg->GetRegisterSetNode(*test.lambda_h->cvargument(0));
  auto & lambda_h_cv1 = ptg->GetRegisterSetNode(*test.lambda_h->cvargument(1));

  assert(TargetsExactly(pdelta_f, { &delta_f }));

  assert(TargetsExactly(plambda_g, { &lambda_g }));
  assert(TargetsExactly(plambda_h, { &lambda_h }));

  assert(TargetsExactly(lambda_g_arg0, { &delta_f }));

  assert(TargetsExactly(lambda_h_cv0, { &delta_f }));
  assert(TargetsExactly(lambda_h_cv1, { &lambda_g }));

  assert(EscapedIsExactly(*ptg, { &lambda_h }));
}

static void
TestDelta2()
{
  jlm::tests::DeltaTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 2);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumRegisterSetNodes() == 4);

  auto & delta_d1 = ptg->GetDeltaNode(*test.delta_d1);
  auto & delta_d1_out = ptg->GetRegisterSetNode(*test.delta_d1->output());

  auto & delta_d2 = ptg->GetDeltaNode(*test.delta_d2);
  auto & delta_d2_out = ptg->GetRegisterSetNode(*test.delta_d2->output());

  auto & lambda_f1 = ptg->GetLambdaNode(*test.lambda_f1);
  auto & lambda_f1_out = ptg->GetRegisterSetNode(*test.lambda_f1->output());
  auto & lambda_f1_cvd1 = ptg->GetRegisterSetNode(*test.lambda_f1->cvargument(0));

  auto & lambda_f2 = ptg->GetLambdaNode(*test.lambda_f2);
  auto & lambda_f2_out = ptg->GetRegisterSetNode(*test.lambda_f2->output());
  auto & lambda_f2_cvd1 = ptg->GetRegisterSetNode(*test.lambda_f2->cvargument(0));
  auto & lambda_f2_cvd2 = ptg->GetRegisterSetNode(*test.lambda_f2->cvargument(1));
  auto & lambda_f2_cvf1 = ptg->GetRegisterSetNode(*test.lambda_f2->cvargument(2));

  assert(TargetsExactly(delta_d1_out, { &delta_d1 }));
  assert(TargetsExactly(delta_d2_out, { &delta_d2 }));

  assert(TargetsExactly(lambda_f1_out, { &lambda_f1 }));
  assert(TargetsExactly(lambda_f2_out, { &lambda_f2 }));

  assert(&lambda_f1_cvd1 == &delta_d1_out);
  assert(&lambda_f2_cvd1 == &delta_d1_out);
  assert(&lambda_f2_cvd2 == &delta_d2_out);
  assert(&lambda_f2_cvf1 == &lambda_f1_out);

  assert(EscapedIsExactly(*ptg, { &lambda_f2 }));
}

static void
TestImports()
{
  jlm::tests::ImportTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumImportNodes() == 2);
  assert(ptg->NumRegisterSetNodes() == 4);

  auto & d1 = ptg->GetImportNode(*test.import_d1);
  auto & import_d1 = ptg->GetRegisterSetNode(*test.import_d1);

  auto & d2 = ptg->GetImportNode(*test.import_d2);
  auto & import_d2 = ptg->GetRegisterSetNode(*test.import_d2);

  auto & lambda_f1 = ptg->GetLambdaNode(*test.lambda_f1);
  auto & lambda_f1_out = ptg->GetRegisterSetNode(*test.lambda_f1->output());
  auto & lambda_f1_cvd1 = ptg->GetRegisterSetNode(*test.lambda_f1->cvargument(0));

  auto & lambda_f2 = ptg->GetLambdaNode(*test.lambda_f2);
  auto & lambda_f2_out = ptg->GetRegisterSetNode(*test.lambda_f2->output());
  auto & lambda_f2_cvd1 = ptg->GetRegisterSetNode(*test.lambda_f2->cvargument(0));
  auto & lambda_f2_cvd2 = ptg->GetRegisterSetNode(*test.lambda_f2->cvargument(1));
  auto & lambda_f2_cvf1 = ptg->GetRegisterSetNode(*test.lambda_f2->cvargument(2));

  assert(TargetsExactly(import_d1, { &d1 }));
  assert(TargetsExactly(import_d2, { &d2 }));

  assert(TargetsExactly(lambda_f1_out, { &lambda_f1 }));
  assert(TargetsExactly(lambda_f2_out, { &lambda_f2 }));

  assert(&lambda_f1_cvd1 == &import_d1);
  assert(&lambda_f2_cvd1 == &import_d1);
  assert(&lambda_f2_cvd2 == &import_d2);
  assert(&lambda_f2_cvf1 == &lambda_f1_out);

  assert(EscapedIsExactly(*ptg, { &lambda_f2, &d1, &d2 }));
}

static void
TestPhi1()
{
  jlm::tests::PhiTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 1);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumRegisterSetNodes() == 5);

  auto & lambda_fib = ptg->GetLambdaNode(*test.lambda_fib);
  auto & lambda_fib_out = ptg->GetRegisterSetNode(*test.lambda_fib->output());
  auto & lambda_fib_arg1 = ptg->GetRegisterSetNode(*test.lambda_fib->fctargument(1));

  auto & lambda_test = ptg->GetLambdaNode(*test.lambda_test);
  auto & lambda_test_out = ptg->GetRegisterSetNode(*test.lambda_test->output());

  auto & phi_rv = ptg->GetRegisterSetNode(*test.phi->begin_rv().output());
  auto & phi_rv_arg = ptg->GetRegisterSetNode(*test.phi->begin_rv().output()->argument());

  auto & gamma_result = ptg->GetRegisterSetNode(*test.gamma->subregion(0)->argument(1));
  auto & gamma_fib = ptg->GetRegisterSetNode(*test.gamma->subregion(0)->argument(2));

  auto & alloca = ptg->GetAllocaNode(*test.alloca);
  auto & alloca_out = ptg->GetRegisterSetNode(*test.alloca->output(0));

  assert(TargetsExactly(lambda_fib_out, { &lambda_fib }));
  assert(TargetsExactly(lambda_fib_arg1, { &alloca }));

  assert(TargetsExactly(lambda_test_out, { &lambda_test }));

  assert(&phi_rv == &lambda_fib_out);
  assert(TargetsExactly(phi_rv_arg, { &lambda_fib }));

  assert(TargetsExactly(gamma_result, { &alloca }));
  assert(TargetsExactly(gamma_fib, { &lambda_fib }));

  assert(TargetsExactly(alloca_out, { &alloca }));

  assert(EscapedIsExactly(*ptg, { &lambda_test }));
}

static void
TestExternalMemory()
{
  jlm::tests::ExternalMemoryTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 3);

  auto & lambdaF = ptg->GetLambdaNode(*test.LambdaF);
  auto & lambdaFArgument0 = ptg->GetRegisterSetNode(*test.LambdaF->fctargument(0));
  auto & lambdaFArgument1 = ptg->GetRegisterSetNode(*test.LambdaF->fctargument(1));

  assert(TargetsExactly(lambdaFArgument0, { &lambdaF, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(lambdaFArgument1, { &lambdaF, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambdaF }));
}

static void
TestEscapedMemory1()
{
  jlm::tests::EscapedMemoryTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 4);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 7);

  auto & lambdaTestArgument0 = ptg->GetRegisterSetNode(*test.LambdaTest->fctargument(0));
  auto & lambdaTestCv0 = ptg->GetRegisterSetNode(*test.LambdaTest->cvargument(0));
  auto & loadNode1Output = ptg->GetRegisterSetNode(*test.LoadNode1->output(0));

  auto deltaA = &ptg->GetDeltaNode(*test.DeltaA);
  auto deltaB = &ptg->GetDeltaNode(*test.DeltaB);
  auto deltaX = &ptg->GetDeltaNode(*test.DeltaX);
  auto deltaY = &ptg->GetDeltaNode(*test.DeltaY);
  auto lambdaTest = &ptg->GetLambdaNode(*test.LambdaTest);
  auto externalMemory = &ptg->GetExternalMemoryNode();

  assert(
      TargetsExactly(lambdaTestArgument0, { deltaA, deltaX, deltaY, lambdaTest, externalMemory }));
  assert(TargetsExactly(lambdaTestCv0, { deltaB }));
  assert(TargetsExactly(loadNode1Output, { deltaA, deltaX, deltaY, lambdaTest, externalMemory }));

  assert(EscapedIsExactly(*ptg, { lambdaTest, deltaA, deltaX, deltaY }));
}

static void
TestEscapedMemory2()
{
  jlm::tests::EscapedMemoryTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumImportNodes() == 2);
  assert(ptg->NumLambdaNodes() == 3);
  assert(ptg->NumMallocNodes() == 2);
  assert(ptg->NumRegisterSetNodes() == 8);

  auto returnAddressFunction = &ptg->GetLambdaNode(*test.ReturnAddressFunction);
  auto callExternalFunction1 = &ptg->GetLambdaNode(*test.CallExternalFunction1);
  auto callExternalFunction2 = &ptg->GetLambdaNode(*test.CallExternalFunction2);
  auto returnAddressMalloc = &ptg->GetMallocNode(*test.ReturnAddressMalloc);
  auto callExternalFunction1Malloc = &ptg->GetMallocNode(*test.CallExternalFunction1Malloc);
  auto externalMemory = &ptg->GetExternalMemoryNode();
  auto externalFunction1Import = &ptg->GetImportNode(*test.ExternalFunction1Import);
  auto externalFunction2Import = &ptg->GetImportNode(*test.ExternalFunction2Import);

  auto & externalFunction2CallResult =
      ptg->GetRegisterSetNode(*test.ExternalFunction2Call->Result(0));

  assert(TargetsExactly(
      externalFunction2CallResult,
      { returnAddressFunction,
        callExternalFunction1,
        callExternalFunction2,
        externalMemory,
        returnAddressMalloc,
        callExternalFunction1Malloc,
        externalFunction1Import,
        externalFunction2Import }));

  assert(EscapedIsExactly(
      *ptg,
      { returnAddressFunction,
        callExternalFunction1,
        callExternalFunction2,
        returnAddressMalloc,
        callExternalFunction1Malloc,
        externalFunction1Import,
        externalFunction2Import }));
}

static void
TestEscapedMemory3()
{
  jlm::tests::EscapedMemoryTest3 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 1);
  assert(ptg->NumImportNodes() == 1);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumRegisterSetNodes() == 4);

  auto lambdaTest = &ptg->GetLambdaNode(*test.LambdaTest);
  auto deltaGlobal = &ptg->GetDeltaNode(*test.DeltaGlobal);
  auto importExternalFunction = &ptg->GetImportNode(*test.ImportExternalFunction);
  auto externalMemory = &ptg->GetExternalMemoryNode();

  auto & callExternalFunctionResult =
      ptg->GetRegisterSetNode(*test.CallExternalFunction->Result(0));

  assert(TargetsExactly(
      callExternalFunctionResult,
      { lambdaTest, deltaGlobal, importExternalFunction, externalMemory }));

  assert(EscapedIsExactly(*ptg, { lambdaTest, deltaGlobal, importExternalFunction }));
}

static void
TestMemcpy()
{
  jlm::tests::MemcpyTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 2);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumRegisterSetNodes() == 4);

  auto localArray = &ptg->GetDeltaNode(test.LocalArray());
  auto globalArray = &ptg->GetDeltaNode(test.GlobalArray());

  auto & memCpyDest = ptg->GetRegisterSetNode(*test.Memcpy().input(0)->origin());
  auto & memCpySrc = ptg->GetRegisterSetNode(*test.Memcpy().input(1)->origin());

  auto lambdaF = &ptg->GetLambdaNode(test.LambdaF());
  auto lambdaG = &ptg->GetLambdaNode(test.LambdaG());

  assert(TargetsExactly(memCpyDest, { globalArray }));
  assert(TargetsExactly(memCpySrc, { localArray }));

  assert(EscapedIsExactly(*ptg, { globalArray, localArray, lambdaF, lambdaG }));
}

static void
TestLinkedList()
{
  jlm::tests::LinkedListTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 1);
  assert(ptg->NumDeltaNodes() == 1);
  assert(ptg->NumLambdaNodes() == 1);

  auto & allocaNode = ptg->GetAllocaNode(test.GetAlloca());
  auto & deltaMyListNode = ptg->GetDeltaNode(test.GetDeltaMyList());
  auto & lambdaNextNode = ptg->GetLambdaNode(test.GetLambdaNext());
  auto & externalMemoryNode = ptg->GetExternalMemoryNode();

  assert(TargetsExactly(allocaNode, { &deltaMyListNode, &lambdaNextNode, &externalMemoryNode }));
  assert(
      TargetsExactly(deltaMyListNode, { &deltaMyListNode, &lambdaNextNode, &externalMemoryNode }));
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
  TestCall2();
  TestIndirectCall1();
  TestIndirectCall2();
  TestExternalCall();
  TestGamma();
  TestTheta();
  TestDelta1();
  TestDelta2();
  TestImports();
  TestPhi1();
  TestExternalMemory();
  TestEscapedMemory1();
  TestEscapedMemory2();
  TestEscapedMemory3();
  TestMemcpy();
  TestLinkedList();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen", TestAndersen)
