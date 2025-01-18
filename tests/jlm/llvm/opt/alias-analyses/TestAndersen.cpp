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

static int
TestStore1()
{
  jlm::tests::StoreTest1 test;
  const auto ptg = RunAndersen(test.module());

  // std::unordered_map<const jlm::rvsdg::output*, std::string> outputMap;
  // std::cout << jlm::rvsdg::view(test.graph().GetRootRegion(), outputMap) << std::endl;
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*ptg, outputMap) << std::endl;

  assert(ptg->NumAllocaNodes() == 4);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 5);

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

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestStore1", TestStore1)

static int
TestStore2()
{
  jlm::tests::StoreTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 5);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 6);

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

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestStore2", TestStore2)

static int
TestLoad1()
{
  jlm::tests::LoadTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 3);

  auto & loadResult = ptg->GetRegisterNode(*test.load_p->output(0));

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOutput = ptg->GetRegisterNode(*test.lambda->output());
  auto & lambdaArgument0 = ptg->GetRegisterNode(*test.lambda->GetFunctionArguments()[0]);

  assert(TargetsExactly(loadResult, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(TargetsExactly(lambdaOutput, { &lambda }));
  assert(TargetsExactly(lambdaArgument0, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestLoad1", TestLoad1)

static int
TestLoad2()
{
  jlm::tests::LoadTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 5);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 8);

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

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestLoad2", TestLoad2)

static int
TestLoadFromUndef()
{
  jlm::tests::LoadFromUndefTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 2);

  auto & lambdaMemoryNode = ptg->GetLambdaNode(test.Lambda());
  auto & undefValueNode = ptg->GetRegisterNode(*test.UndefValueNode()->output(0));

  assert(TargetsExactly(undefValueNode, {}));
  assert(EscapedIsExactly(*ptg, { &lambdaMemoryNode }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestLoadFromUndef",
    TestLoadFromUndef)

static int
TestGetElementPtr()
{
  jlm::tests::GetElementPtrTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 4);

  // We only care about the getelemenptr's in this test, skipping the validation for all other nodes
  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & gepX = ptg->GetRegisterNode(*test.getElementPtrX->output(0));
  auto & gepY = ptg->GetRegisterNode(*test.getElementPtrY->output(0));

  // The RegisterNode is the same
  assert(&gepX == &gepY);

  assert(TargetsExactly(gepX, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestGetElementPtr",
    TestGetElementPtr)

static int
TestBitCast()
{
  jlm::tests::BitCastTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 3);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOut = ptg->GetRegisterNode(*test.lambda->output());
  auto & lambdaArg = ptg->GetRegisterNode(*test.lambda->GetFunctionArguments()[0]);
  auto & bitCast = ptg->GetRegisterNode(*test.bitCast->output(0));

  assert(TargetsExactly(lambdaOut, { &lambda }));
  assert(TargetsExactly(lambdaArg, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(bitCast, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestBitCast", TestBitCast)

static int
TestConstantPointerNull()
{
  jlm::tests::ConstantPointerNullTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 3);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaOut = ptg->GetRegisterNode(*test.lambda->output());
  auto & lambdaArg = ptg->GetRegisterNode(*test.lambda->GetFunctionArguments()[0]);

  auto & constantPointerNull = ptg->GetRegisterNode(*test.constantPointerNullNode->output(0));

  assert(TargetsExactly(lambdaOut, { &lambda }));
  assert(TargetsExactly(lambdaArg, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(constantPointerNull, {}));

  assert(EscapedIsExactly(*ptg, { &lambda }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestConstantPointerNull",
    TestConstantPointerNull)

static int
TestBits2Ptr()
{
  jlm::tests::Bits2PtrTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumMappedRegisters() == 5);

  auto & lambdaTestMemoryNode = ptg->GetLambdaNode(test.GetLambdaTest());
  auto & externalMemoryNode = ptg->GetExternalMemoryNode();

  auto & callOutput0 = ptg->GetRegisterNode(*test.GetCallNode().output(0));
  auto & bits2ptr = ptg->GetRegisterNode(*test.GetBitsToPtrNode().output(0));

  assert(TargetsExactly(callOutput0, { &lambdaTestMemoryNode, &externalMemoryNode }));
  assert(TargetsExactly(bits2ptr, { &lambdaTestMemoryNode, &externalMemoryNode }));

  assert(EscapedIsExactly(*ptg, { &lambdaTestMemoryNode }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestBits2Ptr", TestBits2Ptr)

static int
TestCall1()
{
  jlm::tests::CallTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 3);
  assert(ptg->NumLambdaNodes() == 3);
  assert(ptg->NumMappedRegisters() == 12);

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

  auto & lambda_f_arg0 = ptg->GetRegisterNode(*test.lambda_f->GetFunctionArguments()[0]);
  auto & lambda_f_arg1 = ptg->GetRegisterNode(*test.lambda_f->GetFunctionArguments()[1]);

  auto & lambda_g_arg0 = ptg->GetRegisterNode(*test.lambda_g->GetFunctionArguments()[0]);
  auto & lambda_g_arg1 = ptg->GetRegisterNode(*test.lambda_g->GetFunctionArguments()[1]);

  auto & lambda_h_cv0 = ptg->GetRegisterNode(*test.lambda_h->GetContextVars()[0].inner);
  auto & lambda_h_cv1 = ptg->GetRegisterNode(*test.lambda_h->GetContextVars()[1].inner);

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

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestCall1", TestCall1)

static int
TestCall2()
{
  jlm::tests::CallTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 3);
  assert(ptg->NumMallocNodes() == 1);
  assert(ptg->NumImportNodes() == 0);
  assert(ptg->NumMappedRegisters() == 11);

  auto & lambda_create = ptg->GetLambdaNode(*test.lambda_create);
  auto & lambda_create_out = ptg->GetRegisterNode(*test.lambda_create->output());

  auto & lambda_destroy = ptg->GetLambdaNode(*test.lambda_destroy);
  auto & lambda_destroy_out = ptg->GetRegisterNode(*test.lambda_destroy->output());
  auto & lambda_destroy_arg = ptg->GetRegisterNode(*test.lambda_destroy->GetFunctionArguments()[0]);

  auto & lambda_test = ptg->GetLambdaNode(*test.lambda_test);
  auto & lambda_test_out = ptg->GetRegisterNode(*test.lambda_test->output());
  auto & lambda_test_cv1 = ptg->GetRegisterNode(*test.lambda_test->GetContextVars()[0].inner);
  auto & lambda_test_cv2 = ptg->GetRegisterNode(*test.lambda_test->GetContextVars()[1].inner);

  auto & call_create1_out = ptg->GetRegisterNode(*test.CallCreate1().output(0));
  auto & call_create2_out = ptg->GetRegisterNode(*test.CallCreate2().output(0));

  auto & malloc = ptg->GetMallocNode(*test.malloc);
  auto & malloc_out = ptg->GetRegisterNode(*test.malloc->output(0));

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

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestCall2", TestCall2)

static int
TestIndirectCall1()
{
  jlm::tests::IndirectCallTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 4);
  assert(ptg->NumImportNodes() == 0);
  assert(ptg->NumMappedRegisters() == 11);

  auto & lambda_three = ptg->GetLambdaNode(test.GetLambdaThree());
  auto & lambda_three_out = ptg->GetRegisterNode(*test.GetLambdaThree().output());

  auto & lambda_four = ptg->GetLambdaNode(test.GetLambdaFour());
  auto & lambda_four_out = ptg->GetRegisterNode(*test.GetLambdaFour().output());

  auto & lambda_indcall = ptg->GetLambdaNode(test.GetLambdaIndcall());
  auto & lambda_indcall_out = ptg->GetRegisterNode(*test.GetLambdaIndcall().output());
  auto & lambda_indcall_arg =
      ptg->GetRegisterNode(*test.GetLambdaIndcall().GetFunctionArguments()[0]);

  auto & lambda_test = ptg->GetLambdaNode(test.GetLambdaTest());
  auto & lambda_test_out = ptg->GetRegisterNode(*test.GetLambdaTest().output());
  auto & lambda_test_cv0 = ptg->GetRegisterNode(*test.GetLambdaTest().GetContextVars()[0].inner);
  auto & lambda_test_cv1 = ptg->GetRegisterNode(*test.GetLambdaTest().GetContextVars()[1].inner);
  auto & lambda_test_cv2 = ptg->GetRegisterNode(*test.GetLambdaTest().GetContextVars()[2].inner);

  assert(TargetsExactly(lambda_three_out, { &lambda_three }));

  assert(TargetsExactly(lambda_four_out, { &lambda_four }));

  assert(TargetsExactly(lambda_indcall_out, { &lambda_indcall }));
  assert(TargetsExactly(lambda_indcall_arg, { &lambda_three, &lambda_four }));

  assert(TargetsExactly(lambda_test_out, { &lambda_test }));
  assert(TargetsExactly(lambda_test_cv0, { &lambda_indcall }));
  assert(TargetsExactly(lambda_test_cv1, { &lambda_four }));
  assert(TargetsExactly(lambda_test_cv2, { &lambda_three }));

  assert(EscapedIsExactly(*ptg, { &lambda_test }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestIndirectCall1",
    TestIndirectCall1)

static int
TestIndirectCall2()
{
  jlm::tests::IndirectCallTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 3);
  assert(ptg->NumLambdaNodes() == 7);
  assert(ptg->NumDeltaNodes() == 2);
  assert(ptg->NumMappedRegisters() == 27);

  auto & lambdaThree = ptg->GetLambdaNode(test.GetLambdaThree());
  auto & lambdaThreeOutput = ptg->GetRegisterNode(*test.GetLambdaThree().output());

  auto & lambdaFour = ptg->GetLambdaNode(test.GetLambdaFour());
  auto & lambdaFourOutput = ptg->GetRegisterNode(*test.GetLambdaFour().output());

  assert(TargetsExactly(lambdaThreeOutput, { &lambdaThree }));
  assert(TargetsExactly(lambdaFourOutput, { &lambdaFour }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestIndirectCall2",
    TestIndirectCall2)

static int
TestExternalCall1()
{
  jlm::tests::ExternalCallTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 2);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumImportNodes() == 1);
  assert(ptg->NumMappedRegisters() == 10);

  auto & lambdaF = ptg->GetLambdaNode(test.LambdaF());
  auto & lambdaFArgument0 = ptg->GetRegisterNode(*test.LambdaF().GetFunctionArguments()[0]);
  auto & lambdaFArgument1 = ptg->GetRegisterNode(*test.LambdaF().GetFunctionArguments()[1]);
  auto & importG = ptg->GetImportNode(test.ExternalGArgument());

  auto & callResult = ptg->GetRegisterNode(*test.CallG().Result(0));

  auto & externalMemory = ptg->GetExternalMemoryNode();

  assert(TargetsExactly(lambdaFArgument0, { &lambdaF, &importG, &externalMemory }));
  assert(TargetsExactly(lambdaFArgument1, { &lambdaF, &importG, &externalMemory }));
  assert(TargetsExactly(callResult, { &lambdaF, &importG, &externalMemory }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestExternalCall1",
    TestExternalCall1)

static int
TestGamma()
{
  jlm::tests::GammaTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 15);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);

  for (size_t n = 1; n < 5; n++)
  {
    auto & lambdaArgument = ptg->GetRegisterNode(*test.lambda->GetFunctionArguments()[n]);
    assert(TargetsExactly(lambdaArgument, { &lambda, &ptg->GetExternalMemoryNode() }));
  }

  for (size_t n = 0; n < 4; n++)
  {
    auto entryvar = test.gamma->GetEntryVar(n);
    auto & argument0 = ptg->GetRegisterNode(*entryvar.branchArgument[0]);
    auto & argument1 = ptg->GetRegisterNode(*entryvar.branchArgument[1]);

    assert(TargetsExactly(argument0, { &lambda, &ptg->GetExternalMemoryNode() }));
    assert(TargetsExactly(argument1, { &lambda, &ptg->GetExternalMemoryNode() }));
  }

  for (size_t n = 0; n < 4; n++)
  {
    auto & gammaOutput = ptg->GetRegisterNode(*test.gamma->GetExitVars()[0].output);
    assert(TargetsExactly(gammaOutput, { &lambda, &ptg->GetExternalMemoryNode() }));
  }

  assert(EscapedIsExactly(*ptg, { &lambda }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestGamma", TestGamma)

static int
TestTheta()
{
  jlm::tests::ThetaTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 5);

  auto & lambda = ptg->GetLambdaNode(*test.lambda);
  auto & lambdaArgument1 = ptg->GetRegisterNode(*test.lambda->GetFunctionArguments()[1]);
  auto & lambdaOutput = ptg->GetRegisterNode(*test.lambda->output());

  auto & gepOutput = ptg->GetRegisterNode(*test.gep->output(0));

  auto & thetaArgument2 = ptg->GetRegisterNode(*test.theta->GetLoopVars()[2].pre);
  auto & thetaOutput2 = ptg->GetRegisterNode(*test.theta->output(2));

  assert(TargetsExactly(lambdaArgument1, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(lambdaOutput, { &lambda }));

  assert(TargetsExactly(gepOutput, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(TargetsExactly(thetaArgument2, { &lambda, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(thetaOutput2, { &lambda, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambda }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestTheta", TestTheta)

static int
TestDelta1()
{
  jlm::tests::DeltaTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 1);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumMappedRegisters() == 6);

  auto & delta_f = ptg->GetDeltaNode(*test.delta_f);
  auto & pdelta_f = ptg->GetRegisterNode(*test.delta_f->output());

  auto & lambda_g = ptg->GetLambdaNode(*test.lambda_g);
  auto & plambda_g = ptg->GetRegisterNode(*test.lambda_g->output());
  auto & lambda_g_arg0 = ptg->GetRegisterNode(*test.lambda_g->GetFunctionArguments()[0]);

  auto & lambda_h = ptg->GetLambdaNode(*test.lambda_h);
  auto & plambda_h = ptg->GetRegisterNode(*test.lambda_h->output());
  auto & lambda_h_cv0 = ptg->GetRegisterNode(*test.lambda_h->GetContextVars()[0].inner);
  auto & lambda_h_cv1 = ptg->GetRegisterNode(*test.lambda_h->GetContextVars()[1].inner);

  assert(TargetsExactly(pdelta_f, { &delta_f }));

  assert(TargetsExactly(plambda_g, { &lambda_g }));
  assert(TargetsExactly(plambda_h, { &lambda_h }));

  assert(TargetsExactly(lambda_g_arg0, { &delta_f }));

  assert(TargetsExactly(lambda_h_cv0, { &delta_f }));
  assert(TargetsExactly(lambda_h_cv1, { &lambda_g }));

  assert(EscapedIsExactly(*ptg, { &lambda_h }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestDelta1", TestDelta1)

static int
TestDelta2()
{
  jlm::tests::DeltaTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 2);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumMappedRegisters() == 8);

  auto & delta_d1 = ptg->GetDeltaNode(*test.delta_d1);
  auto & delta_d1_out = ptg->GetRegisterNode(*test.delta_d1->output());

  auto & delta_d2 = ptg->GetDeltaNode(*test.delta_d2);
  auto & delta_d2_out = ptg->GetRegisterNode(*test.delta_d2->output());

  auto & lambda_f1 = ptg->GetLambdaNode(*test.lambda_f1);
  auto & lambda_f1_out = ptg->GetRegisterNode(*test.lambda_f1->output());
  auto & lambda_f1_cvd1 = ptg->GetRegisterNode(*test.lambda_f1->GetContextVars()[0].inner);

  auto & lambda_f2 = ptg->GetLambdaNode(*test.lambda_f2);
  auto & lambda_f2_out = ptg->GetRegisterNode(*test.lambda_f2->output());
  auto & lambda_f2_cvd1 = ptg->GetRegisterNode(*test.lambda_f2->GetContextVars()[0].inner);
  auto & lambda_f2_cvd2 = ptg->GetRegisterNode(*test.lambda_f2->GetContextVars()[1].inner);
  auto & lambda_f2_cvf1 = ptg->GetRegisterNode(*test.lambda_f2->GetContextVars()[2].inner);

  assert(TargetsExactly(delta_d1_out, { &delta_d1 }));
  assert(TargetsExactly(delta_d2_out, { &delta_d2 }));

  assert(TargetsExactly(lambda_f1_out, { &lambda_f1 }));
  assert(TargetsExactly(lambda_f2_out, { &lambda_f2 }));

  assert(&lambda_f1_cvd1 == &delta_d1_out);
  assert(&lambda_f2_cvd1 == &delta_d1_out);
  assert(&lambda_f2_cvd2 == &delta_d2_out);
  assert(&lambda_f2_cvf1 == &lambda_f1_out);

  assert(EscapedIsExactly(*ptg, { &lambda_f2 }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestDelta2", TestDelta2)

static int
TestImports()
{
  jlm::tests::ImportTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumImportNodes() == 2);
  assert(ptg->NumMappedRegisters() == 8);

  auto & d1 = ptg->GetImportNode(*test.import_d1);
  auto & import_d1 = ptg->GetRegisterNode(*test.import_d1);

  auto & d2 = ptg->GetImportNode(*test.import_d2);
  auto & import_d2 = ptg->GetRegisterNode(*test.import_d2);

  auto & lambda_f1 = ptg->GetLambdaNode(*test.lambda_f1);
  auto & lambda_f1_out = ptg->GetRegisterNode(*test.lambda_f1->output());
  auto & lambda_f1_cvd1 = ptg->GetRegisterNode(*test.lambda_f1->GetContextVars()[0].inner);

  auto & lambda_f2 = ptg->GetLambdaNode(*test.lambda_f2);
  auto & lambda_f2_out = ptg->GetRegisterNode(*test.lambda_f2->output());
  auto & lambda_f2_cvd1 = ptg->GetRegisterNode(*test.lambda_f2->GetContextVars()[0].inner);
  auto & lambda_f2_cvd2 = ptg->GetRegisterNode(*test.lambda_f2->GetContextVars()[1].inner);
  auto & lambda_f2_cvf1 = ptg->GetRegisterNode(*test.lambda_f2->GetContextVars()[2].inner);

  assert(TargetsExactly(import_d1, { &d1 }));
  assert(TargetsExactly(import_d2, { &d2 }));

  assert(TargetsExactly(lambda_f1_out, { &lambda_f1 }));
  assert(TargetsExactly(lambda_f2_out, { &lambda_f2 }));

  assert(&lambda_f1_cvd1 == &import_d1);
  assert(&lambda_f2_cvd1 == &import_d1);
  assert(&lambda_f2_cvd2 == &import_d2);
  assert(&lambda_f2_cvf1 == &lambda_f1_out);

  assert(EscapedIsExactly(*ptg, { &lambda_f2, &d1, &d2 }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestImports", TestImports)

static int
TestPhi1()
{
  jlm::tests::PhiTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumAllocaNodes() == 1);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumMappedRegisters() == 16);

  auto & lambda_fib = ptg->GetLambdaNode(*test.lambda_fib);
  auto & lambda_fib_out = ptg->GetRegisterNode(*test.lambda_fib->output());
  auto & lambda_fib_arg1 = ptg->GetRegisterNode(*test.lambda_fib->GetFunctionArguments()[1]);

  auto & lambda_test = ptg->GetLambdaNode(*test.lambda_test);
  auto & lambda_test_out = ptg->GetRegisterNode(*test.lambda_test->output());

  auto & phi_rv = ptg->GetRegisterNode(*test.phi->begin_rv().output());
  auto & phi_rv_arg = ptg->GetRegisterNode(*test.phi->begin_rv().output()->argument());

  auto & gamma_result = ptg->GetRegisterNode(*test.gamma->subregion(0)->argument(1));
  auto & gamma_fib = ptg->GetRegisterNode(*test.gamma->subregion(0)->argument(2));

  auto & alloca = ptg->GetAllocaNode(*test.alloca);
  auto & alloca_out = ptg->GetRegisterNode(*test.alloca->output(0));

  assert(TargetsExactly(lambda_fib_out, { &lambda_fib }));
  assert(TargetsExactly(lambda_fib_arg1, { &alloca }));

  assert(TargetsExactly(lambda_test_out, { &lambda_test }));

  assert(&phi_rv == &lambda_fib_out);
  assert(TargetsExactly(phi_rv_arg, { &lambda_fib }));

  assert(TargetsExactly(gamma_result, { &alloca }));
  assert(TargetsExactly(gamma_fib, { &lambda_fib }));

  assert(TargetsExactly(alloca_out, { &alloca }));

  assert(EscapedIsExactly(*ptg, { &lambda_test }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestPhi1", TestPhi1)

static int
TestExternalMemory()
{
  jlm::tests::ExternalMemoryTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 3);

  auto & lambdaF = ptg->GetLambdaNode(*test.LambdaF);
  auto & lambdaFArgument0 = ptg->GetRegisterNode(*test.LambdaF->GetFunctionArguments()[0]);
  auto & lambdaFArgument1 = ptg->GetRegisterNode(*test.LambdaF->GetFunctionArguments()[1]);

  assert(TargetsExactly(lambdaFArgument0, { &lambdaF, &ptg->GetExternalMemoryNode() }));
  assert(TargetsExactly(lambdaFArgument1, { &lambdaF, &ptg->GetExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { &lambdaF }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestExternalMemory",
    TestExternalMemory)

static int
TestEscapedMemory1()
{
  jlm::tests::EscapedMemoryTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 4);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 10);

  auto & lambdaTestArgument0 = ptg->GetRegisterNode(*test.LambdaTest->GetFunctionArguments()[0]);
  auto & lambdaTestCv0 = ptg->GetRegisterNode(*test.LambdaTest->GetContextVars()[0].inner);
  auto & loadNode1Output = ptg->GetRegisterNode(*test.LoadNode1->output(0));

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

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestEscapedMemory1",
    TestEscapedMemory1)

static int
TestEscapedMemory2()
{
  jlm::tests::EscapedMemoryTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumImportNodes() == 2);
  assert(ptg->NumLambdaNodes() == 3);
  assert(ptg->NumMallocNodes() == 2);
  assert(ptg->NumMappedRegisters() == 10);

  auto returnAddressFunction = &ptg->GetLambdaNode(*test.ReturnAddressFunction);
  auto callExternalFunction1 = &ptg->GetLambdaNode(*test.CallExternalFunction1);
  auto callExternalFunction2 = &ptg->GetLambdaNode(*test.CallExternalFunction2);
  auto returnAddressMalloc = &ptg->GetMallocNode(*test.ReturnAddressMalloc);
  auto callExternalFunction1Malloc = &ptg->GetMallocNode(*test.CallExternalFunction1Malloc);
  auto externalMemory = &ptg->GetExternalMemoryNode();
  auto externalFunction1Import = &ptg->GetImportNode(*test.ExternalFunction1Import);
  auto externalFunction2Import = &ptg->GetImportNode(*test.ExternalFunction2Import);

  auto & externalFunction2CallResult = ptg->GetRegisterNode(*test.ExternalFunction2Call->Result(0));

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

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestEscapedMemory2",
    TestEscapedMemory2)

static int
TestEscapedMemory3()
{
  jlm::tests::EscapedMemoryTest3 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 1);
  assert(ptg->NumImportNodes() == 1);
  assert(ptg->NumLambdaNodes() == 1);
  assert(ptg->NumMappedRegisters() == 5);

  auto lambdaTest = &ptg->GetLambdaNode(*test.LambdaTest);
  auto deltaGlobal = &ptg->GetDeltaNode(*test.DeltaGlobal);
  auto importExternalFunction = &ptg->GetImportNode(*test.ImportExternalFunction);
  auto externalMemory = &ptg->GetExternalMemoryNode();

  auto & callExternalFunctionResult = ptg->GetRegisterNode(*test.CallExternalFunction->Result(0));

  assert(TargetsExactly(
      callExternalFunctionResult,
      { lambdaTest, deltaGlobal, importExternalFunction, externalMemory }));

  assert(EscapedIsExactly(*ptg, { lambdaTest, deltaGlobal, importExternalFunction }));

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestEscapedMemory3",
    TestEscapedMemory3)

static int
TestMemcpy()
{
  jlm::tests::MemcpyTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->NumDeltaNodes() == 2);
  assert(ptg->NumLambdaNodes() == 2);
  assert(ptg->NumMappedRegisters() == 11);

  auto localArray = &ptg->GetDeltaNode(test.LocalArray());
  auto globalArray = &ptg->GetDeltaNode(test.GlobalArray());

  auto & memCpyDest = ptg->GetRegisterNode(*test.Memcpy().input(0)->origin());
  auto & memCpySrc = ptg->GetRegisterNode(*test.Memcpy().input(1)->origin());

  auto lambdaF = &ptg->GetLambdaNode(test.LambdaF());
  auto lambdaG = &ptg->GetLambdaNode(test.LambdaG());

  assert(TargetsExactly(memCpyDest, { globalArray }));
  assert(TargetsExactly(memCpySrc, { localArray }));

  assert(EscapedIsExactly(*ptg, { globalArray, localArray, lambdaF, lambdaG }));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestMemcpy", TestMemcpy)

static int
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

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestLinkedList", TestLinkedList)

static int
TestStatistics()
{
  // Arrange
  jlm::tests::LoadTest1 test;
  jlm::util::StatisticsCollectorSettings statisticsCollectorSettings(
      { jlm::util::Statistics::Id::AndersenAnalysis });
  jlm::util::StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  // Act
  jlm::llvm::aa::Andersen andersen;
  auto ptg = andersen.Analyze(test.module(), statisticsCollector);

  // Assert
  assert(statisticsCollector.NumCollectedStatistics() == 1);
  const auto & statistics = *statisticsCollector.CollectedStatistics().begin();

  assert(statistics.GetMeasurementValue<uint64_t>("#StoreConstraints") == 0);
  assert(statistics.GetMeasurementValue<uint64_t>("#LoadConstraints") == 1);
  assert(statistics.GetMeasurementValue<uint64_t>("#PointsToGraphNodes") == ptg->NumNodes());
  assert(statistics.GetTimerElapsedNanoseconds("AnalysisTimer") > 0);

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestStatistics", TestStatistics)

static int
TestConfiguration()
{
  using namespace jlm::llvm::aa;
  auto config = Andersen::Configuration::NaiveSolverConfiguration();

  // Arrange
  config.EnableOfflineVariableSubstitution(false);
  config.EnableOfflineConstraintNormalization(true);
  config.SetSolver(Andersen::Configuration::Solver::Naive);

  // Act
  auto configString = config.ToString();

  // Assert
  assert(!config.IsOfflineVariableSubstitutionEnabled());
  assert(config.IsOfflineConstraintNormalizationEnabled());
  assert(config.GetSolver() == Andersen::Configuration::Solver::Naive);
  assert(configString.find("OVS") == std::string::npos);
  assert(configString.find("NORM") != std::string::npos);
  assert(configString.find("Solver=Naive") != std::string::npos);

  // Arrange some more
  auto policy = PointerObjectConstraintSet::WorklistSolverPolicy::TwoPhaseLeastRecentlyFired;
  config.SetSolver(Andersen::Configuration::Solver::Worklist);
  config.SetWorklistSolverPolicy(policy);
  config.EnableOfflineVariableSubstitution(false);
  config.EnableOnlineCycleDetection(true);

  // Act
  configString = config.ToString();

  // Assert
  assert(configString.find("Solver=Worklist") != std::string::npos);
  assert(configString.find("Policy=TwoPhaseLeastRecentlyFired") != std::string::npos);
  assert(configString.find("OVS") == std::string::npos);
  assert(configString.find("OnlineCD") != std::string::npos);

  // Arrange some more
  config.EnableOnlineCycleDetection(false);
  config.EnableHybridCycleDetection(true);

  // Act
  configString = config.ToString();

  // Assert
  assert(configString.find("OnlineCD") == std::string::npos);
  assert(configString.find("HybridCD") != std::string::npos);

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestConfiguration",
    TestConfiguration)

static int
TestConstructPointsToGraph()
{
  using namespace jlm::llvm::aa;

  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  // Arrange a very standard set of memory objects and registers
  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), true);
  auto allocaR = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput());
  auto importR = set.CreateRegisterPointerObject(rvsdg.GetImportOutput());
  auto lambda0 = set.CreateFunctionMemoryObject(rvsdg.GetLambdaNode());
  auto lambdaR = set.CreateRegisterPointerObject(rvsdg.GetLambdaOutput());
  auto malloc0 = set.CreateMallocMemoryObject(rvsdg.GetMallocNode(), true);
  auto mallocR = set.CreateRegisterPointerObject(rvsdg.GetMallocOutput());
  set.AddToPointsToSet(allocaR, alloca0);
  set.AddToPointsToSet(importR, import0);
  set.AddToPointsToSet(lambdaR, lambda0);
  set.AddToPointsToSet(mallocR, malloc0);

  // Make an exception for the delta node: Map its output to importR's PointerObject instead
  [[maybe_unused]] auto delta0 = set.CreateGlobalMemoryObject(rvsdg.GetDeltaNode(), true);
  set.MapRegisterToExistingPointerObject(rvsdg.GetDeltaOutput(), importR);

  // Make alloca0 point to lambda0
  set.AddToPointsToSet(alloca0, lambda0);

  // create a dummy node
  auto dummy = set.CreateDummyRegisterPointerObject();

  // Unify allocaR with dummy, and importR with dummy
  set.UnifyPointerObjects(dummy, allocaR);
  set.UnifyPointerObjects(dummy, importR);

  // Unify a register and a memory object
  set.UnifyPointerObjects(lambdaR, malloc0);

  // Mark a register as pointing to external
  set.MarkAsPointingToExternal(mallocR);
  // And a memory object as escaped
  set.MarkAsEscaped(delta0);

  auto ptg = Andersen::ConstructPointsToGraphFromPointerObjectSet(set);

  // Assert
  auto & allocaNode = ptg->GetAllocaNode(rvsdg.GetAllocaNode());
  auto & allocaRNode = ptg->GetRegisterNode(rvsdg.GetAllocaOutput());
  auto & importNode = ptg->GetImportNode(rvsdg.GetImportOutput());
  auto & importRNode = ptg->GetRegisterNode(rvsdg.GetImportOutput());
  auto & lambdaNode = ptg->GetLambdaNode(rvsdg.GetLambdaNode());
  auto & lambdaRNode = ptg->GetRegisterNode(rvsdg.GetLambdaOutput());
  auto & mallocNode = ptg->GetMallocNode(rvsdg.GetMallocNode());
  auto & mallocRNode = ptg->GetRegisterNode(rvsdg.GetMallocOutput());
  auto & deltaNode = ptg->GetDeltaNode(rvsdg.GetDeltaNode());
  auto & deltaRNode = ptg->GetRegisterNode(rvsdg.GetDeltaOutput());

  // Make sure unification causes allocaR to point to all of its pointees
  assert(TargetsExactly(allocaRNode, { &allocaNode, &importNode }));
  // The unified registers should share RegisterNode
  assert(&allocaRNode == &importRNode);
  // deltaR was mapped to importR, so it too should share RegisterNode
  assert(&allocaRNode == &deltaRNode);

  // alloca0 -> lambda0
  assert(TargetsExactly(allocaNode, { &lambdaNode }));

  // Unifying a register with a non-register does not affect it
  assert(lambdaRNode.GetOutputs().Size() == 1);
  // But it does share pointees with the other nodes
  assert(TargetsExactly(mallocNode, { &lambdaNode }));

  // deltaNode has escaped, and should be pointed to by mallocR and itself
  assert(deltaNode.NumSources() == 2);

  auto & externalMemory = ptg->GetExternalMemoryNode();
  // deltaNode points to everything that has escaped
  assert(TargetsExactly(deltaNode, { &deltaNode, &importNode, &externalMemory }));
  // importNode points to nothing, as it is not marked "CanPoint"
  assert(TargetsExactly(importNode, {}));
  // mallocR points to mallocNode, as well as everything that has escaped
  assert(TargetsExactly(mallocRNode, { &mallocNode, &deltaNode, &importNode, &externalMemory }));

  // Adding up the out-edges for all nodes
  auto [_, numPointsToRelations] = ptg->NumEdges();
  assert(numPointsToRelations == 2 * 3 + 1 + 1 + 1 + 3 + 4);

  return 0;
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestConstructPointsToGraph",
    TestConstructPointsToGraph)
