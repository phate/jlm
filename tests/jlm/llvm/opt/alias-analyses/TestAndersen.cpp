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
 * @param pointsToGraph
 * @param ptgNode the source node
 * @param expectedTargets a set of nodes that \p node should point to.
 * @return false if the check fails, true otherwise
 */
[[nodiscard]] static bool
TargetsExactly(
    const jlm::llvm::aa::PointsToGraph & pointsToGraph,
    const jlm::llvm::aa::PointsToGraph::NodeIndex ptgNode,
    const std::unordered_set<jlm::llvm::aa::PointsToGraph::NodeIndex> & expectedTargets)
{
  using namespace jlm::llvm::aa;

  std::unordered_set<PointsToGraph::NodeIndex> actualTargets;
  for (const auto target : pointsToGraph.getExplicitTargets(ptgNode).Items())
    actualTargets.insert(target);

  // If the node targets all externally available memory, add those nodes
  if (pointsToGraph.isTargetingAllExternallyAvailable(ptgNode))
    for (const auto target : pointsToGraph.getExternallyAvailableNodes())
      actualTargets.insert(target);

  return actualTargets == expectedTargets;
}

/**
 * @brief Checks that the set of Memory Nodes escaping the PointsToGraph is exactly equal
 * to the given set of nodes. The external node is included implicitly if omitted.
 * @param pointsToGraph the PointsToGraph
 * @param nodes the complete set of nodes that should have escaped
 * @return true if the \p pointsToGraph's escaped set is identical to \p nodes, false otherwise.
 */
[[nodiscard]] static bool
EscapedIsExactly(
    const jlm::llvm::aa::PointsToGraph & pointsToGraph,
    const std::unordered_set<jlm::llvm::aa::PointsToGraph::NodeIndex> & nodes)
{
  jlm::util::HashSet expected(nodes);
  expected.insert(pointsToGraph.getExternalMemoryNode());

  jlm::util::HashSet<jlm::llvm::aa::PointsToGraph::NodeIndex> actual;
  for (const auto escaped : pointsToGraph.getExternallyAvailableNodes())
    actual.insert(escaped);

  return expected == actual;
}

static void
TestStore1()
{
  jlm::tests::StoreTest1 test;
  const auto ptg = RunAndersen(test.module());

  // std::unordered_map<const jlm::rvsdg::output*, std::string> outputMap;
  // std::cout << jlm::rvsdg::view(test.graph().GetRootRegion(), outputMap) << std::endl;
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*ptg, outputMap) << std::endl;

  assert(ptg->numAllocaNodes() == 4);
  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 5);

  auto alloca_a = ptg->getNodeForAlloca(*test.alloca_a);
  auto alloca_b = ptg->getNodeForAlloca(*test.alloca_b);
  auto alloca_c = ptg->getNodeForAlloca(*test.alloca_c);
  auto alloca_d = ptg->getNodeForAlloca(*test.alloca_d);

  auto palloca_a = ptg->getNodeForRegister(*test.alloca_a->output(0));
  auto palloca_b = ptg->getNodeForRegister(*test.alloca_b->output(0));
  auto palloca_c = ptg->getNodeForRegister(*test.alloca_c->output(0));
  auto palloca_d = ptg->getNodeForRegister(*test.alloca_d->output(0));

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto plambda = ptg->getNodeForRegister(*test.lambda->output());

  assert(TargetsExactly(*ptg, alloca_a, { alloca_b }));
  assert(TargetsExactly(*ptg, alloca_b, { alloca_c }));
  assert(TargetsExactly(*ptg, alloca_c, { alloca_d }));
  assert(TargetsExactly(*ptg, alloca_d, {}));

  assert(TargetsExactly(*ptg, palloca_a, { alloca_a }));
  assert(TargetsExactly(*ptg, palloca_b, { alloca_b }));
  assert(TargetsExactly(*ptg, palloca_c, { alloca_c }));
  assert(TargetsExactly(*ptg, palloca_d, { alloca_d }));

  assert(TargetsExactly(*ptg, lambda, {}));
  assert(TargetsExactly(*ptg, plambda, { lambda }));

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestStore1", TestStore1)

static void
TestStore2()
{
  jlm::tests::StoreTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numAllocaNodes() == 5);
  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 6);

  auto alloca_a = ptg->getNodeForAlloca(*test.alloca_a);
  auto alloca_b = ptg->getNodeForAlloca(*test.alloca_b);
  auto alloca_x = ptg->getNodeForAlloca(*test.alloca_x);
  auto alloca_y = ptg->getNodeForAlloca(*test.alloca_y);
  auto alloca_p = ptg->getNodeForAlloca(*test.alloca_p);

  auto palloca_a = ptg->getNodeForRegister(*test.alloca_a->output(0));
  auto palloca_b = ptg->getNodeForRegister(*test.alloca_b->output(0));
  auto palloca_x = ptg->getNodeForRegister(*test.alloca_x->output(0));
  auto palloca_y = ptg->getNodeForRegister(*test.alloca_y->output(0));
  auto palloca_p = ptg->getNodeForRegister(*test.alloca_p->output(0));

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto plambda = ptg->getNodeForRegister(*test.lambda->output());

  assert(TargetsExactly(*ptg, alloca_a, {}));
  assert(TargetsExactly(*ptg, alloca_b, {}));
  assert(TargetsExactly(*ptg, alloca_x, { alloca_a }));
  assert(TargetsExactly(*ptg, alloca_y, { alloca_b }));
  assert(TargetsExactly(*ptg, alloca_p, { alloca_x, alloca_y }));

  assert(TargetsExactly(*ptg, palloca_a, { alloca_a }));
  assert(TargetsExactly(*ptg, palloca_b, { alloca_b }));
  assert(TargetsExactly(*ptg, palloca_x, { alloca_x }));
  assert(TargetsExactly(*ptg, palloca_y, { alloca_y }));
  assert(TargetsExactly(*ptg, palloca_p, { alloca_p }));

  assert(TargetsExactly(*ptg, lambda, {}));
  assert(TargetsExactly(*ptg, plambda, { lambda }));

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestStore2", TestStore2)

static void
TestLoad1()
{
  jlm::tests::LoadTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 3);

  auto loadResult = ptg->getNodeForRegister(*test.load_p->output(0));

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaOutput = ptg->getNodeForRegister(*test.lambda->output());
  auto lambdaArgument0 = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[0]);

  assert(TargetsExactly(*ptg, loadResult, { lambda, ptg->getExternalMemoryNode() }));

  assert(TargetsExactly(*ptg, lambdaOutput, { lambda }));
  assert(TargetsExactly(*ptg, lambdaArgument0, { lambda, ptg->getExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestLoad1", TestLoad1)

static void
TestLoad2()
{
  jlm::tests::LoadTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numAllocaNodes() == 5);
  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 8);

  auto alloca_a = ptg->getNodeForAlloca(*test.alloca_a);
  auto alloca_b = ptg->getNodeForAlloca(*test.alloca_b);
  auto alloca_x = ptg->getNodeForAlloca(*test.alloca_x);
  auto alloca_y = ptg->getNodeForAlloca(*test.alloca_y);
  auto alloca_p = ptg->getNodeForAlloca(*test.alloca_p);

  auto pload_x = ptg->getNodeForRegister(*test.load_x->output(0));
  auto pload_a = ptg->getNodeForRegister(*test.load_a->output(0));

  auto lambdaMemoryNode = ptg->getNodeForLambda(*test.lambda);

  assert(TargetsExactly(*ptg, alloca_x, { alloca_a }));
  assert(TargetsExactly(*ptg, alloca_y, { alloca_a, alloca_b }));
  assert(TargetsExactly(*ptg, alloca_p, { alloca_x }));

  assert(TargetsExactly(*ptg, pload_x, { alloca_x }));
  assert(TargetsExactly(*ptg, pload_a, { alloca_a }));

  assert(EscapedIsExactly(*ptg, { lambdaMemoryNode }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestLoad2", TestLoad2)

static void
TestLoadFromUndef()
{
  jlm::tests::LoadFromUndefTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 2);

  auto lambdaMemoryNode = ptg->getNodeForLambda(test.Lambda());
  auto undefValueNode = ptg->getNodeForRegister(*test.UndefValueNode()->output(0));

  assert(TargetsExactly(*ptg, undefValueNode, {}));
  assert(EscapedIsExactly(*ptg, { lambdaMemoryNode }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestLoadFromUndef",
    TestLoadFromUndef)

static void
TestGetElementPtr()
{
  jlm::tests::GetElementPtrTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 4);

  // We only care about the getelemenptr's in this test, skipping the validation for all other nodes
  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto gepX = ptg->getNodeForRegister(*test.getElementPtrX->output(0));
  auto gepY = ptg->getNodeForRegister(*test.getElementPtrY->output(0));

  // The RegisterNode is the same
  assert(gepX == gepY);

  assert(TargetsExactly(*ptg, gepX, { lambda, ptg->getExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestGetElementPtr",
    TestGetElementPtr)

static void
TestBitCast()
{
  jlm::tests::BitCastTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 3);

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaOut = ptg->getNodeForRegister(*test.lambda->output());
  auto lambdaArg = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[0]);
  auto bitCast = ptg->getNodeForRegister(*test.bitCast->output(0));

  assert(TargetsExactly(*ptg, lambdaOut, { lambda }));
  assert(TargetsExactly(*ptg, lambdaArg, { lambda, ptg->getExternalMemoryNode() }));
  assert(TargetsExactly(*ptg, bitCast, { lambda, ptg->getExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestBitCast", TestBitCast)

static void
TestConstantPointerNull()
{
  jlm::tests::ConstantPointerNullTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 3);

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaOut = ptg->getNodeForRegister(*test.lambda->output());
  auto lambdaArg = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[0]);

  auto constantPointerNull = ptg->getNodeForRegister(*test.constantPointerNullNode->output(0));

  assert(TargetsExactly(*ptg, lambdaOut, { lambda }));
  assert(TargetsExactly(*ptg, lambdaArg, { lambda, ptg->getExternalMemoryNode() }));
  assert(TargetsExactly(*ptg, constantPointerNull, {}));

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestConstantPointerNull",
    TestConstantPointerNull)

static void
TestBits2Ptr()
{
  jlm::tests::Bits2PtrTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 2);
  assert(ptg->numMappedRegisters() == 5);

  auto lambdaTestMemoryNode = ptg->getNodeForLambda(test.GetLambdaTest());
  auto externalMemoryNode = ptg->getExternalMemoryNode();

  auto callOutput0 = ptg->getNodeForRegister(*test.GetCallNode().output(0));
  auto bits2ptr = ptg->getNodeForRegister(*test.GetBitsToPtrNode().output(0));

  assert(TargetsExactly(*ptg, callOutput0, { lambdaTestMemoryNode, externalMemoryNode }));
  assert(TargetsExactly(*ptg, bits2ptr, { lambdaTestMemoryNode, externalMemoryNode }));

  assert(EscapedIsExactly(*ptg, { lambdaTestMemoryNode }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestBits2Ptr", TestBits2Ptr)

static void
TestCall1()
{
  jlm::tests::CallTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numAllocaNodes() == 3);
  assert(ptg->numLambdaNodes() == 3);
  assert(ptg->numMappedRegisters() == 12);

  auto alloca_x = ptg->getNodeForAlloca(*test.alloca_x);
  auto alloca_y = ptg->getNodeForAlloca(*test.alloca_y);
  auto alloca_z = ptg->getNodeForAlloca(*test.alloca_z);

  auto palloca_x = ptg->getNodeForRegister(*test.alloca_x->output(0));
  auto palloca_y = ptg->getNodeForRegister(*test.alloca_y->output(0));
  auto palloca_z = ptg->getNodeForRegister(*test.alloca_z->output(0));

  auto lambda_f = ptg->getNodeForLambda(*test.lambda_f);
  auto lambda_g = ptg->getNodeForLambda(*test.lambda_g);
  auto lambda_h = ptg->getNodeForLambda(*test.lambda_h);

  auto plambda_f = ptg->getNodeForRegister(*test.lambda_f->output());
  auto plambda_g = ptg->getNodeForRegister(*test.lambda_g->output());
  auto plambda_h = ptg->getNodeForRegister(*test.lambda_h->output());

  auto lambda_f_arg0 = ptg->getNodeForRegister(*test.lambda_f->GetFunctionArguments()[0]);
  auto lambda_f_arg1 = ptg->getNodeForRegister(*test.lambda_f->GetFunctionArguments()[1]);

  auto lambda_g_arg0 = ptg->getNodeForRegister(*test.lambda_g->GetFunctionArguments()[0]);
  auto lambda_g_arg1 = ptg->getNodeForRegister(*test.lambda_g->GetFunctionArguments()[1]);

  auto lambda_h_cv0 = ptg->getNodeForRegister(*test.lambda_h->GetContextVars()[0].inner);
  auto lambda_h_cv1 = ptg->getNodeForRegister(*test.lambda_h->GetContextVars()[1].inner);

  assert(TargetsExactly(*ptg, palloca_x, { alloca_x }));
  assert(TargetsExactly(*ptg, palloca_y, { alloca_y }));
  assert(TargetsExactly(*ptg, palloca_z, { alloca_z }));

  assert(TargetsExactly(*ptg, plambda_f, { lambda_f }));
  assert(TargetsExactly(*ptg, plambda_g, { lambda_g }));
  assert(TargetsExactly(*ptg, plambda_h, { lambda_h }));

  assert(TargetsExactly(*ptg, lambda_f_arg0, { alloca_x }));
  assert(TargetsExactly(*ptg, lambda_f_arg1, { alloca_y }));

  assert(TargetsExactly(*ptg, lambda_g_arg0, { alloca_z }));
  assert(TargetsExactly(*ptg, lambda_g_arg1, { alloca_z }));

  assert(TargetsExactly(*ptg, lambda_h_cv0, { lambda_f }));
  assert(TargetsExactly(*ptg, lambda_h_cv1, { lambda_g }));

  assert(EscapedIsExactly(*ptg, { lambda_h }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestCall1", TestCall1)

static void
TestCall2()
{
  jlm::tests::CallTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 3);
  assert(ptg->numMallocNodes() == 1);
  assert(ptg->numImportNodes() == 0);
  assert(ptg->numMappedRegisters() == 11);

  auto lambda_create = ptg->getNodeForLambda(*test.lambda_create);
  auto lambda_create_out = ptg->getNodeForRegister(*test.lambda_create->output());

  auto lambda_destroy = ptg->getNodeForLambda(*test.lambda_destroy);
  auto lambda_destroy_out = ptg->getNodeForRegister(*test.lambda_destroy->output());
  auto lambda_destroy_arg =
      ptg->getNodeForRegister(*test.lambda_destroy->GetFunctionArguments()[0]);

  auto lambda_test = ptg->getNodeForLambda(*test.lambda_test);
  auto lambda_test_out = ptg->getNodeForRegister(*test.lambda_test->output());
  auto lambda_test_cv1 = ptg->getNodeForRegister(*test.lambda_test->GetContextVars()[0].inner);
  auto lambda_test_cv2 = ptg->getNodeForRegister(*test.lambda_test->GetContextVars()[1].inner);

  auto call_create1_out = ptg->getNodeForRegister(*test.CallCreate1().output(0));
  auto call_create2_out = ptg->getNodeForRegister(*test.CallCreate2().output(0));

  auto malloc = ptg->getNodeForMalloc(*test.malloc);
  auto malloc_out = ptg->getNodeForRegister(*test.malloc->output(0));

  assert(TargetsExactly(*ptg, lambda_create_out, { lambda_create }));

  assert(TargetsExactly(*ptg, lambda_destroy_out, { lambda_destroy }));
  assert(TargetsExactly(*ptg, lambda_destroy_arg, { malloc }));

  assert(TargetsExactly(*ptg, lambda_test_out, { lambda_test }));
  assert(TargetsExactly(*ptg, lambda_test_cv1, { lambda_create }));
  assert(TargetsExactly(*ptg, lambda_test_cv2, { lambda_destroy }));

  assert(TargetsExactly(*ptg, call_create1_out, { malloc }));
  assert(TargetsExactly(*ptg, call_create2_out, { malloc }));

  assert(TargetsExactly(*ptg, malloc_out, { malloc }));

  assert(EscapedIsExactly(*ptg, { lambda_test }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestCall2", TestCall2)

static void
TestIndirectCall1()
{
  jlm::tests::IndirectCallTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 4);
  assert(ptg->numImportNodes() == 0);
  assert(ptg->numMappedRegisters() == 11);

  auto lambda_three = ptg->getNodeForLambda(test.GetLambdaThree());
  auto lambda_three_out = ptg->getNodeForRegister(*test.GetLambdaThree().output());

  auto lambda_four = ptg->getNodeForLambda(test.GetLambdaFour());
  auto lambda_four_out = ptg->getNodeForRegister(*test.GetLambdaFour().output());

  auto lambda_indcall = ptg->getNodeForLambda(test.GetLambdaIndcall());
  auto lambda_indcall_out = ptg->getNodeForRegister(*test.GetLambdaIndcall().output());
  auto lambda_indcall_arg =
      ptg->getNodeForRegister(*test.GetLambdaIndcall().GetFunctionArguments()[0]);

  auto lambda_test = ptg->getNodeForLambda(test.GetLambdaTest());
  auto lambda_test_out = ptg->getNodeForRegister(*test.GetLambdaTest().output());
  auto lambda_test_cv0 = ptg->getNodeForRegister(*test.GetLambdaTest().GetContextVars()[0].inner);
  auto lambda_test_cv1 = ptg->getNodeForRegister(*test.GetLambdaTest().GetContextVars()[1].inner);
  auto lambda_test_cv2 = ptg->getNodeForRegister(*test.GetLambdaTest().GetContextVars()[2].inner);

  assert(TargetsExactly(*ptg, lambda_three_out, { lambda_three }));

  assert(TargetsExactly(*ptg, lambda_four_out, { lambda_four }));

  assert(TargetsExactly(*ptg, lambda_indcall_out, { lambda_indcall }));
  assert(TargetsExactly(*ptg, lambda_indcall_arg, { lambda_three, lambda_four }));

  assert(TargetsExactly(*ptg, lambda_test_out, { lambda_test }));
  assert(TargetsExactly(*ptg, lambda_test_cv0, { lambda_indcall }));
  assert(TargetsExactly(*ptg, lambda_test_cv1, { lambda_four }));
  assert(TargetsExactly(*ptg, lambda_test_cv2, { lambda_three }));

  assert(EscapedIsExactly(*ptg, { lambda_test }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestIndirectCall1",
    TestIndirectCall1)

static void
TestIndirectCall2()
{
  jlm::tests::IndirectCallTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numAllocaNodes() == 3);
  assert(ptg->numLambdaNodes() == 7);
  assert(ptg->numDeltaNodes() == 2);
  assert(ptg->numMappedRegisters() == 27);

  auto lambdaThree = ptg->getNodeForLambda(test.GetLambdaThree());
  auto lambdaThreeOutput = ptg->getNodeForRegister(*test.GetLambdaThree().output());

  auto lambdaFour = ptg->getNodeForLambda(test.GetLambdaFour());
  auto lambdaFourOutput = ptg->getNodeForRegister(*test.GetLambdaFour().output());

  assert(TargetsExactly(*ptg, lambdaThreeOutput, { lambdaThree }));
  assert(TargetsExactly(*ptg, lambdaFourOutput, { lambdaFour }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestIndirectCall2",
    TestIndirectCall2)

static void
TestExternalCall1()
{
  jlm::tests::ExternalCallTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numAllocaNodes() == 2);
  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numImportNodes() == 1);
  assert(ptg->numMappedRegisters() == 10);

  auto lambdaF = ptg->getNodeForLambda(test.LambdaF());
  auto lambdaFArgument0 = ptg->getNodeForRegister(*test.LambdaF().GetFunctionArguments()[0]);
  auto lambdaFArgument1 = ptg->getNodeForRegister(*test.LambdaF().GetFunctionArguments()[1]);
  auto importG = ptg->getNodeForImport(test.ExternalGArgument());

  auto callResult = ptg->getNodeForRegister(*test.CallG().output(0));

  auto externalMemory = ptg->getExternalMemoryNode();

  assert(TargetsExactly(*ptg, lambdaFArgument0, { lambdaF, importG, externalMemory }));
  assert(TargetsExactly(*ptg, lambdaFArgument1, { lambdaF, importG, externalMemory }));
  assert(TargetsExactly(*ptg, callResult, { lambdaF, importG, externalMemory }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestExternalCall1",
    TestExternalCall1)

static void
TestGamma()
{
  jlm::tests::GammaTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 15);

  auto lambda = ptg->getNodeForLambda(*test.lambda);

  for (size_t n = 1; n < 5; n++)
  {
    auto lambdaArgument = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[n]);
    assert(TargetsExactly(*ptg, lambdaArgument, { lambda, ptg->getExternalMemoryNode() }));
  }

  auto entryvars = test.gamma->GetEntryVars();
  assert(entryvars.size() == 4);
  for (const auto & entryvar : entryvars)
  {
    auto argument0 = ptg->getNodeForRegister(*entryvar.branchArgument[0]);
    auto argument1 = ptg->getNodeForRegister(*entryvar.branchArgument[1]);

    assert(TargetsExactly(*ptg, argument0, { lambda, ptg->getExternalMemoryNode() }));
    assert(TargetsExactly(*ptg, argument1, { lambda, ptg->getExternalMemoryNode() }));
  }

  for (size_t n = 0; n < 4; n++)
  {
    auto gammaOutput = ptg->getNodeForRegister(*test.gamma->GetExitVars()[0].output);
    assert(TargetsExactly(*ptg, gammaOutput, { lambda, ptg->getExternalMemoryNode() }));
  }

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestGamma", TestGamma)

static void
TestTheta()
{
  jlm::tests::ThetaTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 5);

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaArgument1 = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[1]);
  auto lambdaOutput = ptg->getNodeForRegister(*test.lambda->output());

  auto gepOutput = ptg->getNodeForRegister(*test.gep->output(0));

  auto thetaArgument2 = ptg->getNodeForRegister(*test.theta->GetLoopVars()[2].pre);
  auto thetaOutput2 = ptg->getNodeForRegister(*test.theta->output(2));

  assert(TargetsExactly(*ptg, lambdaArgument1, { lambda, ptg->getExternalMemoryNode() }));
  assert(TargetsExactly(*ptg, lambdaOutput, { lambda }));

  assert(TargetsExactly(*ptg, gepOutput, { lambda, ptg->getExternalMemoryNode() }));

  assert(TargetsExactly(*ptg, thetaArgument2, { lambda, ptg->getExternalMemoryNode() }));
  assert(TargetsExactly(*ptg, thetaOutput2, { lambda, ptg->getExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { lambda }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestTheta", TestTheta)

static void
TestDelta1()
{
  jlm::tests::DeltaTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numDeltaNodes() == 1);
  assert(ptg->numLambdaNodes() == 2);
  assert(ptg->numMappedRegisters() == 6);

  auto delta_f = ptg->getNodeForDelta(*test.delta_f);
  auto pdelta_f = ptg->getNodeForRegister(test.delta_f->output());

  auto lambda_g = ptg->getNodeForLambda(*test.lambda_g);
  auto plambda_g = ptg->getNodeForRegister(*test.lambda_g->output());
  auto lambda_g_arg0 = ptg->getNodeForRegister(*test.lambda_g->GetFunctionArguments()[0]);

  auto lambda_h = ptg->getNodeForLambda(*test.lambda_h);
  auto plambda_h = ptg->getNodeForRegister(*test.lambda_h->output());
  auto lambda_h_cv0 = ptg->getNodeForRegister(*test.lambda_h->GetContextVars()[0].inner);
  auto lambda_h_cv1 = ptg->getNodeForRegister(*test.lambda_h->GetContextVars()[1].inner);

  assert(TargetsExactly(*ptg, pdelta_f, { delta_f }));

  assert(TargetsExactly(*ptg, plambda_g, { lambda_g }));
  assert(TargetsExactly(*ptg, plambda_h, { lambda_h }));

  assert(TargetsExactly(*ptg, lambda_g_arg0, { delta_f }));

  assert(TargetsExactly(*ptg, lambda_h_cv0, { delta_f }));
  assert(TargetsExactly(*ptg, lambda_h_cv1, { lambda_g }));

  assert(EscapedIsExactly(*ptg, { lambda_h }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestDelta1", TestDelta1)

static void
TestDelta2()
{
  jlm::tests::DeltaTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numDeltaNodes() == 2);
  assert(ptg->numLambdaNodes() == 2);
  assert(ptg->numMappedRegisters() == 8);

  auto delta_d1 = ptg->getNodeForDelta(*test.delta_d1);
  auto delta_d1_out = ptg->getNodeForRegister(test.delta_d1->output());

  auto delta_d2 = ptg->getNodeForDelta(*test.delta_d2);
  auto delta_d2_out = ptg->getNodeForRegister(test.delta_d2->output());

  auto lambda_f1 = ptg->getNodeForLambda(*test.lambda_f1);
  auto lambda_f1_out = ptg->getNodeForRegister(*test.lambda_f1->output());
  auto lambda_f1_cvd1 = ptg->getNodeForRegister(*test.lambda_f1->GetContextVars()[0].inner);

  auto lambda_f2 = ptg->getNodeForLambda(*test.lambda_f2);
  auto lambda_f2_out = ptg->getNodeForRegister(*test.lambda_f2->output());
  auto lambda_f2_cvd1 = ptg->getNodeForRegister(*test.lambda_f2->GetContextVars()[0].inner);
  auto lambda_f2_cvd2 = ptg->getNodeForRegister(*test.lambda_f2->GetContextVars()[1].inner);
  auto lambda_f2_cvf1 = ptg->getNodeForRegister(*test.lambda_f2->GetContextVars()[2].inner);

  assert(TargetsExactly(*ptg, delta_d1_out, { delta_d1 }));
  assert(TargetsExactly(*ptg, delta_d2_out, { delta_d2 }));

  assert(TargetsExactly(*ptg, lambda_f1_out, { lambda_f1 }));
  assert(TargetsExactly(*ptg, lambda_f2_out, { lambda_f2 }));

  assert(lambda_f1_cvd1 == delta_d1_out);
  assert(lambda_f2_cvd1 == delta_d1_out);
  assert(lambda_f2_cvd2 == delta_d2_out);
  assert(lambda_f2_cvf1 == lambda_f1_out);

  assert(EscapedIsExactly(*ptg, { lambda_f2 }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestDelta2", TestDelta2)

static void
TestImports()
{
  jlm::tests::ImportTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 2);
  assert(ptg->numImportNodes() == 2);
  assert(ptg->numMappedRegisters() == 8);

  auto d1 = ptg->getNodeForImport(*test.import_d1);
  auto import_d1 = ptg->getNodeForRegister(*test.import_d1);

  auto d2 = ptg->getNodeForImport(*test.import_d2);
  auto import_d2 = ptg->getNodeForRegister(*test.import_d2);

  auto lambda_f1 = ptg->getNodeForLambda(*test.lambda_f1);
  auto lambda_f1_out = ptg->getNodeForRegister(*test.lambda_f1->output());
  auto lambda_f1_cvd1 = ptg->getNodeForRegister(*test.lambda_f1->GetContextVars()[0].inner);

  auto lambda_f2 = ptg->getNodeForLambda(*test.lambda_f2);
  auto lambda_f2_out = ptg->getNodeForRegister(*test.lambda_f2->output());
  auto lambda_f2_cvd1 = ptg->getNodeForRegister(*test.lambda_f2->GetContextVars()[0].inner);
  auto lambda_f2_cvd2 = ptg->getNodeForRegister(*test.lambda_f2->GetContextVars()[1].inner);
  auto lambda_f2_cvf1 = ptg->getNodeForRegister(*test.lambda_f2->GetContextVars()[2].inner);

  assert(TargetsExactly(*ptg, import_d1, { d1 }));
  assert(TargetsExactly(*ptg, import_d2, { d2 }));

  assert(TargetsExactly(*ptg, lambda_f1_out, { lambda_f1 }));
  assert(TargetsExactly(*ptg, lambda_f2_out, { lambda_f2 }));

  assert(lambda_f1_cvd1 == import_d1);
  assert(lambda_f2_cvd1 == import_d1);
  assert(lambda_f2_cvd2 == import_d2);
  assert(lambda_f2_cvf1 == lambda_f1_out);

  assert(EscapedIsExactly(*ptg, { lambda_f2, d1, d2 }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestImports", TestImports)

static void
TestPhi1()
{
  jlm::tests::PhiTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numAllocaNodes() == 1);
  assert(ptg->numLambdaNodes() == 2);
  assert(ptg->numMappedRegisters() == 16);

  auto lambda_fib = ptg->getNodeForLambda(*test.lambda_fib);
  auto lambda_fib_out = ptg->getNodeForRegister(*test.lambda_fib->output());
  auto lambda_fib_arg1 = ptg->getNodeForRegister(*test.lambda_fib->GetFunctionArguments()[1]);

  auto lambda_test = ptg->getNodeForLambda(*test.lambda_test);
  auto lambda_test_out = ptg->getNodeForRegister(*test.lambda_test->output());

  auto phi_rv = ptg->getNodeForRegister(*test.phi->GetFixVars()[0].output);
  auto phi_rv_arg = ptg->getNodeForRegister(*test.phi->GetFixVars()[0].recref);

  auto gamma_result = ptg->getNodeForRegister(*test.gamma->GetEntryVars()[1].branchArgument[0]);
  auto gamma_fib = ptg->getNodeForRegister(*test.gamma->GetEntryVars()[2].branchArgument[0]);

  auto alloca = ptg->getNodeForAlloca(*test.alloca);
  auto alloca_out = ptg->getNodeForRegister(*test.alloca->output(0));

  assert(TargetsExactly(*ptg, lambda_fib_out, { lambda_fib }));
  assert(TargetsExactly(*ptg, lambda_fib_arg1, { alloca }));

  assert(TargetsExactly(*ptg, lambda_test_out, { lambda_test }));

  assert(phi_rv == lambda_fib_out);
  assert(TargetsExactly(*ptg, phi_rv_arg, { lambda_fib }));

  assert(TargetsExactly(*ptg, gamma_result, { alloca }));
  assert(TargetsExactly(*ptg, gamma_fib, { lambda_fib }));

  assert(TargetsExactly(*ptg, alloca_out, { alloca }));

  assert(EscapedIsExactly(*ptg, { lambda_test }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestPhi1", TestPhi1)

static void
TestExternalMemory()
{
  jlm::tests::ExternalMemoryTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 3);

  auto lambdaF = ptg->getNodeForLambda(*test.LambdaF);
  auto lambdaFArgument0 = ptg->getNodeForRegister(*test.LambdaF->GetFunctionArguments()[0]);
  auto lambdaFArgument1 = ptg->getNodeForRegister(*test.LambdaF->GetFunctionArguments()[1]);

  assert(TargetsExactly(*ptg, lambdaFArgument0, { lambdaF, ptg->getExternalMemoryNode() }));
  assert(TargetsExactly(*ptg, lambdaFArgument1, { lambdaF, ptg->getExternalMemoryNode() }));

  assert(EscapedIsExactly(*ptg, { lambdaF }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestExternalMemory",
    TestExternalMemory)

static void
TestEscapedMemory1()
{
  jlm::tests::EscapedMemoryTest1 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numDeltaNodes() == 4);
  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 10);

  auto lambdaTestArgument0 = ptg->getNodeForRegister(*test.LambdaTest->GetFunctionArguments()[0]);
  auto lambdaTestCv0 = ptg->getNodeForRegister(*test.LambdaTest->GetContextVars()[0].inner);
  auto loadNode1Output = ptg->getNodeForRegister(*test.LoadNode1->output(0));

  auto deltaA = ptg->getNodeForDelta(*test.DeltaA);
  auto deltaB = ptg->getNodeForDelta(*test.DeltaB);
  auto deltaX = ptg->getNodeForDelta(*test.DeltaX);
  auto deltaY = ptg->getNodeForDelta(*test.DeltaY);
  auto lambdaTest = ptg->getNodeForLambda(*test.LambdaTest);
  auto externalMemory = ptg->getExternalMemoryNode();

  assert(TargetsExactly(
      *ptg,
      lambdaTestArgument0,
      { deltaA, deltaX, deltaY, lambdaTest, externalMemory }));
  assert(TargetsExactly(*ptg, lambdaTestCv0, { deltaB }));
  assert(TargetsExactly(
      *ptg,
      loadNode1Output,
      { deltaA, deltaX, deltaY, lambdaTest, externalMemory }));

  assert(EscapedIsExactly(*ptg, { lambdaTest, deltaA, deltaX, deltaY }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestEscapedMemory1",
    TestEscapedMemory1)

static void
TestEscapedMemory2()
{
  jlm::tests::EscapedMemoryTest2 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numImportNodes() == 2);
  assert(ptg->numLambdaNodes() == 3);
  assert(ptg->numMallocNodes() == 2);
  assert(ptg->numMappedRegisters() == 10);

  auto returnAddressFunction = ptg->getNodeForLambda(*test.ReturnAddressFunction);
  auto callExternalFunction1 = ptg->getNodeForLambda(*test.CallExternalFunction1);
  auto callExternalFunction2 = ptg->getNodeForLambda(*test.CallExternalFunction2);
  auto returnAddressMalloc = ptg->getNodeForMalloc(*test.ReturnAddressMalloc);
  auto callExternalFunction1Malloc = ptg->getNodeForMalloc(*test.CallExternalFunction1Malloc);
  auto externalMemory = ptg->getExternalMemoryNode();
  auto externalFunction1Import = ptg->getNodeForImport(*test.ExternalFunction1Import);
  auto externalFunction2Import = ptg->getNodeForImport(*test.ExternalFunction2Import);

  auto externalFunction2CallResult =
      ptg->getNodeForRegister(*test.ExternalFunction2Call->output(0));

  assert(TargetsExactly(
      *ptg,
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
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestEscapedMemory2",
    TestEscapedMemory2)

static void
TestEscapedMemory3()
{
  jlm::tests::EscapedMemoryTest3 test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numDeltaNodes() == 1);
  assert(ptg->numImportNodes() == 1);
  assert(ptg->numLambdaNodes() == 1);
  assert(ptg->numMappedRegisters() == 5);

  auto lambdaTest = ptg->getNodeForLambda(*test.LambdaTest);
  auto deltaGlobal = ptg->getNodeForDelta(*test.DeltaGlobal);
  auto importExternalFunction = ptg->getNodeForImport(*test.ImportExternalFunction);
  auto externalMemory = ptg->getExternalMemoryNode();

  auto callExternalFunctionResult = ptg->getNodeForRegister(*test.CallExternalFunction->output(0));

  assert(TargetsExactly(
      *ptg,
      callExternalFunctionResult,
      { lambdaTest, deltaGlobal, importExternalFunction, externalMemory }));

  assert(EscapedIsExactly(*ptg, { lambdaTest, deltaGlobal, importExternalFunction }));
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestEscapedMemory3",
    TestEscapedMemory3)

static void
TestMemcpy()
{
  jlm::tests::MemcpyTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numDeltaNodes() == 2);
  assert(ptg->numLambdaNodes() == 2);
  assert(ptg->numMappedRegisters() == 11);

  auto localArray = ptg->getNodeForDelta(test.LocalArray());
  auto globalArray = ptg->getNodeForDelta(test.GlobalArray());

  auto memCpyDest = ptg->getNodeForRegister(*test.Memcpy().input(0)->origin());
  auto memCpySrc = ptg->getNodeForRegister(*test.Memcpy().input(1)->origin());

  auto lambdaF = ptg->getNodeForLambda(test.LambdaF());
  auto lambdaG = ptg->getNodeForLambda(test.LambdaG());

  assert(TargetsExactly(*ptg, memCpyDest, { globalArray }));
  assert(TargetsExactly(*ptg, memCpySrc, { localArray }));

  assert(EscapedIsExactly(*ptg, { globalArray, localArray, lambdaF, lambdaG }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestMemcpy", TestMemcpy)

static void
TestLinkedList()
{
  jlm::tests::LinkedListTest test;
  const auto ptg = RunAndersen(test.module());

  assert(ptg->numAllocaNodes() == 1);
  assert(ptg->numDeltaNodes() == 1);
  assert(ptg->numLambdaNodes() == 1);

  auto allocaNode = ptg->getNodeForAlloca(test.GetAlloca());
  auto deltaMyListNode = ptg->getNodeForDelta(test.GetDeltaMyList());
  auto lambdaNextNode = ptg->getNodeForLambda(test.GetLambdaNext());
  auto externalMemoryNode = ptg->getExternalMemoryNode();

  assert(TargetsExactly(*ptg, allocaNode, { deltaMyListNode, lambdaNextNode, externalMemoryNode }));
  assert(TargetsExactly(
      *ptg,
      deltaMyListNode,
      { deltaMyListNode, lambdaNextNode, externalMemoryNode }));
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestLinkedList", TestLinkedList)

static void
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
  assert(statistics.GetMeasurementValue<uint64_t>("#PointsToGraphNodes") == ptg->numNodes());
  assert(statistics.GetTimerElapsedNanoseconds("AnalysisTimer") > 0);
}
JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestAndersen-TestStatistics", TestStatistics)

static void
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
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestConfiguration",
    TestConfiguration)

static void
TestConstructPointsToGraph()
{
  using namespace jlm::llvm::aa;

  jlm::tests::AllMemoryNodesTest rvsdg;
  rvsdg.InitializeTest();

  // Arrange a very standard set of memory objects and registers
  PointerObjectSet set;
  auto alloca0 = set.CreateAllocaMemoryObject(rvsdg.GetAllocaNode(), true);
  auto allocaR = set.CreateRegisterPointerObject(rvsdg.GetAllocaOutput());
  auto import0 = set.CreateImportMemoryObject(rvsdg.GetImportOutput(), true);
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
  auto allocaNode = ptg->getNodeForAlloca(rvsdg.GetAllocaNode());
  auto allocaRNode = ptg->getNodeForRegister(rvsdg.GetAllocaOutput());
  auto importNode = ptg->getNodeForImport(rvsdg.GetImportOutput());
  auto importRNode = ptg->getNodeForRegister(rvsdg.GetImportOutput());
  auto lambdaNode = ptg->getNodeForLambda(rvsdg.GetLambdaNode());
  auto lambdaRNode = ptg->getNodeForRegister(rvsdg.GetLambdaOutput());
  auto mallocNode = ptg->getNodeForMalloc(rvsdg.GetMallocNode());
  auto mallocRNode = ptg->getNodeForRegister(rvsdg.GetMallocOutput());
  auto deltaNode = ptg->getNodeForDelta(rvsdg.GetDeltaNode());
  auto deltaRNode = ptg->getNodeForRegister(rvsdg.GetDeltaOutput());

  auto externalMemory = ptg->getExternalMemoryNode();

  // ==== Targets of allocaR, importR and deltaR ==== (2 explicit, 2 total)
  // Make sure unification causes allocaR to point to all of its pointees
  assert(TargetsExactly(*ptg, allocaRNode, { allocaNode, importNode }));
  // The unified registers should share RegisterNode
  assert(allocaRNode == importRNode);
  // deltaR was mapped to importR, so it too should share RegisterNode
  assert(allocaRNode == deltaRNode);

  // ==== Targets of lambdaR ==== (1 explicit, 1 total)
  assert(TargetsExactly(*ptg, lambdaRNode, { lambdaNode }));

  // ==== Targets of mallocR ==== (1 explicit, 4 total)
  // mallocR points to mallocNode, as well as everything that has escaped
  assert(TargetsExactly(*ptg, mallocRNode, { mallocNode, deltaNode, importNode, externalMemory }));

  // ==== Targets of alloca0 ==== (1 explicit, 1 total)
  assert(TargetsExactly(*ptg, allocaNode, { lambdaNode }));

  // ==== Targets of malloc0 ==== (1 explicit, 1 total)
  // Because malloc0 was unified with lambdaR, they have the same targets, but are not the same node
  assert(lambdaRNode != mallocNode);
  assert(TargetsExactly(*ptg, mallocNode, { lambdaNode }));

  // ==== Targets of delta0 ==== (0 explicit, 3 total)
  // deltaNode points to everything that has escaped, but only implicitly
  assert(ptg->isExternallyAvailable(deltaNode));
  assert(ptg->isTargetingAllExternallyAvailable(deltaNode));
  assert(TargetsExactly(*ptg, deltaNode, { deltaNode, importNode, externalMemory }));

  // ==== Targets of import0 ==== (0 explicit, 3 total)
  // The importNode should be flagged as targeting everything externally available
  assert(ptg->isExternallyAvailable(importNode));
  assert(ptg->isTargetingAllExternallyAvailable(importNode));
  assert(TargetsExactly(*ptg, importNode, { deltaNode, importNode, externalMemory }));

  // ==== Targets of lambda0 ==== (0 explicit, 0 total)
  // Because lambda nodes are marked as CanPoint()=false, they have no targets
  assert(!ptg->isTargetingAllExternallyAvailable(lambdaNode));
  assert(!ptg->isExternallyAvailable(lambdaNode));
  assert(TargetsExactly(*ptg, lambdaNode, {}));

  // ==== Targets of externalMemoryNode ==== (0 explicit, 3 total)
  // While the external node has no explicit targets, it targets everything externally available
  assert(ptg->getExplicitTargets(externalMemory).Size() == 0);
  assert(TargetsExactly(*ptg, externalMemory, { deltaNode, importNode, externalMemory }));

  // 5 memory node + register pairs, plus one external memory node,
  // minus two registers that have been unified into allocaRNode
  assert(ptg->numNodes() == 5 * 2 + 1 - 2);

  // Adding up the out-edges for all nodes
  auto [numExplicitEdges, numTotalEdges] = ptg->numEdges();
  assert(numExplicitEdges == 2 + 1 + 1 + 1 + 1);
  // total edges also includes the outgoing implicit edges from the external node
  assert(numTotalEdges == 2 + 1 + 4 + 1 + 1 + 3 + 3 + 0 + 3);
}
JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestAndersen-TestConstructPointsToGraph",
    TestConstructPointsToGraph)
