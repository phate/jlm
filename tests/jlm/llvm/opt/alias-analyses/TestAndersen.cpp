/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
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

TEST(AndersenTests, TestStore1)
{
  jlm::llvm::StoreTest1 test;
  const auto ptg = RunAndersen(test.module());

  // std::unordered_map<const jlm::rvsdg::output*, std::string> outputMap;
  // std::cout << jlm::rvsdg::view(test.graph().GetRootRegion(), outputMap) << std::endl;
  // std::cout << jlm::llvm::aa::PointsToGraph::dumpGraph(*ptg, outputMap) << std::endl;

  EXPECT_EQ(ptg->numAllocaNodes(), 4);
  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 5);

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

  EXPECT_TRUE(TargetsExactly(*ptg, alloca_a, { alloca_b }));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_b, { alloca_c }));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_c, { alloca_d }));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_d, {}));

  EXPECT_TRUE(TargetsExactly(*ptg, palloca_a, { alloca_a }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_b, { alloca_b }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_c, { alloca_c }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_d, { alloca_d }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda, {}));
  EXPECT_TRUE(TargetsExactly(*ptg, plambda, { lambda }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestStore2)
{
  jlm::llvm::StoreTest2 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numAllocaNodes(), 5);
  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 6);

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

  EXPECT_TRUE(TargetsExactly(*ptg, alloca_a, {}));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_b, {}));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_x, { alloca_a }));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_y, { alloca_b }));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_p, { alloca_x, alloca_y }));

  EXPECT_TRUE(TargetsExactly(*ptg, palloca_a, { alloca_a }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_b, { alloca_b }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_x, { alloca_x }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_y, { alloca_y }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_p, { alloca_p }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda, {}));
  EXPECT_TRUE(TargetsExactly(*ptg, plambda, { lambda }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestLoad1)
{
  jlm::llvm::LoadTest1 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 3);

  auto loadResult = ptg->getNodeForRegister(*test.load_p->output(0));

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaOutput = ptg->getNodeForRegister(*test.lambda->output());
  auto lambdaArgument0 = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[0]);

  EXPECT_TRUE(TargetsExactly(*ptg, loadResult, { lambda, ptg->getExternalMemoryNode() }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambdaOutput, { lambda }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaArgument0, { lambda, ptg->getExternalMemoryNode() }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestLoad2)
{
  jlm::llvm::LoadTest2 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numAllocaNodes(), 5);
  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 8);

  auto alloca_a = ptg->getNodeForAlloca(*test.alloca_a);
  auto alloca_b = ptg->getNodeForAlloca(*test.alloca_b);
  auto alloca_x = ptg->getNodeForAlloca(*test.alloca_x);
  auto alloca_y = ptg->getNodeForAlloca(*test.alloca_y);
  auto alloca_p = ptg->getNodeForAlloca(*test.alloca_p);

  auto pload_x = ptg->getNodeForRegister(*test.load_x->output(0));
  auto pload_a = ptg->getNodeForRegister(*test.load_a->output(0));

  auto lambdaMemoryNode = ptg->getNodeForLambda(*test.lambda);

  EXPECT_TRUE(TargetsExactly(*ptg, alloca_x, { alloca_a }));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_y, { alloca_a, alloca_b }));
  EXPECT_TRUE(TargetsExactly(*ptg, alloca_p, { alloca_x }));

  EXPECT_TRUE(TargetsExactly(*ptg, pload_x, { alloca_x }));
  EXPECT_TRUE(TargetsExactly(*ptg, pload_a, { alloca_a }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambdaMemoryNode }));
}

TEST(AndersenTests, TestLoadFromUndef)
{
  jlm::llvm::LoadFromUndefTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 2);

  auto lambdaMemoryNode = ptg->getNodeForLambda(test.Lambda());
  auto undefValueNode = ptg->getNodeForRegister(*test.UndefValueNode()->output(0));

  EXPECT_TRUE(TargetsExactly(*ptg, undefValueNode, {}));
  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambdaMemoryNode }));
}

TEST(AndersenTests, TestGetElementPtr)
{
  jlm::llvm::GetElementPtrTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 4);

  // We only care about the getelemenptr's in this test, skipping the validation for all other nodes
  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto gepX = ptg->getNodeForRegister(*test.getElementPtrX->output(0));
  auto gepY = ptg->getNodeForRegister(*test.getElementPtrY->output(0));

  // The RegisterNode is the same
  EXPECT_EQ(gepX, gepY);

  EXPECT_TRUE(TargetsExactly(*ptg, gepX, { lambda, ptg->getExternalMemoryNode() }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestBitCast)
{
  jlm::llvm::BitCastTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 3);

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaOut = ptg->getNodeForRegister(*test.lambda->output());
  auto lambdaArg = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[0]);
  auto bitCast = ptg->getNodeForRegister(*test.bitCast->output(0));

  EXPECT_TRUE(TargetsExactly(*ptg, lambdaOut, { lambda }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaArg, { lambda, ptg->getExternalMemoryNode() }));
  EXPECT_TRUE(TargetsExactly(*ptg, bitCast, { lambda, ptg->getExternalMemoryNode() }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestConstantPointerNull)
{
  jlm::llvm::ConstantPointerNullTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 3);

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaOut = ptg->getNodeForRegister(*test.lambda->output());
  auto lambdaArg = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[0]);

  auto constantPointerNull = ptg->getNodeForRegister(*test.constantPointerNullNode->output(0));

  EXPECT_TRUE(TargetsExactly(*ptg, lambdaOut, { lambda }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaArg, { lambda, ptg->getExternalMemoryNode() }));
  EXPECT_TRUE(TargetsExactly(*ptg, constantPointerNull, {}));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestBits2Ptr)
{
  jlm::llvm::Bits2PtrTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 5);

  auto lambdaTestMemoryNode = ptg->getNodeForLambda(test.GetLambdaTest());
  auto externalMemoryNode = ptg->getExternalMemoryNode();

  auto callOutput0 = ptg->getNodeForRegister(*test.GetCallNode().output(0));
  auto bits2ptr = ptg->getNodeForRegister(*test.GetBitsToPtrNode().output(0));

  EXPECT_TRUE(TargetsExactly(*ptg, callOutput0, { lambdaTestMemoryNode, externalMemoryNode }));
  EXPECT_TRUE(TargetsExactly(*ptg, bits2ptr, { lambdaTestMemoryNode, externalMemoryNode }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambdaTestMemoryNode }));
}

TEST(AndersenTests, TestCall1)
{
  jlm::llvm::CallTest1 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numAllocaNodes(), 3);
  EXPECT_EQ(ptg->numLambdaNodes(), 3);
  EXPECT_EQ(ptg->numMappedRegisters(), 12);

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

  EXPECT_TRUE(TargetsExactly(*ptg, palloca_x, { alloca_x }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_y, { alloca_y }));
  EXPECT_TRUE(TargetsExactly(*ptg, palloca_z, { alloca_z }));

  EXPECT_TRUE(TargetsExactly(*ptg, plambda_f, { lambda_f }));
  EXPECT_TRUE(TargetsExactly(*ptg, plambda_g, { lambda_g }));
  EXPECT_TRUE(TargetsExactly(*ptg, plambda_h, { lambda_h }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_f_arg0, { alloca_x }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_f_arg1, { alloca_y }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_g_arg0, { alloca_z }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_g_arg1, { alloca_z }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_h_cv0, { lambda_f }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_h_cv1, { lambda_g }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda_h }));
}

TEST(AndersenTests, TestCall2)
{
  jlm::llvm::CallTest2 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 3);
  EXPECT_EQ(ptg->numMallocNodes(), 1);
  EXPECT_EQ(ptg->numImportNodes(), 0);
  EXPECT_EQ(ptg->numMappedRegisters(), 11);

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

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_create_out, { lambda_create }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_destroy_out, { lambda_destroy }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_destroy_arg, { malloc }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_out, { lambda_test }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_cv1, { lambda_create }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_cv2, { lambda_destroy }));

  EXPECT_TRUE(TargetsExactly(*ptg, call_create1_out, { malloc }));
  EXPECT_TRUE(TargetsExactly(*ptg, call_create2_out, { malloc }));

  EXPECT_TRUE(TargetsExactly(*ptg, malloc_out, { malloc }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda_test }));
}

TEST(AndersenTests, TestIndirectCall1)
{
  jlm::llvm::IndirectCallTest1 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 4);
  EXPECT_EQ(ptg->numImportNodes(), 0);
  EXPECT_EQ(ptg->numMappedRegisters(), 11);

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

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_three_out, { lambda_three }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_four_out, { lambda_four }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_indcall_out, { lambda_indcall }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_indcall_arg, { lambda_three, lambda_four }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_out, { lambda_test }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_cv0, { lambda_indcall }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_cv1, { lambda_four }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_cv2, { lambda_three }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda_test }));
}

TEST(AndersenTests, TestIndirectCall2)
{
  jlm::llvm::IndirectCallTest2 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numAllocaNodes(), 3);
  EXPECT_EQ(ptg->numLambdaNodes(), 7);
  EXPECT_EQ(ptg->numDeltaNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 27);

  auto lambdaThree = ptg->getNodeForLambda(test.GetLambdaThree());
  auto lambdaThreeOutput = ptg->getNodeForRegister(*test.GetLambdaThree().output());

  auto lambdaFour = ptg->getNodeForLambda(test.GetLambdaFour());
  auto lambdaFourOutput = ptg->getNodeForRegister(*test.GetLambdaFour().output());

  EXPECT_TRUE(TargetsExactly(*ptg, lambdaThreeOutput, { lambdaThree }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaFourOutput, { lambdaFour }));
}

TEST(AndersenTests, TestExternalCall1)
{
  jlm::llvm::ExternalCallTest1 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numAllocaNodes(), 2);
  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numImportNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 10);

  auto lambdaF = ptg->getNodeForLambda(test.LambdaF());
  auto lambdaFArgument0 = ptg->getNodeForRegister(*test.LambdaF().GetFunctionArguments()[0]);
  auto lambdaFArgument1 = ptg->getNodeForRegister(*test.LambdaF().GetFunctionArguments()[1]);
  auto importG = ptg->getNodeForImport(test.ExternalGArgument());

  auto callResult = ptg->getNodeForRegister(*test.CallG().output(0));

  auto externalMemory = ptg->getExternalMemoryNode();

  EXPECT_TRUE(TargetsExactly(*ptg, lambdaFArgument0, { lambdaF, importG, externalMemory }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaFArgument1, { lambdaF, importG, externalMemory }));
  EXPECT_TRUE(TargetsExactly(*ptg, callResult, { lambdaF, importG, externalMemory }));
}

TEST(AndersenTests, TestGamma)
{
  jlm::llvm::GammaTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 15);

  auto lambda = ptg->getNodeForLambda(*test.lambda);

  for (size_t n = 1; n < 5; n++)
  {
    auto lambdaArgument = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[n]);
    EXPECT_TRUE(TargetsExactly(*ptg, lambdaArgument, { lambda, ptg->getExternalMemoryNode() }));
  }

  auto entryvars = test.gamma->GetEntryVars();
  EXPECT_EQ(entryvars.size(), 4);
  for (const auto & entryvar : entryvars)
  {
    auto argument0 = ptg->getNodeForRegister(*entryvar.branchArgument[0]);
    auto argument1 = ptg->getNodeForRegister(*entryvar.branchArgument[1]);

    EXPECT_TRUE(TargetsExactly(*ptg, argument0, { lambda, ptg->getExternalMemoryNode() }));
    EXPECT_TRUE(TargetsExactly(*ptg, argument1, { lambda, ptg->getExternalMemoryNode() }));
  }

  for (size_t n = 0; n < 4; n++)
  {
    auto gammaOutput = ptg->getNodeForRegister(*test.gamma->GetExitVars()[0].output);
    EXPECT_TRUE(TargetsExactly(*ptg, gammaOutput, { lambda, ptg->getExternalMemoryNode() }));
  }

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestTheta)
{
  jlm::llvm::ThetaTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 5);

  auto lambda = ptg->getNodeForLambda(*test.lambda);
  auto lambdaArgument1 = ptg->getNodeForRegister(*test.lambda->GetFunctionArguments()[1]);
  auto lambdaOutput = ptg->getNodeForRegister(*test.lambda->output());

  auto gepOutput = ptg->getNodeForRegister(*test.gep->output(0));

  auto thetaArgument2 = ptg->getNodeForRegister(*test.theta->GetLoopVars()[2].pre);
  auto thetaOutput2 = ptg->getNodeForRegister(*test.theta->output(2));

  EXPECT_TRUE(TargetsExactly(*ptg, lambdaArgument1, { lambda, ptg->getExternalMemoryNode() }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaOutput, { lambda }));

  EXPECT_TRUE(TargetsExactly(*ptg, gepOutput, { lambda, ptg->getExternalMemoryNode() }));

  EXPECT_TRUE(TargetsExactly(*ptg, thetaArgument2, { lambda, ptg->getExternalMemoryNode() }));
  EXPECT_TRUE(TargetsExactly(*ptg, thetaOutput2, { lambda, ptg->getExternalMemoryNode() }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda }));
}

TEST(AndersenTests, TestDelta1)
{
  jlm::llvm::DeltaTest1 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numDeltaNodes(), 1);
  EXPECT_EQ(ptg->numLambdaNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 6);

  auto delta_f = ptg->getNodeForDelta(*test.delta_f);
  auto pdelta_f = ptg->getNodeForRegister(test.delta_f->output());

  auto lambda_g = ptg->getNodeForLambda(*test.lambda_g);
  auto plambda_g = ptg->getNodeForRegister(*test.lambda_g->output());
  auto lambda_g_arg0 = ptg->getNodeForRegister(*test.lambda_g->GetFunctionArguments()[0]);

  auto lambda_h = ptg->getNodeForLambda(*test.lambda_h);
  auto plambda_h = ptg->getNodeForRegister(*test.lambda_h->output());
  auto lambda_h_cv0 = ptg->getNodeForRegister(*test.lambda_h->GetContextVars()[0].inner);
  auto lambda_h_cv1 = ptg->getNodeForRegister(*test.lambda_h->GetContextVars()[1].inner);

  EXPECT_TRUE(TargetsExactly(*ptg, pdelta_f, { delta_f }));

  EXPECT_TRUE(TargetsExactly(*ptg, plambda_g, { lambda_g }));
  EXPECT_TRUE(TargetsExactly(*ptg, plambda_h, { lambda_h }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_g_arg0, { delta_f }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_h_cv0, { delta_f }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_h_cv1, { lambda_g }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda_h }));
}

TEST(AndersenTests, TestDelta2)
{
  jlm::llvm::DeltaTest2 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numDeltaNodes(), 2);
  EXPECT_EQ(ptg->numLambdaNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 8);

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

  EXPECT_TRUE(TargetsExactly(*ptg, delta_d1_out, { delta_d1 }));
  EXPECT_TRUE(TargetsExactly(*ptg, delta_d2_out, { delta_d2 }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_f1_out, { lambda_f1 }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_f2_out, { lambda_f2 }));

  EXPECT_EQ(lambda_f1_cvd1, delta_d1_out);
  EXPECT_EQ(lambda_f2_cvd1, delta_d1_out);
  EXPECT_EQ(lambda_f2_cvd2, delta_d2_out);
  EXPECT_EQ(lambda_f2_cvf1, lambda_f1_out);

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda_f2 }));
}

TEST(AndersenTests, TestImports)
{
  jlm::llvm::ImportTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 2);
  EXPECT_EQ(ptg->numImportNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 8);

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

  EXPECT_TRUE(TargetsExactly(*ptg, import_d1, { d1 }));
  EXPECT_TRUE(TargetsExactly(*ptg, import_d2, { d2 }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_f1_out, { lambda_f1 }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_f2_out, { lambda_f2 }));

  EXPECT_EQ(lambda_f1_cvd1, import_d1);
  EXPECT_EQ(lambda_f2_cvd1, import_d1);
  EXPECT_EQ(lambda_f2_cvd2, import_d2);
  EXPECT_EQ(lambda_f2_cvf1, lambda_f1_out);

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda_f2, d1, d2 }));
}

TEST(AndersenTests, TestPhi1)
{
  jlm::llvm::PhiTest1 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numAllocaNodes(), 1);
  EXPECT_EQ(ptg->numLambdaNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 16);

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

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_fib_out, { lambda_fib }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambda_fib_arg1, { alloca }));

  EXPECT_TRUE(TargetsExactly(*ptg, lambda_test_out, { lambda_test }));

  EXPECT_EQ(phi_rv, lambda_fib_out);
  EXPECT_TRUE(TargetsExactly(*ptg, phi_rv_arg, { lambda_fib }));

  EXPECT_TRUE(TargetsExactly(*ptg, gamma_result, { alloca }));
  EXPECT_TRUE(TargetsExactly(*ptg, gamma_fib, { lambda_fib }));

  EXPECT_TRUE(TargetsExactly(*ptg, alloca_out, { alloca }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambda_test }));
}

TEST(AndersenTests, TestExternalMemory)
{
  jlm::llvm::ExternalMemoryTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 3);

  auto lambdaF = ptg->getNodeForLambda(*test.LambdaF);
  auto lambdaFArgument0 = ptg->getNodeForRegister(*test.LambdaF->GetFunctionArguments()[0]);
  auto lambdaFArgument1 = ptg->getNodeForRegister(*test.LambdaF->GetFunctionArguments()[1]);

  EXPECT_TRUE(TargetsExactly(*ptg, lambdaFArgument0, { lambdaF, ptg->getExternalMemoryNode() }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaFArgument1, { lambdaF, ptg->getExternalMemoryNode() }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambdaF }));
}

TEST(AndersenTests, TestEscapedMemory1)
{
  jlm::llvm::EscapedMemoryTest1 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numDeltaNodes(), 4);
  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 10);

  auto lambdaTestArgument0 = ptg->getNodeForRegister(*test.LambdaTest->GetFunctionArguments()[0]);
  auto lambdaTestCv0 = ptg->getNodeForRegister(*test.LambdaTest->GetContextVars()[0].inner);
  auto loadNode1Output = ptg->getNodeForRegister(*test.LoadNode1->output(0));

  auto deltaA = ptg->getNodeForDelta(*test.DeltaA);
  auto deltaB = ptg->getNodeForDelta(*test.DeltaB);
  auto deltaX = ptg->getNodeForDelta(*test.DeltaX);
  auto deltaY = ptg->getNodeForDelta(*test.DeltaY);
  auto lambdaTest = ptg->getNodeForLambda(*test.LambdaTest);
  auto externalMemory = ptg->getExternalMemoryNode();

  EXPECT_TRUE(TargetsExactly(
      *ptg,
      lambdaTestArgument0,
      { deltaA, deltaX, deltaY, lambdaTest, externalMemory }));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaTestCv0, { deltaB }));
  EXPECT_TRUE(TargetsExactly(
      *ptg,
      loadNode1Output,
      { deltaA, deltaX, deltaY, lambdaTest, externalMemory }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambdaTest, deltaA, deltaX, deltaY }));
}

TEST(AndersenTests, TestEscapedMemory2)
{
  jlm::llvm::EscapedMemoryTest2 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numImportNodes(), 2);
  EXPECT_EQ(ptg->numLambdaNodes(), 3);
  EXPECT_EQ(ptg->numMallocNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 10);

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

  EXPECT_TRUE(TargetsExactly(
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

  EXPECT_TRUE(EscapedIsExactly(
      *ptg,
      { returnAddressFunction,
        callExternalFunction1,
        callExternalFunction2,
        returnAddressMalloc,
        callExternalFunction1Malloc,
        externalFunction1Import,
        externalFunction2Import }));
}

TEST(AndersenTests, TestEscapedMemory3)
{
  jlm::llvm::EscapedMemoryTest3 test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numDeltaNodes(), 1);
  EXPECT_EQ(ptg->numImportNodes(), 1);
  EXPECT_EQ(ptg->numLambdaNodes(), 1);
  EXPECT_EQ(ptg->numMappedRegisters(), 5);

  auto lambdaTest = ptg->getNodeForLambda(*test.LambdaTest);
  auto deltaGlobal = ptg->getNodeForDelta(*test.DeltaGlobal);
  auto importExternalFunction = ptg->getNodeForImport(*test.ImportExternalFunction);
  auto externalMemory = ptg->getExternalMemoryNode();

  auto callExternalFunctionResult = ptg->getNodeForRegister(*test.CallExternalFunction->output(0));

  EXPECT_TRUE(TargetsExactly(
      *ptg,
      callExternalFunctionResult,
      { lambdaTest, deltaGlobal, importExternalFunction, externalMemory }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { lambdaTest, deltaGlobal, importExternalFunction }));
}

TEST(AndersenTests, TestMemcpy)
{
  jlm::llvm::MemcpyTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numDeltaNodes(), 2);
  EXPECT_EQ(ptg->numLambdaNodes(), 2);
  EXPECT_EQ(ptg->numMappedRegisters(), 11);

  auto localArray = ptg->getNodeForDelta(test.LocalArray());
  auto globalArray = ptg->getNodeForDelta(test.GlobalArray());

  auto memCpyDest = ptg->getNodeForRegister(*test.Memcpy().input(0)->origin());
  auto memCpySrc = ptg->getNodeForRegister(*test.Memcpy().input(1)->origin());

  auto lambdaF = ptg->getNodeForLambda(test.LambdaF());
  auto lambdaG = ptg->getNodeForLambda(test.LambdaG());

  EXPECT_TRUE(TargetsExactly(*ptg, memCpyDest, { globalArray }));
  EXPECT_TRUE(TargetsExactly(*ptg, memCpySrc, { localArray }));

  EXPECT_TRUE(EscapedIsExactly(*ptg, { globalArray, localArray, lambdaF, lambdaG }));
}

TEST(AndersenTests, TestLinkedList)
{
  jlm::llvm::LinkedListTest test;
  const auto ptg = RunAndersen(test.module());

  EXPECT_EQ(ptg->numAllocaNodes(), 1);
  EXPECT_EQ(ptg->numDeltaNodes(), 1);
  EXPECT_EQ(ptg->numLambdaNodes(), 1);

  auto allocaNode = ptg->getNodeForAlloca(test.GetAlloca());
  auto deltaMyListNode = ptg->getNodeForDelta(test.GetDeltaMyList());
  auto lambdaNextNode = ptg->getNodeForLambda(test.GetLambdaNext());
  auto externalMemoryNode = ptg->getExternalMemoryNode();

  EXPECT_TRUE(
      TargetsExactly(*ptg, allocaNode, { deltaMyListNode, lambdaNextNode, externalMemoryNode }));
  EXPECT_TRUE(TargetsExactly(
      *ptg,
      deltaMyListNode,
      { deltaMyListNode, lambdaNextNode, externalMemoryNode }));
}

TEST(AndersenTests, TestStatistics)
{
  // Arrange
  jlm::llvm::LoadTest1 test;
  jlm::util::StatisticsCollectorSettings statisticsCollectorSettings(
      { jlm::util::Statistics::Id::AndersenAnalysis });
  jlm::util::StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  // Act
  jlm::llvm::aa::Andersen andersen;
  auto ptg = andersen.Analyze(test.module(), statisticsCollector);

  // Assert
  EXPECT_EQ(statisticsCollector.NumCollectedStatistics(), 1);
  const auto & statistics = *statisticsCollector.CollectedStatistics().begin();

  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#StoreConstraints"), 0);
  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#LoadConstraints"), 1);
  EXPECT_EQ(statistics.GetMeasurementValue<uint64_t>("#PointsToGraphNodes"), ptg->numNodes());
  EXPECT_GT(statistics.GetTimerElapsedNanoseconds("AnalysisTimer"), 0);
}

TEST(AndersenTests, TestConfiguration)
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
  EXPECT_FALSE(config.IsOfflineVariableSubstitutionEnabled());
  EXPECT_TRUE(config.IsOfflineConstraintNormalizationEnabled());
  EXPECT_EQ(config.GetSolver(), Andersen::Configuration::Solver::Naive);
  EXPECT_EQ(configString.find("OVS"), std::string::npos);
  EXPECT_NE(configString.find("NORM"), std::string::npos);
  EXPECT_NE(configString.find("Solver=Naive"), std::string::npos);

  // Arrange some more
  auto policy = PointerObjectConstraintSet::WorklistSolverPolicy::TwoPhaseLeastRecentlyFired;
  config.SetSolver(Andersen::Configuration::Solver::Worklist);
  config.SetWorklistSolverPolicy(policy);
  config.EnableOfflineVariableSubstitution(false);
  config.EnableOnlineCycleDetection(true);

  // Act
  configString = config.ToString();

  // Assert
  EXPECT_NE(configString.find("Solver=Worklist"), std::string::npos);
  EXPECT_NE(configString.find("Policy=TwoPhaseLeastRecentlyFired"), std::string::npos);
  EXPECT_EQ(configString.find("OVS"), std::string::npos);
  EXPECT_NE(configString.find("OnlineCD"), std::string::npos);

  // Arrange some more
  config.EnableOnlineCycleDetection(false);
  config.EnableHybridCycleDetection(true);

  // Act
  configString = config.ToString();

  // Assert
  EXPECT_EQ(configString.find("OnlineCD"), std::string::npos);
  EXPECT_NE(configString.find("HybridCD"), std::string::npos);
}

TEST(AndersenTests, TestConstructPointsToGraph)
{
  using namespace jlm::llvm::aa;

  jlm::llvm::AllMemoryNodesTest rvsdg;
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
  EXPECT_TRUE(TargetsExactly(*ptg, allocaRNode, { allocaNode, importNode }));
  // The unified registers should share RegisterNode
  EXPECT_EQ(allocaRNode, importRNode);
  // deltaR was mapped to importR, so it too should share RegisterNode
  EXPECT_EQ(allocaRNode, deltaRNode);

  // ==== Targets of lambdaR ==== (1 explicit, 1 total)
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaRNode, { lambdaNode }));

  // ==== Targets of mallocR ==== (1 explicit, 4 total)
  // mallocR points to mallocNode, as well as everything that has escaped
  EXPECT_TRUE(
      TargetsExactly(*ptg, mallocRNode, { mallocNode, deltaNode, importNode, externalMemory }));

  // ==== Targets of alloca0 ==== (1 explicit, 1 total)
  EXPECT_TRUE(TargetsExactly(*ptg, allocaNode, { lambdaNode }));

  // ==== Targets of malloc0 ==== (1 explicit, 1 total)
  // Because malloc0 was unified with lambdaR, they have the same targets, but are not the same node
  EXPECT_NE(lambdaRNode, mallocNode);
  EXPECT_TRUE(TargetsExactly(*ptg, mallocNode, { lambdaNode }));

  // ==== Targets of delta0 ==== (0 explicit, 3 total)
  // deltaNode points to everything that has escaped, but only implicitly
  EXPECT_TRUE(ptg->isExternallyAvailable(deltaNode));
  EXPECT_TRUE(ptg->isTargetingAllExternallyAvailable(deltaNode));
  EXPECT_TRUE(TargetsExactly(*ptg, deltaNode, { deltaNode, importNode, externalMemory }));

  // ==== Targets of import0 ==== (0 explicit, 3 total)
  // The importNode should be flagged as targeting everything externally available
  EXPECT_TRUE(ptg->isExternallyAvailable(importNode));
  EXPECT_TRUE(ptg->isTargetingAllExternallyAvailable(importNode));
  EXPECT_TRUE(TargetsExactly(*ptg, importNode, { deltaNode, importNode, externalMemory }));

  // ==== Targets of lambda0 ==== (0 explicit, 0 total)
  // Because lambda nodes are marked as CanPoint()=false, they have no targets
  EXPECT_FALSE(ptg->isTargetingAllExternallyAvailable(lambdaNode));
  EXPECT_FALSE(ptg->isExternallyAvailable(lambdaNode));
  EXPECT_TRUE(TargetsExactly(*ptg, lambdaNode, {}));

  // ==== Targets of externalMemoryNode ==== (0 explicit, 3 total)
  // While the external node has no explicit targets, it targets everything externally available
  EXPECT_EQ(ptg->getExplicitTargets(externalMemory).Size(), 0);
  EXPECT_TRUE(TargetsExactly(*ptg, externalMemory, { deltaNode, importNode, externalMemory }));

  // 5 memory node + register pairs, plus one external memory node,
  // minus two registers that have been unified into allocaRNode
  EXPECT_EQ(ptg->numNodes(), 5 * 2 + 1 - 2);

  // Adding up the out-edges for all nodes
  auto [numExplicitEdges, numTotalEdges] = ptg->numEdges();
  EXPECT_EQ(numExplicitEdges, 2 + 1 + 1 + 1 + 1);
  // total edges also includes the outgoing implicit edges from the external node
  EXPECT_EQ(numTotalEdges, 2 + 1 + 4 + 1 + 1 + 3 + 3 + 0 + 3);
}
