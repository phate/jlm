/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static std::unique_ptr<jlm::llvm::aa::PointsToGraph>
RunSteensgaard(jlm::llvm::RvsdgModule & module)
{
  using namespace jlm::llvm;

  aa::Steensgaard steensgaard;
  jlm::util::StatisticsCollector statisticsCollector;
  return steensgaard.Analyze(module, statisticsCollector);
}

static void
assertTargets(
    const jlm::llvm::aa::PointsToGraph::Node & node,
    const std::unordered_set<const jlm::llvm::aa::PointsToGraph::Node *> & targets)
{
  using namespace jlm::llvm::aa;

  assert(node.NumTargets() == targets.size());

  std::unordered_set<const PointsToGraph::Node *> node_targets;
  for (auto & target : node.Targets())
    node_targets.insert(&target);

  assert(targets == node_targets);
}

/**
 * @brief Checks that the set of Memory Nodes escaping the PointsToGraph is exactly equal
 * to the given set of nodes. The external node is included implicitly if omitted.
 * @param ptg the PointsToGraph
 * @param nodes the complete set of nodes that should have escaped
 * @return true if the \p ptg's escaped set is identical to \p nodes, false otherwise
 */
[[nodiscard]] static bool
EscapedIsExactly(
    const jlm::llvm::aa::PointsToGraph & ptg,
    const std::unordered_set<const jlm::llvm::aa::PointsToGraph::MemoryNode *> & nodes)
{
  jlm::util::HashSet hashSet(nodes);
  hashSet.Insert(&ptg.GetExternalMemoryNode());
  return ptg.GetEscapedMemoryNodes() == hashSet;
}

static void
TestStore1()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::StoreTest1 & test)
  {
    assert(ptg.NumAllocaNodes() == 4);
    assert(ptg.NumLambdaNodes() == 1);
    assert(ptg.NumRegisterNodes() == 5);

    auto & alloca_a = ptg.GetAllocaNode(*test.alloca_a);
    auto & alloca_b = ptg.GetAllocaNode(*test.alloca_b);
    auto & alloca_c = ptg.GetAllocaNode(*test.alloca_c);
    auto & alloca_d = ptg.GetAllocaNode(*test.alloca_d);

    auto & palloca_a = ptg.GetRegisterNode(*test.alloca_a->output(0));
    auto & palloca_b = ptg.GetRegisterNode(*test.alloca_b->output(0));
    auto & palloca_c = ptg.GetRegisterNode(*test.alloca_c->output(0));
    auto & palloca_d = ptg.GetRegisterNode(*test.alloca_d->output(0));

    auto & lambda = ptg.GetLambdaNode(*test.lambda);
    auto & plambda = ptg.GetRegisterNode(*test.lambda->output());

    assertTargets(alloca_a, { &alloca_b });
    assertTargets(alloca_b, { &alloca_c });
    assertTargets(alloca_c, { &alloca_d });
    assertTargets(alloca_d, {});

    assertTargets(palloca_a, { &alloca_a });
    assertTargets(palloca_b, { &alloca_b });
    assertTargets(palloca_c, { &alloca_c });
    assertTargets(palloca_d, { &alloca_d });

    assertTargets(lambda, {});
    assertTargets(plambda, { &lambda });

    assert(EscapedIsExactly(ptg, { &lambda }));
  };

  jlm::tests::StoreTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestStore2()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::StoreTest2 & test)
  {
    assert(ptg.NumAllocaNodes() == 5);
    assert(ptg.NumLambdaNodes() == 1);
    assert(ptg.NumRegisterNodes() == 4);

    auto & alloca_a = ptg.GetAllocaNode(*test.alloca_a);
    auto & alloca_b = ptg.GetAllocaNode(*test.alloca_b);
    auto & alloca_x = ptg.GetAllocaNode(*test.alloca_x);
    auto & alloca_y = ptg.GetAllocaNode(*test.alloca_y);
    auto & alloca_p = ptg.GetAllocaNode(*test.alloca_p);

    auto & palloca_a = ptg.GetRegisterNode(*test.alloca_a->output(0));
    auto & palloca_b = ptg.GetRegisterNode(*test.alloca_b->output(0));
    auto & palloca_x = ptg.GetRegisterNode(*test.alloca_x->output(0));
    auto & palloca_y = ptg.GetRegisterNode(*test.alloca_y->output(0));
    auto & palloca_p = ptg.GetRegisterNode(*test.alloca_p->output(0));

    auto & lambda = ptg.GetLambdaNode(*test.lambda);
    auto & plambda = ptg.GetRegisterNode(*test.lambda->output());

    assertTargets(alloca_a, {});
    assertTargets(alloca_b, {});
    assertTargets(alloca_x, { &alloca_a, &alloca_b });
    assertTargets(alloca_y, { &alloca_a, &alloca_b });
    assertTargets(alloca_p, { &alloca_x, &alloca_y });

    assertTargets(palloca_a, { &alloca_a, &alloca_b });
    assertTargets(palloca_b, { &alloca_a, &alloca_b });
    assertTargets(palloca_x, { &alloca_x, &alloca_y });
    assertTargets(palloca_y, { &alloca_x, &alloca_y });
    assertTargets(palloca_p, { &alloca_p });

    assertTargets(lambda, {});
    assertTargets(plambda, { &lambda });

    assert(EscapedIsExactly(ptg, { &lambda }));
  };

  jlm::tests::StoreTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestLoad1()
{
  auto ValidatePointsToGraph =
      [](const jlm::llvm::aa::PointsToGraph & pointsToGraph, const jlm::tests::LoadTest1 & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & loadResult = pointsToGraph.GetRegisterNode(*test.load_p->output(0));

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaOutput = pointsToGraph.GetRegisterNode(*test.lambda->output());
    auto & lambdaArgument0 = pointsToGraph.GetRegisterNode(*test.lambda->GetFunctionArguments()[0]);

    assertTargets(loadResult, { &lambda, &pointsToGraph.GetExternalMemoryNode() });

    assertTargets(lambdaOutput, { &lambda });
    assertTargets(lambdaArgument0, { &lambda, &pointsToGraph.GetExternalMemoryNode() });

    assert(EscapedIsExactly(pointsToGraph, { &lambda }));
  };

  jlm::tests::LoadTest1 test;
  // jlm::rvsdg::view(test.graph()->GetRootRegion(), stdout);
  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestLoad2()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::LoadTest2 & test)
  {
    assert(ptg.NumAllocaNodes() == 5);
    assert(ptg.NumLambdaNodes() == 1);
    assert(ptg.NumRegisterNodes() == 5);

    /*
      We only care about the loads in this test, skipping the validation
      for all other nodes.
    */
    auto & alloca_a = ptg.GetAllocaNode(*test.alloca_a);
    auto & alloca_b = ptg.GetAllocaNode(*test.alloca_b);
    auto & alloca_x = ptg.GetAllocaNode(*test.alloca_x);
    auto & alloca_y = ptg.GetAllocaNode(*test.alloca_y);

    auto & pload_x = ptg.GetRegisterNode(*test.load_x->output(0));
    auto & pload_a = ptg.GetRegisterNode(*test.load_a->output(0));

    auto & lambdaMemoryNode = ptg.GetLambdaNode(*test.lambda);

    assertTargets(pload_x, { &alloca_x, &alloca_y });
    assertTargets(pload_a, { &alloca_a, &alloca_b });

    assert(EscapedIsExactly(ptg, { &lambdaMemoryNode }));
  };

  jlm::tests::LoadTest2 test;
  //	jlm::rvsdg::view(test.graph()->GetRootRegion(), stdout);
  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);

  validate_ptg(*ptg, test);
}

static void
TestLoadFromUndef()
{
  auto ValidatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::LoadFromUndefTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 2);

    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(test.Lambda());
    auto & undefValueNode = pointsToGraph.GetRegisterNode(*test.UndefValueNode()->output(0));

    assertTargets(undefValueNode, {});

    assert(EscapedIsExactly(pointsToGraph, { &lambdaMemoryNode }));
  };

  jlm::tests::LoadFromUndefTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);
  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestGetElementPtr()
{
  auto ValidatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::GetElementPtrTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 2);

    /*
      We only care about the getelemenptr's in this test, skipping the validation
      for all other nodes.
    */
    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & gepX = pointsToGraph.GetRegisterNode(*test.getElementPtrX->output(0));
    auto & gepY = pointsToGraph.GetRegisterNode(*test.getElementPtrY->output(0));

    assertTargets(gepX, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
    assertTargets(gepY, { &lambda, &pointsToGraph.GetExternalMemoryNode() });

    assert(EscapedIsExactly(pointsToGraph, { &lambda }));
  };

  jlm::tests::GetElementPtrTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestBitCast()
{
  auto ValidatePointsToGraph =
      [](const jlm::llvm::aa::PointsToGraph & pointsToGraph, const jlm::tests::BitCastTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 2);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaOut = pointsToGraph.GetRegisterNode(*test.lambda->output());
    auto & lambdaArg = pointsToGraph.GetRegisterNode(*test.lambda->GetFunctionArguments()[0]);

    auto & bitCast = pointsToGraph.GetRegisterNode(*test.bitCast->output(0));

    assertTargets(lambdaOut, { &lambda });
    assertTargets(lambdaArg, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
    assertTargets(bitCast, { &lambda, &pointsToGraph.GetExternalMemoryNode() });

    assert(EscapedIsExactly(pointsToGraph, { &lambda }));
  };

  jlm::tests::BitCastTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestConstantPointerNull()
{
  auto ValidatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::ConstantPointerNullTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaOut = pointsToGraph.GetRegisterNode(*test.lambda->output());
    auto & lambdaArg = pointsToGraph.GetRegisterNode(*test.lambda->GetFunctionArguments()[0]);
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    auto & constantPointerNull =
        pointsToGraph.GetRegisterNode(*test.constantPointerNullNode->output(0));

    assertTargets(lambdaOut, { &lambda });
    assertTargets(lambdaArg, { &lambda, &externalMemoryNode });
    assertTargets(constantPointerNull, { &lambda, &externalMemoryNode });

    assert(EscapedIsExactly(pointsToGraph, { &lambda }));
  };

  jlm::tests::ConstantPointerNullTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestBits2Ptr()
{
  auto validatePointsToGraph =
      [](const jlm::llvm::aa::PointsToGraph & pointsToGraph, const jlm::tests::Bits2PtrTest & test)
  {
    using namespace jlm::llvm::aa;

    assert(pointsToGraph.NumLambdaNodes() == 2);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & lambdaTestMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaTest());
    auto & lambdaBitsToPtrMemoryNode = pointsToGraph.GetLambdaNode(test.GetLambdaBits2Ptr());
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    std::unordered_set<const PointsToGraph::Node *> expectedMemoryNodes(
        { &lambdaTestMemoryNode, &lambdaBitsToPtrMemoryNode, &externalMemoryNode });

    auto & callOutput0 = pointsToGraph.GetRegisterNode(*test.GetCallNode().output(0));
    assertTargets(callOutput0, expectedMemoryNodes);

    auto & bits2ptr = pointsToGraph.GetRegisterNode(*test.GetBitsToPtrNode().output(0));
    assertTargets(bits2ptr, expectedMemoryNodes);

    assert(EscapedIsExactly(pointsToGraph, { &lambdaTestMemoryNode }));
  };

  jlm::tests::Bits2PtrTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph) << std::flush;
  validatePointsToGraph(*pointsToGraph, test);
}

static void
TestCall1()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::CallTest1 & test)
  {
    assert(ptg.NumAllocaNodes() == 3);
    assert(ptg.NumLambdaNodes() == 3);
    assert(ptg.NumRegisterNodes() == 6);

    auto & alloca_x = ptg.GetAllocaNode(*test.alloca_x);
    auto & alloca_y = ptg.GetAllocaNode(*test.alloca_y);
    auto & alloca_z = ptg.GetAllocaNode(*test.alloca_z);

    auto & palloca_x = ptg.GetRegisterNode(*test.alloca_x->output(0));
    auto & palloca_y = ptg.GetRegisterNode(*test.alloca_y->output(0));
    auto & palloca_z = ptg.GetRegisterNode(*test.alloca_z->output(0));

    auto & lambda_f = ptg.GetLambdaNode(*test.lambda_f);
    auto & lambda_g = ptg.GetLambdaNode(*test.lambda_g);
    auto & lambda_h = ptg.GetLambdaNode(*test.lambda_h);

    auto & plambda_f = ptg.GetRegisterNode(*test.lambda_f->output());
    auto & plambda_g = ptg.GetRegisterNode(*test.lambda_g->output());
    auto & plambda_h = ptg.GetRegisterNode(*test.lambda_h->output());

    auto & lambda_f_arg0 = ptg.GetRegisterNode(*test.lambda_f->GetFunctionArguments()[0]);
    auto & lambda_f_arg1 = ptg.GetRegisterNode(*test.lambda_f->GetFunctionArguments()[1]);

    auto & lambda_g_arg0 = ptg.GetRegisterNode(*test.lambda_g->GetFunctionArguments()[0]);
    auto & lambda_g_arg1 = ptg.GetRegisterNode(*test.lambda_g->GetFunctionArguments()[1]);

    auto & lambda_h_cv0 = ptg.GetRegisterNode(*test.lambda_h->GetContextVars()[0].inner);
    auto & lambda_h_cv1 = ptg.GetRegisterNode(*test.lambda_h->GetContextVars()[1].inner);

    assertTargets(palloca_x, { &alloca_x });
    assertTargets(palloca_y, { &alloca_y });
    assertTargets(palloca_z, { &alloca_z });

    assertTargets(plambda_f, { &lambda_f });
    assertTargets(plambda_g, { &lambda_g });
    assertTargets(plambda_h, { &lambda_h });

    assertTargets(lambda_f_arg0, { &alloca_x });
    assertTargets(lambda_f_arg1, { &alloca_y });

    assertTargets(lambda_g_arg0, { &alloca_z });
    assertTargets(lambda_g_arg1, { &alloca_z });

    assertTargets(lambda_h_cv0, { &lambda_f });
    assertTargets(lambda_h_cv1, { &lambda_g });

    assert(EscapedIsExactly(ptg, { &lambda_h }));
  };

  jlm::tests::CallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestCall2()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::CallTest2 & test)
  {
    assert(ptg.NumLambdaNodes() == 3);
    assert(ptg.NumMallocNodes() == 1);
    assert(ptg.NumImportNodes() == 0);
    assert(ptg.NumRegisterNodes() == 4);

    auto & lambda_create = ptg.GetLambdaNode(*test.lambda_create);
    auto & lambda_create_out = ptg.GetRegisterNode(*test.lambda_create->output());

    auto & lambda_destroy = ptg.GetLambdaNode(*test.lambda_destroy);
    auto & lambda_destroy_out = ptg.GetRegisterNode(*test.lambda_destroy->output());
    auto & lambda_destroy_arg =
        ptg.GetRegisterNode(*test.lambda_destroy->GetFunctionArguments()[0]);

    auto & lambda_test = ptg.GetLambdaNode(*test.lambda_test);
    auto & lambda_test_out = ptg.GetRegisterNode(*test.lambda_test->output());
    auto & lambda_test_cv1 = ptg.GetRegisterNode(*test.lambda_test->GetContextVars()[0].inner);
    auto & lambda_test_cv2 = ptg.GetRegisterNode(*test.lambda_test->GetContextVars()[1].inner);

    auto & call_create1_out = ptg.GetRegisterNode(*test.CallCreate1().output(0));
    auto & call_create2_out = ptg.GetRegisterNode(*test.CallCreate2().output(0));

    auto & malloc = ptg.GetMallocNode(*test.malloc);
    auto & malloc_out = ptg.GetRegisterNode(*test.malloc->output(0));

    assertTargets(lambda_create_out, { &lambda_create });

    assertTargets(lambda_destroy_out, { &lambda_destroy });
    assertTargets(lambda_destroy_arg, { &malloc });

    assertTargets(lambda_test_out, { &lambda_test });
    assertTargets(lambda_test_cv1, { &lambda_create });
    assertTargets(lambda_test_cv2, { &lambda_destroy });

    assertTargets(call_create1_out, { &malloc });
    assertTargets(call_create2_out, { &malloc });

    assertTargets(malloc_out, { &malloc });

    assert(EscapedIsExactly(ptg, { &lambda_test }));
  };

  jlm::tests::CallTest2 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestIndirectCall()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::IndirectCallTest1 & test)
  {
    assert(ptg.NumLambdaNodes() == 4);
    assert(ptg.NumImportNodes() == 0);
    assert(ptg.NumRegisterNodes() == 3);

    auto & lambda_three = ptg.GetLambdaNode(test.GetLambdaThree());
    auto & lambda_three_out = ptg.GetRegisterNode(*test.GetLambdaThree().output());

    auto & lambda_four = ptg.GetLambdaNode(test.GetLambdaFour());
    auto & lambda_four_out = ptg.GetRegisterNode(*test.GetLambdaFour().output());

    auto & lambda_indcall = ptg.GetLambdaNode(test.GetLambdaIndcall());
    auto & lambda_indcall_out = ptg.GetRegisterNode(*test.GetLambdaIndcall().output());
    auto & lambda_indcall_arg =
        ptg.GetRegisterNode(*test.GetLambdaIndcall().GetFunctionArguments()[0]);

    auto & lambda_test = ptg.GetLambdaNode(test.GetLambdaTest());
    auto & lambda_test_out = ptg.GetRegisterNode(*test.GetLambdaTest().output());
    auto & lambda_test_cv0 = ptg.GetRegisterNode(*test.GetLambdaTest().GetContextVars()[0].inner);
    auto & lambda_test_cv1 = ptg.GetRegisterNode(*test.GetLambdaTest().GetContextVars()[1].inner);
    auto & lambda_test_cv2 = ptg.GetRegisterNode(*test.GetLambdaTest().GetContextVars()[2].inner);

    assertTargets(lambda_three_out, { &lambda_three, &lambda_four });

    assertTargets(lambda_four_out, { &lambda_three, &lambda_four });

    assertTargets(lambda_indcall_out, { &lambda_indcall });
    assertTargets(lambda_indcall_arg, { &lambda_three, &lambda_four });

    assertTargets(lambda_test_out, { &lambda_test });
    assertTargets(lambda_test_cv0, { &lambda_indcall });
    assertTargets(lambda_test_cv1, { &lambda_three, &lambda_four });
    assertTargets(lambda_test_cv2, { &lambda_three, &lambda_four });

    assert(EscapedIsExactly(ptg, { &lambda_test }));
  };

  jlm::tests::IndirectCallTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestIndirectCall2()
{

  auto validatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::IndirectCallTest2 & test)
  {
    assert(pointsToGraph.NumAllocaNodes() == 3);
    assert(pointsToGraph.NumLambdaNodes() == 7);
    assert(pointsToGraph.NumDeltaNodes() == 2);
    assert(pointsToGraph.NumRegisterNodes() == 10);

    auto & lambdaThree = pointsToGraph.GetLambdaNode(test.GetLambdaThree());
    auto & lambdaThreeOutput = pointsToGraph.GetRegisterNode(*test.GetLambdaThree().output());

    auto & lambdaFour = pointsToGraph.GetLambdaNode(test.GetLambdaFour());
    auto & lambdaFourOutput = pointsToGraph.GetRegisterNode(*test.GetLambdaFour().output());

    assertTargets(lambdaThreeOutput, { &lambdaThree, &lambdaFour });
    assertTargets(lambdaFourOutput, { &lambdaThree, &lambdaFour });
  };

  jlm::tests::IndirectCallTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  validatePointsToGraph(*pointsToGraph, test);
}

static void
TestExternalCall1()
{
  auto validatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::ExternalCallTest1 & test)
  {
    assert(pointsToGraph.NumAllocaNodes() == 2);
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumImportNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 7);

    auto & lambdaF = pointsToGraph.GetLambdaNode(test.LambdaF());
    auto & lambdaFArgument0 =
        pointsToGraph.GetRegisterNode(*test.LambdaF().GetFunctionArguments()[0]);
    auto & lambdaFArgument1 =
        pointsToGraph.GetRegisterNode(*test.LambdaF().GetFunctionArguments()[1]);

    auto & callResult = pointsToGraph.GetRegisterNode(*test.CallG().output(0));

    auto & externalMemory = pointsToGraph.GetExternalMemoryNode();

    assertTargets(lambdaFArgument0, { &lambdaF, &externalMemory });
    assertTargets(lambdaFArgument1, { &lambdaF, &externalMemory });
    assertTargets(callResult, { &lambdaF, &externalMemory });
  };

  jlm::tests::ExternalCallTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph) << std::flush;

  validatePointsToGraph(*pointsToGraph, test);
}

static void
TestExternalCall2()
{
  // Arrange
  jlm::tests::ExternalCallTest2 test;
  std::unordered_map<const jlm::rvsdg::Output *, std::string> outputMap;
  std::cout << jlm::rvsdg::view(&test.graph().GetRootRegion(), outputMap) << std::flush;

  // Act
  auto pointsToGraph = RunSteensgaard(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph, outputMap) << std::flush;

  // Assert
  assert(pointsToGraph->NumAllocaNodes() == 1);
  assert(pointsToGraph->NumLambdaNodes() == 1);
  assert(pointsToGraph->NumImportNodes() == 3);
  assert(pointsToGraph->NumRegisterNodes() == 7);

  for (auto & registerNode : pointsToGraph->RegisterNodes())
  {
    assert(registerNode.NumTargets() != 0);
  }
}

static void
TestGamma()
{
  auto ValidatePointsToGraph =
      [](const jlm::llvm::aa::PointsToGraph & pointsToGraph, const jlm::tests::GammaTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);

    for (size_t n = 1; n < 5; n++)
    {
      auto & lambdaArgument =
          pointsToGraph.GetRegisterNode(*test.lambda->GetFunctionArguments()[n]);
      assertTargets(lambdaArgument, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
    }

    auto entryvars = test.gamma->GetEntryVars();
    assert(entryvars.size() == 4);
    for (const auto & entryvar : entryvars)
    {
      auto & argument0 = pointsToGraph.GetRegisterNode(*entryvar.branchArgument[0]);
      auto & argument1 = pointsToGraph.GetRegisterNode(*entryvar.branchArgument[1]);

      assertTargets(argument0, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
      assertTargets(argument1, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
    }

    for (size_t n = 0; n < 4; n++)
    {
      auto & gammaOutput = pointsToGraph.GetRegisterNode(*test.gamma->GetExitVars()[0].output);
      assertTargets(gammaOutput, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
    }

    assert(EscapedIsExactly(pointsToGraph, { &lambda }));
  };

  jlm::tests::GammaTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestTheta()
{
  auto ValidatePointsToGraph =
      [](const jlm::llvm::aa::PointsToGraph & pointsToGraph, const jlm::tests::ThetaTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 2);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaArgument1 = pointsToGraph.GetRegisterNode(*test.lambda->GetFunctionArguments()[1]);
    auto & lambdaOutput = pointsToGraph.GetRegisterNode(*test.lambda->output());

    auto & gepOutput = pointsToGraph.GetRegisterNode(*test.gep->output(0));

    auto & thetaArgument2 = pointsToGraph.GetRegisterNode(*test.theta->GetLoopVars()[2].pre);
    auto & thetaOutput2 = pointsToGraph.GetRegisterNode(*test.theta->output(2));

    assertTargets(lambdaArgument1, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
    assertTargets(lambdaOutput, { &lambda });

    assertTargets(gepOutput, { &lambda, &pointsToGraph.GetExternalMemoryNode() });

    assertTargets(thetaArgument2, { &lambda, &pointsToGraph.GetExternalMemoryNode() });
    assertTargets(thetaOutput2, { &lambda, &pointsToGraph.GetExternalMemoryNode() });

    assert(EscapedIsExactly(pointsToGraph, { &lambda }));
  };

  jlm::tests::ThetaTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestDelta1()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::DeltaTest1 & test)
  {
    assert(ptg.NumDeltaNodes() == 1);
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumRegisterNodes() == 3);

    auto & delta_f = ptg.GetDeltaNode(*test.delta_f);
    auto & pdelta_f = ptg.GetRegisterNode(*test.delta_f->output());

    auto & lambda_g = ptg.GetLambdaNode(*test.lambda_g);
    auto & plambda_g = ptg.GetRegisterNode(*test.lambda_g->output());
    auto & lambda_g_arg0 = ptg.GetRegisterNode(*test.lambda_g->GetFunctionArguments()[0]);

    auto & lambda_h = ptg.GetLambdaNode(*test.lambda_h);
    auto & plambda_h = ptg.GetRegisterNode(*test.lambda_h->output());
    auto & lambda_h_cv0 = ptg.GetRegisterNode(*test.lambda_h->GetContextVars()[0].inner);
    auto & lambda_h_cv1 = ptg.GetRegisterNode(*test.lambda_h->GetContextVars()[1].inner);

    assertTargets(pdelta_f, { &delta_f });

    assertTargets(plambda_g, { &lambda_g });
    assertTargets(plambda_h, { &lambda_h });

    assertTargets(lambda_g_arg0, { &delta_f });

    assertTargets(lambda_h_cv0, { &delta_f });
    assertTargets(lambda_h_cv1, { &lambda_g });

    assert(EscapedIsExactly(ptg, { &lambda_h }));
  };

  jlm::tests::DeltaTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestDelta2()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::DeltaTest2 & test)
  {
    assert(ptg.NumDeltaNodes() == 2);
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumRegisterNodes() == 4);

    auto & delta_d1 = ptg.GetDeltaNode(*test.delta_d1);
    auto & delta_d1_out = ptg.GetRegisterNode(*test.delta_d1->output());

    auto & delta_d2 = ptg.GetDeltaNode(*test.delta_d2);
    auto & delta_d2_out = ptg.GetRegisterNode(*test.delta_d2->output());

    auto & lambda_f1 = ptg.GetLambdaNode(*test.lambda_f1);
    auto & lambda_f1_out = ptg.GetRegisterNode(*test.lambda_f1->output());
    auto & lambda_f1_cvd1 = ptg.GetRegisterNode(*test.lambda_f1->GetContextVars()[0].inner);

    auto & lambda_f2 = ptg.GetLambdaNode(*test.lambda_f2);
    auto & lambda_f2_out = ptg.GetRegisterNode(*test.lambda_f2->output());
    auto & lambda_f2_cvd1 = ptg.GetRegisterNode(*test.lambda_f2->GetContextVars()[0].inner);
    auto & lambda_f2_cvd2 = ptg.GetRegisterNode(*test.lambda_f2->GetContextVars()[1].inner);
    auto & lambda_f2_cvf1 = ptg.GetRegisterNode(*test.lambda_f2->GetContextVars()[2].inner);

    assertTargets(delta_d1_out, { &delta_d1 });
    assertTargets(delta_d2_out, { &delta_d2 });

    assertTargets(lambda_f1_out, { &lambda_f1 });
    assertTargets(lambda_f1_cvd1, { &delta_d1 });

    assertTargets(lambda_f2_out, { &lambda_f2 });
    assertTargets(lambda_f2_cvd1, { &delta_d1 });
    assertTargets(lambda_f2_cvd2, { &delta_d2 });
    assertTargets(lambda_f2_cvf1, { &lambda_f1 });

    assert(EscapedIsExactly(ptg, { &lambda_f2 }));
  };

  jlm::tests::DeltaTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*ptg);
  validate_ptg(*ptg, test);
}

static void
TestImports()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::ImportTest & test)
  {
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumImportNodes() == 2);
    assert(ptg.NumRegisterNodes() == 4);

    auto & d1 = ptg.GetImportNode(*test.import_d1);
    auto & import_d1 = ptg.GetRegisterNode(*test.import_d1);

    auto & d2 = ptg.GetImportNode(*test.import_d2);
    auto & import_d2 = ptg.GetRegisterNode(*test.import_d2);

    auto & lambda_f1 = ptg.GetLambdaNode(*test.lambda_f1);
    auto & lambda_f1_out = ptg.GetRegisterNode(*test.lambda_f1->output());
    auto & lambda_f1_cvd1 = ptg.GetRegisterNode(*test.lambda_f1->GetContextVars()[0].inner);

    auto & lambda_f2 = ptg.GetLambdaNode(*test.lambda_f2);
    auto & lambda_f2_out = ptg.GetRegisterNode(*test.lambda_f2->output());
    auto & lambda_f2_cvd1 = ptg.GetRegisterNode(*test.lambda_f2->GetContextVars()[0].inner);
    auto & lambda_f2_cvd2 = ptg.GetRegisterNode(*test.lambda_f2->GetContextVars()[1].inner);
    auto & lambda_f2_cvf1 = ptg.GetRegisterNode(*test.lambda_f2->GetContextVars()[2].inner);

    assertTargets(import_d1, { &d1 });
    assertTargets(import_d2, { &d2 });

    assertTargets(lambda_f1_out, { &lambda_f1 });
    assertTargets(lambda_f1_cvd1, { &d1 });

    assertTargets(lambda_f2_out, { &lambda_f2 });
    assertTargets(lambda_f2_cvd1, { &d1 });
    assertTargets(lambda_f2_cvd2, { &d2 });
    assertTargets(lambda_f2_cvf1, { &lambda_f1 });

    assert(EscapedIsExactly(ptg, { &lambda_f2 }));
  };

  jlm::tests::ImportTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*ptg);
  validate_ptg(*ptg, test);
}

static void
TestPhi1()
{
  auto validate_ptg =
      [](const jlm::llvm::aa::PointsToGraph & ptg, const jlm::tests::PhiTest1 & test)
  {
    assert(ptg.NumAllocaNodes() == 1);
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumRegisterNodes() == 3);

    auto & lambda_fib = ptg.GetLambdaNode(*test.lambda_fib);
    auto & lambda_fib_out = ptg.GetRegisterNode(*test.lambda_fib->output());
    auto & lambda_fib_arg1 = ptg.GetRegisterNode(*test.lambda_fib->GetFunctionArguments()[1]);

    auto & lambda_test = ptg.GetLambdaNode(*test.lambda_test);
    auto & lambda_test_out = ptg.GetRegisterNode(*test.lambda_test->output());

    auto & phi_rv = ptg.GetRegisterNode(*test.phi->GetFixVars()[0].output);
    auto & phi_rv_arg = ptg.GetRegisterNode(*test.phi->GetFixVars()[0].recref);

    auto & gamma_result = ptg.GetRegisterNode(*test.gamma->GetEntryVars()[1].branchArgument[0]);
    auto & gamma_fib = ptg.GetRegisterNode(*test.gamma->GetEntryVars()[2].branchArgument[0]);

    auto & alloca = ptg.GetAllocaNode(*test.alloca);
    auto & alloca_out = ptg.GetRegisterNode(*test.alloca->output(0));

    assertTargets(lambda_fib_out, { &lambda_fib });
    assertTargets(lambda_fib_arg1, { &alloca });

    assertTargets(lambda_test_out, { &lambda_test });

    assertTargets(phi_rv, { &lambda_fib });
    assertTargets(phi_rv_arg, { &lambda_fib });

    assertTargets(gamma_result, { &alloca });
    assertTargets(gamma_fib, { &lambda_fib });

    assertTargets(alloca_out, { &alloca });

    assert(EscapedIsExactly(ptg, { &lambda_test }));
  };

  jlm::tests::PhiTest1 test;
  //	jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto ptg = RunSteensgaard(test.module());
  //	std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestExternalMemory()
{
  auto ValidatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::ExternalMemoryTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & lambdaF = pointsToGraph.GetLambdaNode(*test.LambdaF);
    auto & lambdaFArgument0 =
        pointsToGraph.GetRegisterNode(*test.LambdaF->GetFunctionArguments()[0]);
    auto & lambdaFArgument1 =
        pointsToGraph.GetRegisterNode(*test.LambdaF->GetFunctionArguments()[1]);

    assertTargets(lambdaFArgument0, { &lambdaF, &pointsToGraph.GetExternalMemoryNode() });
    assertTargets(lambdaFArgument1, { &lambdaF, &pointsToGraph.GetExternalMemoryNode() });

    assert(EscapedIsExactly(pointsToGraph, { &lambdaF }));
  };

  jlm::tests::ExternalMemoryTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestEscapedMemory1()
{
  auto ValidatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::EscapedMemoryTest1 & test)
  {
    assert(pointsToGraph.NumDeltaNodes() == 4);
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 7);

    auto & lambdaTestArgument0 =
        pointsToGraph.GetRegisterNode(*test.LambdaTest->GetFunctionArguments()[0]);
    auto & lambdaTestCv0 =
        pointsToGraph.GetRegisterNode(*test.LambdaTest->GetContextVars()[0].inner);
    auto & loadNode1Output = pointsToGraph.GetRegisterNode(*test.LoadNode1->output(0));

    auto deltaA = &pointsToGraph.GetDeltaNode(*test.DeltaA);
    auto deltaB = &pointsToGraph.GetDeltaNode(*test.DeltaB);
    auto deltaX = &pointsToGraph.GetDeltaNode(*test.DeltaX);
    auto deltaY = &pointsToGraph.GetDeltaNode(*test.DeltaY);
    auto lambdaTest = &pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto externalMemory = &pointsToGraph.GetExternalMemoryNode();

    assertTargets(lambdaTestArgument0, { deltaA, deltaX, deltaY, lambdaTest, externalMemory });
    assertTargets(lambdaTestCv0, { deltaB });
    assertTargets(loadNode1Output, { deltaA, deltaX, deltaY, lambdaTest, externalMemory });

    assert(EscapedIsExactly(pointsToGraph, { lambdaTest, deltaA, deltaX, deltaY }));
  };

  jlm::tests::EscapedMemoryTest1 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestEscapedMemory2()
{
  auto ValidatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::EscapedMemoryTest2 & test)
  {
    assert(pointsToGraph.NumImportNodes() == 2);
    assert(pointsToGraph.NumLambdaNodes() == 3);
    assert(pointsToGraph.NumMallocNodes() == 2);
    assert(pointsToGraph.NumRegisterNodes() == 8);

    auto returnAddressFunction = &pointsToGraph.GetLambdaNode(*test.ReturnAddressFunction);
    auto callExternalFunction1 = &pointsToGraph.GetLambdaNode(*test.CallExternalFunction1);
    auto callExternalFunction2 = &pointsToGraph.GetLambdaNode(*test.CallExternalFunction2);
    auto returnAddressMalloc = &pointsToGraph.GetMallocNode(*test.ReturnAddressMalloc);
    auto callExternalFunction1Malloc =
        &pointsToGraph.GetMallocNode(*test.CallExternalFunction1Malloc);
    auto externalMemory = &pointsToGraph.GetExternalMemoryNode();

    auto & externalFunction2CallResult =
        pointsToGraph.GetRegisterNode(*test.ExternalFunction2Call->output(0));

    assertTargets(
        externalFunction2CallResult,
        { returnAddressFunction,
          callExternalFunction1,
          callExternalFunction2,
          externalMemory,
          returnAddressMalloc,
          callExternalFunction1Malloc });

    assert(EscapedIsExactly(
        pointsToGraph,
        { returnAddressFunction,
          callExternalFunction1,
          callExternalFunction2,
          returnAddressMalloc,
          callExternalFunction1Malloc }));
  };

  jlm::tests::EscapedMemoryTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestEscapedMemory3()
{
  auto ValidatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::EscapedMemoryTest3 & test)
  {
    assert(pointsToGraph.NumDeltaNodes() == 1);
    assert(pointsToGraph.NumImportNodes() == 1);
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 4);

    auto lambdaTest = &pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto deltaGlobal = &pointsToGraph.GetDeltaNode(*test.DeltaGlobal);
    auto externalMemory = &pointsToGraph.GetExternalMemoryNode();

    auto & callExternalFunctionResult =
        pointsToGraph.GetRegisterNode(*test.CallExternalFunction->output(0));

    assertTargets(callExternalFunctionResult, { lambdaTest, deltaGlobal, externalMemory });

    assert(EscapedIsExactly(pointsToGraph, { lambdaTest, deltaGlobal }));
  };

  jlm::tests::EscapedMemoryTest3 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestMemcpy()
{
  /*
   * Arrange
   */
  auto ValidatePointsToGraph =
      [](const jlm::llvm::aa::PointsToGraph & pointsToGraph, const jlm::tests::MemcpyTest & test)
  {
    assert(pointsToGraph.NumDeltaNodes() == 2);
    assert(pointsToGraph.NumLambdaNodes() == 2);
    assert(pointsToGraph.NumRegisterNodes() == 4);

    auto localArray = &pointsToGraph.GetDeltaNode(test.LocalArray());
    auto globalArray = &pointsToGraph.GetDeltaNode(test.GlobalArray());

    auto & memCpyDest = pointsToGraph.GetRegisterNode(*test.Memcpy().input(0)->origin());
    auto & memCpySrc = pointsToGraph.GetRegisterNode(*test.Memcpy().input(1)->origin());

    auto lambdaF = &pointsToGraph.GetLambdaNode(test.LambdaF());
    auto lambdaG = &pointsToGraph.GetLambdaNode(test.LambdaG());

    assertTargets(memCpyDest, { globalArray, localArray });
    assertTargets(memCpySrc, { localArray, globalArray });

    assert(EscapedIsExactly(pointsToGraph, { globalArray, localArray, lambdaF, lambdaG }));
  };

  jlm::tests::MemcpyTest test;
  std::unordered_map<const jlm::rvsdg::Output *, std::string> outputMap;
  std::cout << jlm::rvsdg::view(&test.graph().GetRootRegion(), outputMap) << std::flush;

  /*
   * Act
   */
  auto pointsToGraph = RunSteensgaard(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph, outputMap);

  /*
   * Assert
   */
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestMemcpy2()
{
  // Arrange
  jlm::tests::MemcpyTest2 test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  // Act
  auto pointsToGraph = RunSteensgaard(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph) << std::flush;

  // Assert
  assert(pointsToGraph->NumRegisterNodes() == 8);
  assert(pointsToGraph->NumLambdaNodes() == 2);

  auto & lambdaFNode = pointsToGraph->GetLambdaNode(test.LambdaF());
  auto & lambdaFArgument0 =
      pointsToGraph->GetRegisterNode(*test.LambdaF().GetFunctionArguments()[0]);
  auto & lambdaFArgument1 =
      pointsToGraph->GetRegisterNode(*test.LambdaF().GetFunctionArguments()[1]);

  auto & lambdaGArgument0 =
      pointsToGraph->GetRegisterNode(*test.LambdaG().GetFunctionArguments()[0]);
  auto & lambdaGArgument1 =
      pointsToGraph->GetRegisterNode(*test.LambdaG().GetFunctionArguments()[1]);

  auto & memcpyOperand0 = pointsToGraph->GetRegisterNode(*test.Memcpy().input(0)->origin());
  auto & memcpyOperand1 = pointsToGraph->GetRegisterNode(*test.Memcpy().input(1)->origin());

  auto & externalMemoryNode = pointsToGraph->GetExternalMemoryNode();

  assertTargets(lambdaFArgument0, { &lambdaFNode, &externalMemoryNode });
  assertTargets(lambdaFArgument1, { &lambdaFNode, &externalMemoryNode });

  assertTargets(lambdaGArgument0, { &lambdaFNode, &externalMemoryNode });
  assertTargets(lambdaGArgument1, { &lambdaFNode, &externalMemoryNode });

  assertTargets(memcpyOperand0, { &lambdaFNode, &externalMemoryNode });
  assertTargets(memcpyOperand1, { &lambdaFNode, &externalMemoryNode });
}

static void
TestMemcpy3()
{
  // Arrange
  jlm::tests::MemcpyTest3 test;
  std::unordered_map<const jlm::rvsdg::Output *, std::string> outputMap;
  std::cout << jlm::rvsdg::view(&test.graph().GetRootRegion(), outputMap) << std::flush;

  // Act
  auto pointsToGraph = RunSteensgaard(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph, outputMap) << std::flush;

  // Assert
  assert(pointsToGraph->NumRegisterNodes() == 4);
  assert(pointsToGraph->NumLambdaNodes() == 1);
  assert(pointsToGraph->NumAllocaNodes() == 1);

  auto & lambdaNode = pointsToGraph->GetLambdaNode(test.Lambda());
  auto & lambdaArgument0 = pointsToGraph->GetRegisterNode(*test.Lambda().GetFunctionArguments()[0]);

  auto & allocaNode = pointsToGraph->GetAllocaNode(test.Alloca());

  auto & memcpyOperand0 = pointsToGraph->GetRegisterNode(*test.Memcpy().input(0)->origin());
  auto & memcpyOperand1 = pointsToGraph->GetRegisterNode(*test.Memcpy().input(1)->origin());

  auto & externalMemoryNode = pointsToGraph->GetExternalMemoryNode();

  assertTargets(lambdaArgument0, { &lambdaNode, &allocaNode, &externalMemoryNode });

  assertTargets(memcpyOperand0, { &lambdaNode, &allocaNode, &externalMemoryNode });
  assertTargets(memcpyOperand1, { &allocaNode });
}

static void
TestLinkedList()
{
  auto validatePointsToGraph = [](const jlm::llvm::aa::PointsToGraph & pointsToGraph,
                                  const jlm::tests::LinkedListTest & test)
  {
    assert(pointsToGraph.NumAllocaNodes() == 1);
    assert(pointsToGraph.NumDeltaNodes() == 1);
    assert(pointsToGraph.NumLambdaNodes() == 1);

    auto & allocaNode = pointsToGraph.GetAllocaNode(test.GetAlloca());
    auto & deltaMyListNode = pointsToGraph.GetDeltaNode(test.GetDeltaMyList());
    auto & lambdaNextNode = pointsToGraph.GetLambdaNode(test.GetLambdaNext());
    auto & externalMemoryNode = pointsToGraph.GetExternalMemoryNode();

    std::unordered_set<const jlm::llvm::aa::PointsToGraph::Node *> expectedMemoryNodes = {
      &allocaNode,
      &deltaMyListNode,
      &lambdaNextNode,
      &externalMemoryNode
    };

    assertTargets(allocaNode, expectedMemoryNodes);
    assertTargets(deltaMyListNode, expectedMemoryNodes);
  };

  jlm::tests::LinkedListTest test;
  // jlm::rvsdg::view(test.graph().GetRootRegion(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  validatePointsToGraph(*pointsToGraph, test);
}

static void
TestLambdaCallArgumentMismatch()
{
  // Arrange and Act
  jlm::tests::LambdaCallArgumentMismatch test;
  auto pointsToGraph = RunSteensgaard(test.module());

  // Assert
  assert(pointsToGraph->NumAllocaNodes() == 1);
  assert(pointsToGraph->NumLambdaNodes() == 2);
  assert(pointsToGraph->NumRegisterNodes() == 3);
}

static void
TestVariadicFunction1()
{
  // Arrange and Act
  jlm::tests::VariadicFunctionTest1 test;
  auto pointsToGraph = RunSteensgaard(test.module());

  // Assert
  assert(pointsToGraph->NumAllocaNodes() == 1);
  assert(pointsToGraph->NumLambdaNodes() == 2);
  assert(pointsToGraph->NumImportNodes() == 1);
  assert(pointsToGraph->NumRegisterNodes() == 5);

  auto & allocaMemoryNode = pointsToGraph->GetAllocaNode(test.GetAllocaNode());
  auto & externalMemoryNode = pointsToGraph->GetExternalMemoryNode();

  auto & callOutput = pointsToGraph->GetRegisterNode(*test.GetCallH().output(0));

  assert(EscapedIsExactly(*pointsToGraph, { &allocaMemoryNode }));

  assertTargets(callOutput, { &allocaMemoryNode, &externalMemoryNode });
}

static void
TestVariadicFunction2()
{
  std::unordered_map<const jlm::rvsdg::Output *, std::string> outputMap;

  // Arrange
  jlm::tests::VariadicFunctionTest2 test;
  std::cout << jlm::rvsdg::view(&test.module().Rvsdg().GetRootRegion(), outputMap) << std::flush;

  // Act
  auto pointsToGraph = RunSteensgaard(test.module());
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph, outputMap);

  // Assert
  assert(pointsToGraph->NumAllocaNodes() == 1);
  assert(pointsToGraph->NumLambdaNodes() == 2);
  assert(pointsToGraph->NumImportNodes() == 4);
  assert(pointsToGraph->NumRegisterNodes() == 8);

  auto & allocaMemoryNode = pointsToGraph->GetAllocaNode(test.GetAllocaNode());

  assert(EscapedIsExactly(*pointsToGraph, { &allocaMemoryNode }));
}

static void
TestStatistics()
{
  // Arrange
  jlm::tests::LoadTest1 test;

  jlm::util::StatisticsCollectorSettings statisticsCollectorSettings(
      { jlm::util::Statistics::Id::SteensgaardAnalysis });
  jlm::util::StatisticsCollector statisticsCollector(statisticsCollectorSettings);

  // Act
  jlm::llvm::aa::Steensgaard steensgaard;
  steensgaard.Analyze(test.module(), statisticsCollector);

  // Assert
  assert(statisticsCollector.NumCollectedStatistics() == 1);
}

static int
TestSteensgaardAnalysis()
{
  TestStore1();
  TestStore2();

  TestLoad1();
  TestLoad2();
  TestLoadFromUndef();

  TestGetElementPtr();

  TestBitCast();
  TestBits2Ptr();

  TestConstantPointerNull();

  TestCall1();
  TestCall2();

  TestIndirectCall();
  TestIndirectCall2();

  TestExternalCall1();
  TestExternalCall2();

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
  TestMemcpy2();

  TestMemcpy3();

  TestLinkedList();

  TestLambdaCallArgumentMismatch();

  TestVariadicFunction1();
  TestVariadicFunction2();

  TestStatistics();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestSteensgaard", TestSteensgaardAnalysis)
