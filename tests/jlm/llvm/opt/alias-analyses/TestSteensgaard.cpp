/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/util/Statistics.hpp>

static std::unique_ptr<jlm::aa::PointsToGraph>
RunSteensgaard(jlm::RvsdgModule & module)
{
  using namespace jlm;

  aa::Steensgaard steensgaard;
  StatisticsCollector statisticsCollector;
  return steensgaard.Analyze(module, statisticsCollector);
}

static void
assertTargets(
  const jlm::aa::PointsToGraph::Node & node,
  const std::unordered_set<const jlm::aa::PointsToGraph::Node*> & targets)
{
  using namespace jlm::aa;

  assert(node.NumTargets() == targets.size());

  std::unordered_set<const PointsToGraph::Node*> node_targets;
  for (auto & target : node.Targets())
    node_targets.insert(&target);

  assert(targets == node_targets);
}

static void
TestStore1()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const StoreTest1 & test)
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

    assertTargets(alloca_a, {&alloca_b});
    assertTargets(alloca_b, {&alloca_c});
    assertTargets(alloca_c, {&alloca_d});
    assertTargets(alloca_d, {});

    assertTargets(palloca_a, {&alloca_a});
    assertTargets(palloca_b, {&alloca_b});
    assertTargets(palloca_c, {&alloca_c});
    assertTargets(palloca_d, {&alloca_d});

    assertTargets(lambda, {});
    assertTargets(plambda, {&lambda});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  StoreTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestStore2()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const StoreTest2 & test)
  {
    assert(ptg.NumAllocaNodes() == 5);
    assert(ptg.NumLambdaNodes() == 1);
    assert(ptg.NumRegisterNodes() == 6);

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
    assertTargets(alloca_x, {&alloca_a, &alloca_b});
    assertTargets(alloca_y, {&alloca_a, &alloca_b});
    assertTargets(alloca_p, {&alloca_x, &alloca_y});

    assertTargets(palloca_a, {&alloca_a, &alloca_b});
    assertTargets(palloca_b, {&alloca_a, &alloca_b});
    assertTargets(palloca_x, {&alloca_x, &alloca_y});
    assertTargets(palloca_y, {&alloca_x, &alloca_y});
    assertTargets(palloca_p, {&alloca_p});

    assertTargets(lambda, {});
    assertTargets(plambda, {&lambda});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  StoreTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestLoad1()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const LoadTest1 & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & loadResult = pointsToGraph.GetRegisterNode(*test.load_p->output(0));

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaOutput = pointsToGraph.GetRegisterNode(*test.lambda->output());
    auto & lambdaArgument0 = pointsToGraph.GetRegisterNode(*test.lambda->fctargument(0));

    assertTargets(loadResult, {&lambda, &pointsToGraph.GetExternalMemoryNode()});

    assertTargets(lambdaOutput, {&lambda});
    assertTargets(lambdaArgument0, {&lambda, &pointsToGraph.GetExternalMemoryNode()});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  LoadTest1 test;
  // jive::view(test.graph()->root(), stdout);
  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestLoad2()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const LoadTest2 & test)
  {
    assert(ptg.NumAllocaNodes() == 5);
    assert(ptg.NumLambdaNodes() == 1);
    assert(ptg.NumRegisterNodes() == 8);

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

    assertTargets(pload_x, {&alloca_x, &alloca_y});
    assertTargets(pload_a, {&alloca_a, &alloca_b});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambdaMemoryNode});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  LoadTest2 test;
//	jive::view(test.graph()->root(), stdout);
  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);

  validate_ptg(*ptg, test);
}

static void
TestLoadFromUndef()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const LoadFromUndefTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 2);

    auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(test.Lambda());
    auto & undefValueNode = pointsToGraph.GetRegisterNode(*test.UndefValueNode()->output(0));

    assertTargets(undefValueNode, {});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambdaMemoryNode});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  LoadFromUndefTest test;
  // jive::view(test.graph().root(), stdout);
  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestGetElementPtr()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const GetElementPtrTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 4);

    /*
      We only care about the getelemenptr's in this test, skipping the validation
      for all other nodes.
    */
    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & gepX = pointsToGraph.GetRegisterNode(*test.getElementPtrX->output(0));
    auto & gepY = pointsToGraph.GetRegisterNode(*test.getElementPtrY->output(0));

    assertTargets(gepX, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    assertTargets(gepY, {&lambda, &pointsToGraph.GetExternalMemoryNode()});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  GetElementPtrTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestBitCast()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const BitCastTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaOut = pointsToGraph.GetRegisterNode(*test.lambda->output());
    auto & lambdaArg = pointsToGraph.GetRegisterNode(*test.lambda->fctargument(0));

    auto & bitCast = pointsToGraph.GetRegisterNode(*test.bitCast->output(0));

    assertTargets(lambdaOut, {&lambda});
    assertTargets(lambdaArg, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    assertTargets(bitCast, {&lambda, &pointsToGraph.GetExternalMemoryNode()});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  BitCastTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestConstantPointerNull()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const ConstantPointerNullTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaOut = pointsToGraph.GetRegisterNode(*test.lambda->output());
    auto & lambdaArg = pointsToGraph.GetRegisterNode(*test.lambda->fctargument(0));

    auto & constantPointerNull = pointsToGraph.GetRegisterNode(*test.constantPointerNullNode->output(0));

    assertTargets(lambdaOut, {&lambda});
    assertTargets(lambdaArg, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    assertTargets(constantPointerNull, {});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  ConstantPointerNullTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestBits2Ptr()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const Bits2PtrTest & test)
  {
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumRegisterNodes() == 5);

    auto & call_out0 = ptg.GetRegisterNode(*test.call->output(0));
    assertTargets(call_out0, {&ptg.GetUnknownMemoryNode(), &ptg.GetExternalMemoryNode()});

    auto & bits2ptr = ptg.GetRegisterNode(*test.call->output(0));
    assertTargets(bits2ptr, {&ptg.GetUnknownMemoryNode(), &ptg.GetExternalMemoryNode()});

    auto & lambdaTestMemoryNode = ptg.GetLambdaNode(*test.lambda_test);

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambdaTestMemoryNode});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  Bits2PtrTest test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestCall1()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const CallTest1 & test)
  {
    assert(ptg.NumAllocaNodes() == 3);
    assert(ptg.NumLambdaNodes() == 3);
    assert(ptg.NumRegisterNodes() == 12);

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

    auto & lambda_f_arg0 = ptg.GetRegisterNode(*test.lambda_f->fctargument(0));
    auto & lambda_f_arg1 = ptg.GetRegisterNode(*test.lambda_f->fctargument(1));

    auto & lambda_g_arg0 = ptg.GetRegisterNode(*test.lambda_g->fctargument(0));
    auto & lambda_g_arg1 = ptg.GetRegisterNode(*test.lambda_g->fctargument(1));

    auto & lambda_h_cv0 = ptg.GetRegisterNode(*test.lambda_h->cvargument(0));
    auto & lambda_h_cv1 = ptg.GetRegisterNode(*test.lambda_h->cvargument(1));

    assertTargets(palloca_x, {&alloca_x});
    assertTargets(palloca_y, {&alloca_y});
    assertTargets(palloca_z, {&alloca_z});

    assertTargets(plambda_f, {&lambda_f});
    assertTargets(plambda_g, {&lambda_g});
    assertTargets(plambda_h, {&lambda_h});

    assertTargets(lambda_f_arg0, {&alloca_x});
    assertTargets(lambda_f_arg1, {&alloca_y});

    assertTargets(lambda_g_arg0, {&alloca_z});
    assertTargets(lambda_g_arg1, {&alloca_z});

    assertTargets(lambda_h_cv0, {&lambda_f});
    assertTargets(lambda_h_cv1, {&lambda_g});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda_h});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  CallTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestCall2()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const CallTest2 & test)
  {
    assert(ptg.NumLambdaNodes() == 3);
    assert(ptg.NumMallocNodes() == 1);
    assert(ptg.NumImportNodes() == 0);
    assert(ptg.NumRegisterNodes() == 11);

    auto & lambda_create = ptg.GetLambdaNode(*test.lambda_create);
    auto & lambda_create_out = ptg.GetRegisterNode(*test.lambda_create->output());

    auto & lambda_destroy = ptg.GetLambdaNode(*test.lambda_destroy);
    auto & lambda_destroy_out = ptg.GetRegisterNode(*test.lambda_destroy->output());
    auto & lambda_destroy_arg = ptg.GetRegisterNode(*test.lambda_destroy->fctargument(0));

    auto & lambda_test = ptg.GetLambdaNode(*test.lambda_test);
    auto & lambda_test_out = ptg.GetRegisterNode(*test.lambda_test->output());
    auto & lambda_test_cv1 = ptg.GetRegisterNode(*test.lambda_test->cvargument(0));
    auto & lambda_test_cv2 = ptg.GetRegisterNode(*test.lambda_test->cvargument(1));

    auto & call_create1_out = ptg.GetRegisterNode(*test.CallCreate1().output(0));
    auto & call_create2_out = ptg.GetRegisterNode(*test.CallCreate2().output(0));

    auto & malloc = ptg.GetMallocNode(*test.malloc);
    auto & malloc_out = ptg.GetRegisterNode(*test.malloc->output(0));

    assertTargets(lambda_create_out, {&lambda_create});

    assertTargets(lambda_destroy_out, {&lambda_destroy});
    assertTargets(lambda_destroy_arg, {&malloc});

    assertTargets(lambda_test_out, {&lambda_test});
    assertTargets(lambda_test_cv1, {&lambda_create});
    assertTargets(lambda_test_cv2, {&lambda_destroy});

    assertTargets(call_create1_out, {&malloc});
    assertTargets(call_create2_out, {&malloc});

    assertTargets(malloc_out, {&malloc});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda_test});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  CallTest2 test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestIndirectCall()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const IndirectCallTest1 & test)
  {
    assert(ptg.NumLambdaNodes() == 4);
    assert(ptg.NumImportNodes() == 0);
    assert(ptg.NumRegisterNodes() == 8);

    auto & lambda_three = ptg.GetLambdaNode(test.GetLambdaThree());
    auto & lambda_three_out = ptg.GetRegisterNode(*test.GetLambdaThree().output());

    auto & lambda_four = ptg.GetLambdaNode(test.GetLambdaFour());
    auto & lambda_four_out = ptg.GetRegisterNode(*test.GetLambdaFour().output());

    auto & lambda_indcall = ptg.GetLambdaNode(test.GetLambdaIndcall());
    auto & lambda_indcall_out = ptg.GetRegisterNode(*test.GetLambdaIndcall().output());
    auto & lambda_indcall_arg = ptg.GetRegisterNode(*test.GetLambdaIndcall().fctargument(0));

    auto & lambda_test = ptg.GetLambdaNode(test.GetLambdaTest());
    auto & lambda_test_out = ptg.GetRegisterNode(*test.GetLambdaTest().output());
    auto & lambda_test_cv0 = ptg.GetRegisterNode(*test.GetLambdaTest().cvargument(0));
    auto & lambda_test_cv1 = ptg.GetRegisterNode(*test.GetLambdaTest().cvargument(1));
    auto & lambda_test_cv2 = ptg.GetRegisterNode(*test.GetLambdaTest().cvargument(2));

    assertTargets(lambda_three_out, {&lambda_three, &lambda_four});

    assertTargets(lambda_four_out, {&lambda_three, &lambda_four});

    assertTargets(lambda_indcall_out, {&lambda_indcall});
    assertTargets(lambda_indcall_arg, {&lambda_three, &lambda_four});

    assertTargets(lambda_test_out, {&lambda_test});
    assertTargets(lambda_test_cv0, {&lambda_indcall});
    assertTargets(lambda_test_cv1, {&lambda_three, &lambda_four});
    assertTargets(lambda_test_cv2, {&lambda_three, &lambda_four});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda_test});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  IndirectCallTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestIndirectCall2()
{

  auto validatePointsToGraph = [](
    const jlm::aa::PointsToGraph & pointsToGraph,
    const IndirectCallTest2 & test)
  {
    assert(pointsToGraph.NumAllocaNodes() == 3);
    assert(pointsToGraph.NumLambdaNodes() == 7);
    assert(pointsToGraph.NumDeltaNodes() == 2);
    assert(pointsToGraph.NumRegisterNodes() == 24);

    auto & lambdaThree = pointsToGraph.GetLambdaNode(test.GetLambdaThree());
    auto & lambdaThreeOutput = pointsToGraph.GetRegisterNode(*test.GetLambdaThree().output());

    auto & lambdaFour = pointsToGraph.GetLambdaNode(test.GetLambdaFour());
    auto & lambdaFourOutput = pointsToGraph.GetRegisterNode(*test.GetLambdaFour().output());

    assertTargets(lambdaThreeOutput, {&lambdaThree, &lambdaFour});
    assertTargets(lambdaFourOutput, {&lambdaThree, &lambdaFour});
  };

  IndirectCallTest2 test;
	// jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
	// std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  validatePointsToGraph(*pointsToGraph, test);
}

static void
TestExternalCall()
{
  auto validatePointsToGraph = [](
    const jlm::aa::PointsToGraph & pointsToGraph,
    const ExternalCallTest & test)
  {
    assert(pointsToGraph.NumAllocaNodes() == 2);
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumImportNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 10);

    auto & lambdaF = pointsToGraph.GetLambdaNode(test.LambdaF());
    auto & lambdaFArgument0 = pointsToGraph.GetRegisterNode(*test.LambdaF().fctargument(0));
    auto & lambdaFArgument1 = pointsToGraph.GetRegisterNode(*test.LambdaF().fctargument(1));

    auto & callResult = pointsToGraph.GetRegisterNode(*test.CallG().Result(0));

    auto & externalMemory = pointsToGraph.GetExternalMemoryNode();

    assertTargets(lambdaFArgument0, {&lambdaF, &externalMemory});
    assertTargets(lambdaFArgument1, {&lambdaF, &externalMemory});
    assertTargets(callResult, {&lambdaF, &externalMemory});
  };

  ExternalCallTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph) << std::flush;

  validatePointsToGraph(*pointsToGraph, test);
}

static void
TestGamma()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const GammaTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 15);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);

    for (size_t n = 1; n < 5; n++) {
      auto & lambdaArgument = pointsToGraph.GetRegisterNode(*test.lambda->fctargument(n));
      assertTargets(lambdaArgument, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    }

    for (size_t n = 0; n < 4; n++) {
      auto & argument0 = pointsToGraph.GetRegisterNode(*test.gamma->entryvar(n)->argument(0));
      auto & argument1 = pointsToGraph.GetRegisterNode(*test.gamma->entryvar(n)->argument(1));

      assertTargets(argument0, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
      assertTargets(argument1, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    }

    for (size_t n = 0; n < 4; n++) {
      auto & gammaOutput = pointsToGraph.GetRegisterNode(*test.gamma->exitvar(0));
      assertTargets(gammaOutput, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    }

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  GammaTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestTheta()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const ThetaTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 5);

    auto & lambda = pointsToGraph.GetLambdaNode(*test.lambda);
    auto & lambdaArgument1 = pointsToGraph.GetRegisterNode(*test.lambda->fctargument(1));
    auto & lambdaOutput = pointsToGraph.GetRegisterNode(*test.lambda->output());

    auto & gepOutput = pointsToGraph.GetRegisterNode(*test.gep->output(0));

    auto & thetaArgument2 = pointsToGraph.GetRegisterNode(*test.theta->output(2)->argument());
    auto & thetaOutput2 = pointsToGraph.GetRegisterNode(*test.theta->output(2));

    assertTargets(lambdaArgument1, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    assertTargets(lambdaOutput, {&lambda});

    assertTargets(gepOutput, {&lambda, &pointsToGraph.GetExternalMemoryNode()});

    assertTargets(thetaArgument2, {&lambda, &pointsToGraph.GetExternalMemoryNode()});
    assertTargets(thetaOutput2, {&lambda, &pointsToGraph.GetExternalMemoryNode()});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  ThetaTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestDelta1()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const DeltaTest1 & test)
  {
    assert(ptg.NumDeltaNodes() == 1);
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumRegisterNodes() == 6);

    auto & delta_f = ptg.GetDeltaNode(*test.delta_f);
    auto & pdelta_f = ptg.GetRegisterNode(*test.delta_f->output());

    auto & lambda_g = ptg.GetLambdaNode(*test.lambda_g);
    auto & plambda_g = ptg.GetRegisterNode(*test.lambda_g->output());
    auto & lambda_g_arg0 = ptg.GetRegisterNode(*test.lambda_g->fctargument(0));

    auto & lambda_h = ptg.GetLambdaNode(*test.lambda_h);
    auto & plambda_h = ptg.GetRegisterNode(*test.lambda_h->output());
    auto & lambda_h_cv0 = ptg.GetRegisterNode(*test.lambda_h->cvargument(0));
    auto & lambda_h_cv1 = ptg.GetRegisterNode(*test.lambda_h->cvargument(1));

    assertTargets(pdelta_f, {&delta_f});

    assertTargets(plambda_g, {&lambda_g});
    assertTargets(plambda_h, {&lambda_h});

    assertTargets(lambda_g_arg0, {&delta_f});

    assertTargets(lambda_h_cv0, {&delta_f});
    assertTargets(lambda_h_cv1, {&lambda_g});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda_h});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  DeltaTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestDelta2()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const DeltaTest2 & test)
  {
    assert(ptg.NumDeltaNodes() == 2);
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumRegisterNodes() == 8);

    auto & delta_d1 = ptg.GetDeltaNode(*test.delta_d1);
    auto & delta_d1_out = ptg.GetRegisterNode(*test.delta_d1->output());

    auto & delta_d2 = ptg.GetDeltaNode(*test.delta_d2);
    auto & delta_d2_out = ptg.GetRegisterNode(*test.delta_d2->output());

    auto & lambda_f1 = ptg.GetLambdaNode(*test.lambda_f1);
    auto & lambda_f1_out = ptg.GetRegisterNode(*test.lambda_f1->output());
    auto & lambda_f1_cvd1 = ptg.GetRegisterNode(*test.lambda_f1->cvargument(0));

    auto & lambda_f2 = ptg.GetLambdaNode(*test.lambda_f2);
    auto & lambda_f2_out = ptg.GetRegisterNode(*test.lambda_f2->output());
    auto & lambda_f2_cvd1 = ptg.GetRegisterNode(*test.lambda_f2->cvargument(0));
    auto & lambda_f2_cvd2 = ptg.GetRegisterNode(*test.lambda_f2->cvargument(1));
    auto & lambda_f2_cvf1 = ptg.GetRegisterNode(*test.lambda_f2->cvargument(2));

    assertTargets(delta_d1_out, {&delta_d1});
    assertTargets(delta_d2_out, {&delta_d2});

    assertTargets(lambda_f1_out, {&lambda_f1});
    assertTargets(lambda_f1_cvd1, {&delta_d1});

    assertTargets(lambda_f2_out, {&lambda_f2});
    assertTargets(lambda_f2_cvd1, {&delta_d1});
    assertTargets(lambda_f2_cvd2, {&delta_d2});
    assertTargets(lambda_f2_cvf1, {&lambda_f1});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda_f2});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  DeltaTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*ptg);
  validate_ptg(*ptg, test);
}

static void
TestImports()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const ImportTest & test)
  {
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumImportNodes() == 2);
    assert(ptg.NumRegisterNodes() == 8);

    auto & d1 = ptg.GetImportNode(*test.import_d1);
    auto & import_d1 = ptg.GetRegisterNode(*test.import_d1);

    auto & d2 = ptg.GetImportNode(*test.import_d2);
    auto & import_d2 = ptg.GetRegisterNode(*test.import_d2);

    auto & lambda_f1 = ptg.GetLambdaNode(*test.lambda_f1);
    auto & lambda_f1_out = ptg.GetRegisterNode(*test.lambda_f1->output());
    auto & lambda_f1_cvd1 = ptg.GetRegisterNode(*test.lambda_f1->cvargument(0));

    auto & lambda_f2 = ptg.GetLambdaNode(*test.lambda_f2);
    auto & lambda_f2_out = ptg.GetRegisterNode(*test.lambda_f2->output());
    auto & lambda_f2_cvd1 = ptg.GetRegisterNode(*test.lambda_f2->cvargument(0));
    auto & lambda_f2_cvd2 = ptg.GetRegisterNode(*test.lambda_f2->cvargument(1));
    auto & lambda_f2_cvf1 = ptg.GetRegisterNode(*test.lambda_f2->cvargument(2));

    assertTargets(import_d1, {&d1});
    assertTargets(import_d2, {&d2});

    assertTargets(lambda_f1_out, {&lambda_f1});
    assertTargets(lambda_f1_cvd1, {&d1});

    assertTargets(lambda_f2_out, {&lambda_f2});
    assertTargets(lambda_f2_cvd1, {&d1});
    assertTargets(lambda_f2_cvd2, {&d2});
    assertTargets(lambda_f2_cvf1, {&lambda_f1});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda_f2});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  ImportTest test;
  // jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*ptg);
  validate_ptg(*ptg, test);
}

static void
TestPhi1()
{
  auto validate_ptg = [](const jlm::aa::PointsToGraph & ptg, const PhiTest1 & test)
  {
    assert(ptg.NumAllocaNodes() == 1);
    assert(ptg.NumLambdaNodes() == 2);
    assert(ptg.NumRegisterNodes() == 16);

    auto & lambda_fib = ptg.GetLambdaNode(*test.lambda_fib);
    auto & lambda_fib_out = ptg.GetRegisterNode(*test.lambda_fib->output());
    auto & lambda_fib_arg1 = ptg.GetRegisterNode(*test.lambda_fib->fctargument(1));

    auto & lambda_test = ptg.GetLambdaNode(*test.lambda_test);
    auto & lambda_test_out = ptg.GetRegisterNode(*test.lambda_test->output());

    auto & phi_rv = ptg.GetRegisterNode(*test.phi->begin_rv().output());
    auto & phi_rv_arg = ptg.GetRegisterNode(*test.phi->begin_rv().output()->argument());

    auto & gamma_result = ptg.GetRegisterNode(*test.gamma->subregion(0)->argument(1));
    auto & gamma_fib = ptg.GetRegisterNode(*test.gamma->subregion(0)->argument(2));

    auto & alloca = ptg.GetAllocaNode(*test.alloca);
    auto & alloca_out = ptg.GetRegisterNode(*test.alloca->output(0));

    assertTargets(lambda_fib_out, {&lambda_fib});
    assertTargets(lambda_fib_arg1, {&alloca});

    assertTargets(lambda_test_out, {&lambda_test});

    assertTargets(phi_rv, {&lambda_fib});
    assertTargets(phi_rv_arg, {&lambda_fib});

    assertTargets(gamma_result, {&alloca});
    assertTargets(gamma_fib, {&lambda_fib});

    assertTargets(alloca_out, {&alloca});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambda_test});
    assert(ptg.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  PhiTest1 test;
//	jive::view(test.graph().root(), stdout);

  auto ptg = RunSteensgaard(test.module());
//	std::cout << jlm::aa::PointsToGraph::ToDot(*PointsToGraph);
  validate_ptg(*ptg, test);
}

static void
TestExternalMemory()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const ExternalMemoryTest & test)
  {
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 3);

    auto & lambdaF = pointsToGraph.GetLambdaNode(*test.LambdaF);
    auto & lambdaFArgument0 = pointsToGraph.GetRegisterNode(*test.LambdaF->fctargument(0));
    auto & lambdaFArgument1 = pointsToGraph.GetRegisterNode(*test.LambdaF->fctargument(1));

    assertTargets(lambdaFArgument0, {&lambdaF, &pointsToGraph.GetExternalMemoryNode()});
    assertTargets(lambdaFArgument1, {&lambdaF, &pointsToGraph.GetExternalMemoryNode()});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({&lambdaF});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  ExternalMemoryTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestEscapedMemory1()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const EscapedMemoryTest1 & test)
  {
    assert(pointsToGraph.NumDeltaNodes() == 4);
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 10);

    auto & lambdaTestArgument0 = pointsToGraph.GetRegisterNode(*test.LambdaTest->fctargument(0));
    auto & lambdaTestCv0 = pointsToGraph.GetRegisterNode(*test.LambdaTest->cvargument(0));
    auto & loadNode1Output = pointsToGraph.GetRegisterNode(*test.LoadNode1->output(0));

    auto deltaA = &pointsToGraph.GetDeltaNode(*test.DeltaA);
    auto deltaB = &pointsToGraph.GetDeltaNode(*test.DeltaB);
    auto deltaX = &pointsToGraph.GetDeltaNode(*test.DeltaX);
    auto deltaY = &pointsToGraph.GetDeltaNode(*test.DeltaY);
    auto lambdaTest = &pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto externalMemory = &pointsToGraph.GetExternalMemoryNode();

    assertTargets(lambdaTestArgument0, {deltaA, deltaX, deltaY, lambdaTest, externalMemory});
    assertTargets(lambdaTestCv0, {deltaB});
    assertTargets(loadNode1Output, {deltaA, deltaX, deltaY, lambdaTest, externalMemory});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({
      lambdaTest,
      deltaA,
      deltaX,
      deltaY});

    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  EscapedMemoryTest1 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestEscapedMemory2()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const EscapedMemoryTest2 & test)
  {
    assert(pointsToGraph.NumImportNodes() == 2);
    assert(pointsToGraph.NumLambdaNodes() == 3);
    assert(pointsToGraph.NumMallocNodes() == 2);
    assert(pointsToGraph.NumRegisterNodes() == 10);

    auto returnAddressFunction = &pointsToGraph.GetLambdaNode(*test.ReturnAddressFunction);
    auto callExternalFunction1 = &pointsToGraph.GetLambdaNode(*test.CallExternalFunction1);
    auto callExternalFunction2 = &pointsToGraph.GetLambdaNode(*test.CallExternalFunction2);
    auto returnAddressMalloc = &pointsToGraph.GetMallocNode(*test.ReturnAddressMalloc);
    auto callExternalFunction1Malloc = &pointsToGraph.GetMallocNode(*test.CallExternalFunction1Malloc);
    auto externalMemory = &pointsToGraph.GetExternalMemoryNode();

    auto & externalFunction2CallResult = pointsToGraph.GetRegisterNode(*test.ExternalFunction2Call->Result(0));

    assertTargets(
      externalFunction2CallResult,
      {
        returnAddressFunction,
        callExternalFunction1,
        callExternalFunction2,
        externalMemory,
        returnAddressMalloc,
        callExternalFunction1Malloc
      });

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({
      returnAddressFunction,
      callExternalFunction1,
      callExternalFunction2,
      returnAddressMalloc,
      callExternalFunction1Malloc});

    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  EscapedMemoryTest2 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestEscapedMemory3()
{
  auto ValidatePointsToGraph = [](const jlm::aa::PointsToGraph & pointsToGraph, const EscapedMemoryTest3 & test)
  {
    assert(pointsToGraph.NumDeltaNodes() == 1);
    assert(pointsToGraph.NumImportNodes() == 1);
    assert(pointsToGraph.NumLambdaNodes() == 1);
    assert(pointsToGraph.NumRegisterNodes() == 5);

    auto lambdaTest = &pointsToGraph.GetLambdaNode(*test.LambdaTest);
    auto deltaGlobal = &pointsToGraph.GetDeltaNode(*test.DeltaGlobal);
    auto externalMemory = &pointsToGraph.GetExternalMemoryNode();

    auto & callExternalFunctionResult = pointsToGraph.GetRegisterNode(*test.CallExternalFunction->Result(0));

    assertTargets(callExternalFunctionResult, {lambdaTest, deltaGlobal, externalMemory});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes({lambdaTest, deltaGlobal});
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  EscapedMemoryTest3 test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestMemcpy()
{
  /*
   * Arrange
   */
  auto ValidatePointsToGraph = [](
    const jlm::aa::PointsToGraph & pointsToGraph,
    const MemcpyTest & test)
  {
    assert(pointsToGraph.NumDeltaNodes() == 2);
    assert(pointsToGraph.NumLambdaNodes() == 2);
    assert(pointsToGraph.NumRegisterNodes() == 11);

    auto localArray = &pointsToGraph.GetDeltaNode(test.LocalArray());
    auto globalArray = &pointsToGraph.GetDeltaNode(test.GlobalArray());

    auto & memCpyDest = pointsToGraph.GetRegisterNode(*test.Memcpy().input(0)->origin());
    auto & memCpySrc = pointsToGraph.GetRegisterNode(*test.Memcpy().input(1)->origin());

    auto lambdaF = &pointsToGraph.GetLambdaNode(test.LambdaF());
    auto lambdaG = &pointsToGraph.GetLambdaNode(test.LambdaG());

    assertTargets(memCpyDest, {globalArray});
    assertTargets(memCpySrc, {localArray});

    jlm::HashSet<const jlm::aa::PointsToGraph::MemoryNode*> expectedEscapedMemoryNodes(
      {
        globalArray,
        localArray,
        lambdaF,
        lambdaG
      });
    assert(pointsToGraph.GetEscapedMemoryNodes() == expectedEscapedMemoryNodes);
  };

  MemcpyTest test;
  // jive::view(test.graph().root(), stdout);

  /*
   * Act
   */
  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  /*
   * Assert
   */
  ValidatePointsToGraph(*pointsToGraph, test);
}

static void
TestLinkedList()
{
  auto validatePointsToGraph = [](
    const jlm::aa::PointsToGraph & pointsToGraph,
    const LinkedListTest & test)
  {
    assert(pointsToGraph.NumAllocaNodes() == 1);
    assert(pointsToGraph.NumDeltaNodes() == 1);
    assert(pointsToGraph.NumLambdaNodes() == 1);

    auto & allocaNode = pointsToGraph.GetAllocaNode(test.GetAlloca());
    auto & deltaMyListNode = pointsToGraph.GetDeltaNode(test.GetDeltaMyList());

    assertTargets(allocaNode, {&allocaNode, &deltaMyListNode});
    assertTargets(deltaMyListNode, {&allocaNode, &deltaMyListNode});
  };

  LinkedListTest test;
  // jive::view(test.graph().root(), stdout);

  auto pointsToGraph = RunSteensgaard(test.module());
  // std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  validatePointsToGraph(*pointsToGraph, test);
}

static int
test()
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

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestSteensgaard", test)