/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "AliasAnalysesTests.hpp"

#include <test-registry.hpp>

#include <jive/view.hpp>

#include <jlm/opt/alias-analyses/BasicMemoryNodeProvider.hpp>
#include <jlm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/opt/alias-analyses/Operators.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/util/Statistics.hpp>

#include <iostream>

static void
UnlinkUnknownMemoryNode(jlm::aa::PointsToGraph & pointsToGraph)
{
  std::vector<jlm::aa::PointsToGraph::Node*> memoryNodes;
  for (auto & allocaNode : pointsToGraph.AllocaNodes())
    memoryNodes.push_back(&allocaNode);

  for (auto & deltaNode : pointsToGraph.DeltaNodes())
    memoryNodes.push_back(&deltaNode);

  for (auto & lambdaNode : pointsToGraph.LambdaNodes())
    memoryNodes.push_back(&lambdaNode);

  for (auto & mallocNode : pointsToGraph.MallocNodes())
    memoryNodes.push_back(&mallocNode);

  for (auto & node : pointsToGraph.ImportNodes())
    memoryNodes.push_back(&node);

  auto & unknownMemoryNode = pointsToGraph.GetUnknownMemoryNode();
  while (unknownMemoryNode.NumSources() != 0) {
    auto & source = *unknownMemoryNode.Sources().begin();
    for (auto & memoryNode : memoryNodes)
      source.AddEdge(*dynamic_cast<jlm::aa::PointsToGraph::MemoryNode *>(memoryNode));
    source.RemoveEdge(unknownMemoryNode);
  }
};

template <class Test, class Analysis, class Provider> static void
ValidateTest(std::function<void(const Test&)> validateEncoding)
{
  static_assert(
    std::is_base_of<AliasAnalysisTest, Test>::value,
    "Test should be derived from AliasAnalysisTest class.");

  static_assert(
    std::is_base_of<jlm::aa::AliasAnalysis, Analysis>::value,
    "Analysis should be derived from AliasAnalysis class.");

  static_assert(
    std::is_base_of<jlm::aa::MemoryNodeProvider, Provider>::value,
    "Provider should be derived from MemoryNodeProvider class.");

  Test test;
  auto & rvsdgModule = test.module();
  jive::view(rvsdgModule.Rvsdg().root(), stdout);

  jlm::StatisticsDescriptor statisticsDescriptor;

  Analysis aliasAnalysis;
  auto pointsToGraph = aliasAnalysis.Analyze(rvsdgModule, statisticsDescriptor);
  std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  UnlinkUnknownMemoryNode(*pointsToGraph);

  auto provider = Provider::Create(rvsdgModule, *pointsToGraph);

  jlm::aa::MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provider, statisticsDescriptor);

  validateEncoding(test);
}

template <class OP> static bool
is(
  const jive::node & node,
  size_t numInputs,
  size_t numOutputs)
{
  return jive::is<OP>(&node)
         && node.ninputs() == numInputs
         && node.noutputs() == numOutputs;
}

static void
ValidateStoreTest1SteensgaardBasic(const StoreTest1 & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 10);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

  assert(test.alloca_d->output(1)->nusers() == 1);
  assert(test.alloca_c->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_a->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_d->output(1)->begin())) == lambdaExitMerge);

  auto storeD = input_node(*test.alloca_c->output(1)->begin());
  assert(is<StoreOperation>(*storeD, 3, 1));
  assert(storeD->input(0)->origin() == test.alloca_c->output(0));
  assert(storeD->input(1)->origin() == test.alloca_d->output(0));

  auto storeC = input_node(*test.alloca_b->output(1)->begin());
  assert(is<StoreOperation>(*storeC, 3, 1));
  assert(storeC->input(0)->origin() == test.alloca_b->output(0));
  assert(storeC->input(1)->origin() == test.alloca_c->output(0));

  auto storeB = input_node(*test.alloca_a->output(1)->begin());
  assert(is<StoreOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_a->output(0));
  assert(storeB->input(1)->origin() == test.alloca_b->output(0));
}

static void
ValidateStoreTest2SteensgaardBasic(const StoreTest2 & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 12);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(input_node((*test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = input_node(*test.alloca_a->output(0)->begin());
  assert(is<StoreOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = input_node(*test.alloca_b->output(0)->begin());
  assert(is<StoreOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jive::node_output::node(storeB->input(2)->origin()) == storeA);
  assert(jive::node_output::node(storeB->input(3)->origin()) == storeA);

  auto storeX = input_node(*test.alloca_p->output(1)->begin());
  assert(is<StoreOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto storeY = input_node(*storeX->output(0)->begin());
  assert(is<StoreOperation>(*storeY, 3, 1));
  assert(storeY->input(0)->origin() == test.alloca_p->output(0));
  assert(storeY->input(1)->origin() == test.alloca_y->output(0));
}

static void
ValidateLoadTest1SteensgaardBasic(const LoadTest1 & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = input_node(*test.lambda->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

  auto loadA = jive::node_output::node(test.lambda->fctresult(0)->origin());
  auto loadX = jive::node_output::node(loadA->input(0)->origin());

  assert(is<LoadOperation>(*loadA, 3, 3));
  assert(jive::node_output::node(loadA->input(1)->origin()) == loadX);

  assert(is<LoadOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->fctargument(0));
  assert(jive::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);
}

static void
ValidateLoadTest2SteensgaardBasic(const LoadTest2 & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 14);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(input_node((*test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = input_node(*test.alloca_a->output(0)->begin());
  assert(is<StoreOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = input_node(*test.alloca_b->output(0)->begin());
  assert(is<StoreOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jive::node_output::node(storeB->input(2)->origin()) == storeA);
  assert(jive::node_output::node(storeB->input(3)->origin()) == storeA);

  auto storeX = input_node(*test.alloca_p->output(1)->begin());
  assert(is<StoreOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto loadP = input_node(*storeX->output(0)->begin());
  assert(is<LoadOperation>(*loadP, 2, 2));
  assert(loadP->input(0)->origin() == test.alloca_p->output(0));

  auto loadXY = input_node(*loadP->output(0)->begin());
  assert(is<LoadOperation>(*loadXY, 3, 3));
  assert(jive::node_output::node(loadXY->input(1)->origin()) == storeB);
  assert(jive::node_output::node(loadXY->input(2)->origin()) == storeB);

  auto storeY = input_node(*loadXY->output(0)->begin());
  assert(is<StoreOperation>(*storeY, 4, 2));
  assert(storeY->input(0)->origin() == test.alloca_y->output(0));
  assert(jive::node_output::node(storeY->input(2)->origin()) == loadXY);
  assert(jive::node_output::node(storeY->input(3)->origin()) == loadXY);
}

static void
ValidateLoadFromUndefSteensgaardBasic(const LoadFromUndefTest & test)
{
  using namespace jlm;

  assert(test.Lambda().subregion()->nnodes() == 4);

  auto lambdaExitMerge = jive::node_output::node(test.Lambda().fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto load = jive::node_output::node(test.Lambda().fctresult(0)->origin());
  assert(is<LoadOperation>(*load, 1, 1));

  auto lambdaEntrySplit = input_node(*test.Lambda().fctargument(0)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateCallTest1SteensgaardBasic(const CallTest1 & test)
{
  using namespace jlm;

  /* validate f */
  {
    auto lambdaEntrySplit = input_node(*test.lambda_f->fctargument(3)->begin());
    auto lambdaExitMerge = jive::node_output::node(test.lambda_f->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_f->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_f->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadOperation>(*loadX, 2, 2));
    assert(jive::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadOperation>(*loadY, 2, 2));
    assert(jive::node_output::node(loadY->input(1)->origin()) == lambdaEntrySplit);
  }

  /* validate g */
  {
    auto lambdaEntrySplit = input_node(*test.lambda_g->fctargument(3)->begin());
    auto lambdaExitMerge = jive::node_output::node(test.lambda_g->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_g->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_g->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadOperation>(*loadX, 2, 2));
    assert(jive::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadOperation>(*loadY, 2, 2));
    assert(jive::node_output::node(loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge = jive::node_output::node(test.CallF().input(4)->origin());
    auto callExitSplit = input_node(*test.CallF().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 7, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 7));

    callEntryMerge = jive::node_output::node(test.CallG().input(4)->origin());
    callExitSplit = input_node(*test.CallG().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 7, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 7));
  }
}

static void
ValidateCallTest2SteensgaardBasic(const CallTest2 & test)
{
  using namespace jlm;

  /* validate create function */
  {
    assert(test.lambda_create->subregion()->nnodes() == 7);

    auto stateMerge = input_node(*test.malloc->output(1)->begin());
    assert(is<MemStateMergeOperator>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jive::node_output::node(stateMerge->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

    auto lambdaExitMerge = input_node(*stateMerge->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto mallocStateLambdaEntryIndex = stateMerge->input(1)->origin()->index();
    auto mallocStateLambdaExitIndex = (*stateMerge->output(0)->begin())->index();
    assert(mallocStateLambdaEntryIndex == mallocStateLambdaExitIndex);
  }

  /* validate destroy function */
  {
    assert(test.lambda_destroy->subregion()->nnodes() == 4);
  }

  /* validate test function */
  {
    assert(test.lambda_test->subregion()->nnodes() == 16);
  }
}

static void
ValidateIndirectCallTest1SteensgaardBasic(const IndirectCallTest1 & test)
{
  using namespace jlm;

  /* validate indcall function */
  {
    assert(test.lambda_indcall->subregion()->nnodes() == 5);

    auto lambda_exit_mux = jive::node_output::node(test.lambda_indcall->fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jive::node_output::node(lambda_exit_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    auto call = jive::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 4));

    auto call_entry_mux = jive::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jive::node_output::node(call_entry_mux->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 5));
  }

  /* validate test function */
  {
    assert(test.lambda_test->subregion()->nnodes() == 9);

    auto lambda_exit_mux = jive::node_output::node(test.lambda_test->fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jive::node_output::node(lambda_exit_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    auto call = jive::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 5, 4));

    auto call_entry_mux = jive::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    call_exit_mux = jive::node_output::node(call_entry_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    call = jive::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 5, 4));

    call_entry_mux = jive::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jive::node_output::node(call_entry_mux->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 5));
  }
}

static void
ValidateGammaTestSteensgaardBasic(const GammaTest & test)
{
  using namespace jlm;

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jive::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<LoadOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jive::node_output::node(loadTmp2->input(1)->origin());
  assert(is<LoadOperation>(*loadTmp1, 3, 3));

  auto gamma = jive::node_output::node(loadTmp1->input(1)->origin());
  assert(gamma == test.gamma);
}

static void
ValidateThetaTestSteensgaardBasic(const ThetaTest & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambda_exit_mux = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 2, 1));

  auto thetaOutput = AssertedCast<jive::theta_output>(lambda_exit_mux->input(0)->origin());
  auto theta = jive::node_output::node(thetaOutput);
  assert(theta == test.theta);

  auto storeStateOutput = thetaOutput->result()->origin();
  auto store = jive::node_output::node(storeStateOutput);
  assert(is<StoreOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index()+2)->origin() == thetaOutput->argument());

  auto lambda_entry_mux = jive::node_output::node(thetaOutput->input()->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 2));
}

static void
ValidateDeltaTest1SteensgaardBasic(const DeltaTest1 & test)
{
  using namespace jlm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit = input_node(*test.lambda_h->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 4));

  auto storeF = input_node(*test.constantFive->output(0)->begin());
  assert(is<StoreOperation>(*storeF, 3, 1));
  assert(jive::node_output::node(storeF->input(2)->origin()) == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = input_node(*test.lambda_g->fctargument(0)->begin());
  assert(is<LoadOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

static void
ValidateDeltaTest2SteensgaardBasic(const DeltaTest2 & test)
{
  using namespace jlm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
  assert(is<StoreOperation>(*storeD1InF2, 3, 1));
  assert(jive::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
  assert(is<StoreOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
  assert(is<StoreOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
}

static void
ValidateImportTestSteensgaardBasic(const ImportTest & test)
{
  using namespace jlm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
  assert(is<StoreOperation>(*storeD1InF2, 3, 1));
  assert(jive::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
  assert(is<StoreOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
  assert(is<StoreOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
}

static void
ValidatePhiTestSteensgaardBasic(const PhiTest1 & test)
{
  using namespace jlm;

  auto arrayStateIndex = (*test.alloca->output(1)->begin())->index();

  auto lambdaExitMerge = jive::node_output::node(test.lambda_fib->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 4, 1));

  auto store = jive::node_output::node(lambdaExitMerge->input(arrayStateIndex)->origin());
  assert(is<StoreOperation>(*store, 3, 1));

  auto gamma = jive::node_output::node(store->input(2)->origin());
  assert(gamma == test.gamma);

  auto gammaStateIndex = store->input(2)->origin()->index();

  auto load1 = jive::node_output::node(test.gamma->exitvar(gammaStateIndex)->result(0)->origin());
  assert(is<LoadOperation>(*load1, 2, 2));

  auto load2 = jive::node_output::node(load1->input(1)->origin());
  assert(is<LoadOperation>(*load2, 2, 2));

  assert(load2->input(1)->origin()->index() == arrayStateIndex);
}

static void
ValidateMemcpySteensgaardBasic(const MemcpyTest & test)
{
  using namespace jlm;

  /*
   * Validate function f
   */
  {
    auto lambdaExitMerge = jive::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto load = jive::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<LoadOperation>(*load, 2, 2));

    auto store = jive::node_output::node(load->input(1)->origin());
    assert(is<StoreOperation>(*store, 3, 1));

    auto lambdaEntrySplit = jive::node_output::node(store->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));
  }

  /*
   * Validate function g
   */
  {
    auto lambdaExitMerge = jive::node_output::node(test.LambdaG().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto callExitSplit = jive::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 5));

    auto call = jive::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 4));

    auto callEntryMerge = jive::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 5, 1));

    jive::node * memcpy = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jive::node_output::node(callEntryMerge->input(n)->origin());
      if (is<Memcpy>(node))
        memcpy = node;
    }
    assert(memcpy != nullptr);
    assert(is<Memcpy>(*memcpy, 6, 2));

    auto lambdaEntrySplit = jive::node_output::node(memcpy->input(5)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));
  }
}

static int
test()
{
  using namespace jlm::aa;

  ValidateTest<StoreTest1, Steensgaard, BasicMemoryNodeProvider>(ValidateStoreTest1SteensgaardBasic);
  ValidateTest<StoreTest2, Steensgaard, BasicMemoryNodeProvider>(ValidateStoreTest2SteensgaardBasic);

  ValidateTest<LoadTest1, Steensgaard, BasicMemoryNodeProvider>(ValidateLoadTest1SteensgaardBasic);
  ValidateTest<LoadTest2, Steensgaard, BasicMemoryNodeProvider>(ValidateLoadTest2SteensgaardBasic);
  ValidateTest<LoadFromUndefTest, Steensgaard, BasicMemoryNodeProvider>(ValidateLoadFromUndefSteensgaardBasic);

  ValidateTest<CallTest1, Steensgaard, BasicMemoryNodeProvider>(ValidateCallTest1SteensgaardBasic);
  ValidateTest<CallTest2, Steensgaard, BasicMemoryNodeProvider>(ValidateCallTest2SteensgaardBasic);
  ValidateTest<IndirectCallTest1, Steensgaard, BasicMemoryNodeProvider>(ValidateIndirectCallTest1SteensgaardBasic);

  ValidateTest<GammaTest, Steensgaard, BasicMemoryNodeProvider>(ValidateGammaTestSteensgaardBasic);

  ValidateTest<ThetaTest, Steensgaard, BasicMemoryNodeProvider>(ValidateThetaTestSteensgaardBasic);

  ValidateTest<DeltaTest1, Steensgaard, BasicMemoryNodeProvider>(ValidateDeltaTest1SteensgaardBasic);
  ValidateTest<DeltaTest2, Steensgaard, BasicMemoryNodeProvider>(ValidateDeltaTest2SteensgaardBasic);

  ValidateTest<ImportTest, Steensgaard, BasicMemoryNodeProvider>(ValidateImportTestSteensgaardBasic);

  ValidateTest<PhiTest1, Steensgaard, BasicMemoryNodeProvider>(ValidatePhiTestSteensgaardBasic);

  ValidateTest<MemcpyTest, Steensgaard, BasicMemoryNodeProvider>(ValidateMemcpySteensgaardBasic);

  return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/opt/alias-analyses/TestMemoryStateEncoder", test)