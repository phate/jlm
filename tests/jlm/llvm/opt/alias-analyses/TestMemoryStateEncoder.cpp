/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/Operators.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>

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
    std::is_base_of<RvsdgTest, Test>::value,
    "Test should be derived from RvsdgTest class.");

  static_assert(
    std::is_base_of<jlm::aa::AliasAnalysis, Analysis>::value,
    "Analysis should be derived from AliasAnalysis class.");

  static_assert(
    std::is_base_of<jlm::aa::MemoryNodeProvider, Provider>::value,
    "Provider should be derived from MemoryNodeProvider class.");

  Test test;
  auto & rvsdgModule = test.module();
  jive::view(rvsdgModule.Rvsdg().root(), stdout);

  jlm::StatisticsCollector statisticsCollector;

  Analysis aliasAnalysis;
  auto pointsToGraph = aliasAnalysis.Analyze(rvsdgModule, statisticsCollector);
  std::cout << jlm::aa::PointsToGraph::ToDot(*pointsToGraph);

  UnlinkUnknownMemoryNode(*pointsToGraph);

  auto provisioning = Provider::Create(rvsdgModule, *pointsToGraph);

  jlm::aa::MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
  jive::view(rvsdgModule.Rvsdg().root(), stdout);

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
ValidateStoreTest1SteensgaardAgnostic(const StoreTest1 & test)
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
ValidateStoreTest1SteensgaardRegionAware(const StoreTest1 & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 9);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 4, 1));

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
ValidateStoreTest2SteensgaardAgnostic(const StoreTest2 & test)
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
ValidateStoreTest2SteensgaardRegionAware(const StoreTest2 & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 11);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

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
ValidateLoadTest1SteensgaardAgnostic(const LoadTest1 & test)
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
ValidateLoadTest1SteensgaardRegionAware(const LoadTest1 & test)
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
ValidateLoadTest2SteensgaardAgnostic(const LoadTest2 & test)
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
ValidateLoadTest2SteensgaardRegionAware(const LoadTest2 & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 13);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

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
ValidateLoadFromUndefSteensgaardAgnostic(const LoadFromUndefTest & test)
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
ValidateLoadFromUndefSteensgaardRegionAware(const LoadFromUndefTest & test)
{
  using namespace jlm;

  assert(test.Lambda().subregion()->nnodes() == 3);

  auto lambdaExitMerge = jive::node_output::node(test.Lambda().fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 0, 1));

  auto load = jive::node_output::node(test.Lambda().fctresult(0)->origin());
  assert(is<LoadOperation>(*load, 1, 1));
}

static void
ValidateCallTest1SteensgaardAgnostic(const CallTest1 & test)
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
ValidateCallTest1SteensgaardRegionAware(const CallTest1 & test)
{
  using namespace jlm;

  /* validate f */
  {
    auto lambdaEntrySplit = input_node(*test.lambda_f->fctargument(3)->begin());
    auto lambdaExitMerge = jive::node_output::node(test.lambda_f->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_f->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_f->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

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

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));

    assert(is<LoadOperation>(*loadX, 2, 2));
    assert(jive::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadOperation>(*loadY, 2, 2));
    assert(jive::node_output::node(loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge = jive::node_output::node(test.CallF().input(4)->origin());
    auto callExitSplit = input_node(*test.CallF().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 2, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 2));

    callEntryMerge = jive::node_output::node(test.CallG().input(4)->origin());
    callExitSplit = input_node(*test.CallG().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));
  }
}

static void
ValidateCallTest2SteensgaardAgnostic(const CallTest2 & test)
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
ValidateCallTest2SteensgaardRegionAware(const CallTest2 & test)
{
  using namespace jlm;

  /* validate create function */
  {
    assert(test.lambda_create->subregion()->nnodes() == 7);

    auto stateMerge = input_node(*test.malloc->output(1)->begin());
    assert(is<MemStateMergeOperator>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jive::node_output::node(stateMerge->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));

    auto lambdaExitMerge = input_node(*stateMerge->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

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
ValidateIndirectCallTest1SteensgaardAgnostic(const IndirectCallTest1 & test)
{
  using namespace jlm;

  /* validate indcall function */
  {
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 5);

    auto lambda_exit_mux = jive::node_output::node(test.GetLambdaIndcall().fctresult(2)->origin());
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
    assert(test.GetLambdaTest().subregion()->nnodes() == 9);

    auto lambda_exit_mux = jive::node_output::node(test.GetLambdaTest().fctresult(2)->origin());
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
ValidateIndirectCallTest1SteensgaardRegionAware(const IndirectCallTest1 & test)
{
  using namespace jlm;

  /* validate indcall function */
  {
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 5);

    auto lambdaExitMerge = jive::node_output::node(test.GetLambdaIndcall().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto callExitSplit = jive::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto call = jive::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 4));

    auto callEntryMerge = jive::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto lambdaEntrySplit = jive::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }

  /* validate test function */
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 9);

    auto lambdaExitMerge = jive::node_output::node(test.GetLambdaTest().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto callExitSplit = jive::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto call = jive::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 5, 4));

    auto callEntryMerge = jive::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    callExitSplit = jive::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    call = jive::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 5, 4));

    callEntryMerge = jive::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto lambdaEntrySplit = jive::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }
}

static void
ValidateGammaTestSteensgaardAgnostic(const GammaTest & test)
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
ValidateGammaTestSteensgaardRegionAware(const GammaTest & test)
{
  using namespace jlm;

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jive::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<LoadOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jive::node_output::node(loadTmp2->input(1)->origin());
  assert(is<LoadOperation>(*loadTmp1, 3, 3));

  auto lambdaEntrySplit = jive::node_output::node(loadTmp1->input(1)->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateThetaTestSteensgaardAgnostic(const ThetaTest & test)
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
ValidateThetaTestSteensgaardRegionAware(const ThetaTest & test)
{
  using namespace jlm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge = jive::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto thetaOutput = AssertedCast<jive::theta_output>(lambdaExitMerge->input(0)->origin());
  auto theta = jive::node_output::node(thetaOutput);
  assert(theta == test.theta);

  auto storeStateOutput = thetaOutput->result()->origin();
  auto store = jive::node_output::node(storeStateOutput);
  assert(is<StoreOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index()+2)->origin() == thetaOutput->argument());

  auto lambdaEntrySplit = jive::node_output::node(thetaOutput->input()->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateDeltaTest1SteensgaardAgnostic(const DeltaTest1 & test)
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
ValidateDeltaTest1SteensgaardRegionAware(const DeltaTest1 & test)
{
  using namespace jlm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit = input_node(*test.lambda_h->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));

  auto storeF = input_node(*test.constantFive->output(0)->begin());
  assert(is<StoreOperation>(*storeF, 3, 1));
  assert(jive::node_output::node(storeF->input(2)->origin()) == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = input_node(*test.lambda_g->fctargument(0)->begin());
  assert(is<LoadOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

static void
ValidateDeltaTest2SteensgaardAgnostic(const DeltaTest2 & test)
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
ValidateDeltaTest2SteensgaardRegionAware(const DeltaTest2 & test)
{
  using namespace jlm;

  /* Validate f1() */
  {
    assert(test.lambda_f1->subregion()->nnodes() == 4);

    auto lambdaExitMerge = jive::node_output::node(test.lambda_f1->fctresult(1)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto storeNode = jive::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<StoreOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit = jive::node_output::node(storeNode->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = input_node(*test.lambda_f2->cvargument(0)->begin());
    assert(is<StoreOperation>(*storeD1, 3, 1));
    assert(jive::node_output::node(storeD1->input(2)->origin()) == lambdaEntrySplit);

    auto storeD2 = input_node(*test.lambda_f2->cvargument(1)->begin());
    assert(is<StoreOperation>(*storeD2, 3, 1));
    assert(jive::node_output::node(storeD2->input(2)->origin()) == lambdaEntrySplit);

    auto callEntryMerge = input_node(*storeD1->output(0)->begin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto callF1 = input_node(*callEntryMerge->output(0)->begin());
    assert(is<CallOperation>(*callF1, 4, 3));

    auto callExitSplit = input_node(*callF1->output(1)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto lambdaExitMerge = input_node(*callExitSplit->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidateDeltaTest3SteensgaardAgnostic(const DeltaTest3 & test)
{
  using namespace jlm;

  /* validate f() */
  {
    assert(test.LambdaF().subregion()->nnodes() == 6);

    auto lambdaExitMerge = jive::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto truncNode = jive::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<trunc_op>(*truncNode, 1, 1));

    auto loadG1Node = jive::node_output::node(truncNode->input(0)->origin());
    assert(is<LoadOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jive::node_output::node(loadG1Node->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

    jive::node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jive::node_output::node(input->origin());
      if (is<StoreOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jive::node_output::node(storeG2Node->input(2)->origin());
    assert(is<LoadOperation>(*loadG2Node, 2, 2));

    auto node = jive::node_output::node(loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

static void
ValidateDeltaTest3SteensgaardRegionAware(const DeltaTest3 & test)
{
  using namespace jlm;

  /* validate f() */
  {
    assert(test.LambdaF().subregion()->nnodes() == 6);

    auto lambdaExitMerge = jive::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

    auto truncNode = jive::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<trunc_op>(*truncNode, 1, 1));

    auto loadG1Node = jive::node_output::node(truncNode->input(0)->origin());
    assert(is<LoadOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jive::node_output::node(loadG1Node->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

    jive::node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jive::node_output::node(input->origin());
      if (is<StoreOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jive::node_output::node(storeG2Node->input(2)->origin());
    assert(is<LoadOperation>(*loadG2Node, 2, 2));

    auto node = jive::node_output::node(loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

static void
ValidateImportTestSteensgaardAgnostic(const ImportTest & test)
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
ValidateImportTestSteensgaardRegionAware(const ImportTest & test)
{
  using namespace jlm;

  /* Validate f1() */
  {
    assert(test.lambda_f1->subregion()->nnodes() == 4);

    auto lambdaExitMerge = jive::node_output::node(test.lambda_f1->fctresult(1)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto storeNode = jive::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<StoreOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit = jive::node_output::node(storeNode->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = input_node(*test.lambda_f2->cvargument(0)->begin());
    assert(is<StoreOperation>(*storeD1, 3, 1));
    assert(jive::node_output::node(storeD1->input(2)->origin()) == lambdaEntrySplit);

    auto storeD2 = input_node(*test.lambda_f2->cvargument(1)->begin());
    assert(is<StoreOperation>(*storeD2, 3, 1));
    assert(jive::node_output::node(storeD2->input(2)->origin()) == lambdaEntrySplit);

    auto callEntryMerge = input_node(*storeD1->output(0)->begin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto callF1 = input_node(*callEntryMerge->output(0)->begin());
    assert(is<CallOperation>(*callF1, 4, 3));

    auto callExitSplit = input_node(*callF1->output(1)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto lambdaExitMerge = input_node(*callExitSplit->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidatePhiTestSteensgaardAgnostic(const PhiTest1 & test)
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
ValidatePhiTestSteensgaardRegionAware(const PhiTest1 & test)
{
  using namespace jlm;

  auto arrayStateIndex = (*test.alloca->output(1)->begin())->index();

  auto lambdaExitMerge = jive::node_output::node(test.lambda_fib->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

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
ValidateMemcpySteensgaardAgnostic(const MemcpyTest & test)
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

static void
ValidateMemcpySteensgaardRegionAware(const MemcpyTest & test)
{
  using namespace jlm;

  /*
   * Validate function f
   */
  {
    auto lambdaExitMerge = jive::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto load = jive::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<LoadOperation>(*load, 2, 2));

    auto store = jive::node_output::node(load->input(1)->origin());
    assert(is<StoreOperation>(*store, 3, 1));

    auto lambdaEntrySplit = jive::node_output::node(store->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }

  /*
   * Validate function g
   */
  {
    auto callNode = input_node(*test.LambdaG().cvargument(2)->begin());
    assert(is<CallOperation>(*callNode, 4, 4));

    auto callEntryMerge = jive::node_output::node(callNode->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto callExitSplit = input_node(*callNode->output(2)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto memcpyNode = jive::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<Memcpy>(*memcpyNode, 6, 2));

    auto lambdaEntrySplit = jive::node_output::node(memcpyNode->input(4)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
    assert(jive::node_output::node(memcpyNode->input(5)->origin()) == lambdaEntrySplit);

    auto lambdaExitMerge = input_node(*callExitSplit->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
    assert(jive::node_output::node(lambdaExitMerge->input(0)->origin()) == memcpyNode
           || jive::node_output::node(lambdaExitMerge->input(1)->origin()) == memcpyNode);
  }
}

static int
test()
{
  using namespace jlm::aa;

  ValidateTest<StoreTest1, Steensgaard, AgnosticMemoryNodeProvider>(ValidateStoreTest1SteensgaardAgnostic);
  ValidateTest<StoreTest1, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateStoreTest1SteensgaardRegionAware);

  ValidateTest<StoreTest2, Steensgaard, AgnosticMemoryNodeProvider>(ValidateStoreTest2SteensgaardAgnostic);
  ValidateTest<StoreTest2, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateStoreTest2SteensgaardRegionAware);

  ValidateTest<LoadTest1, Steensgaard, AgnosticMemoryNodeProvider>(ValidateLoadTest1SteensgaardAgnostic);
  ValidateTest<LoadTest1, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateLoadTest1SteensgaardRegionAware);

  ValidateTest<LoadTest2, Steensgaard, AgnosticMemoryNodeProvider>(ValidateLoadTest2SteensgaardAgnostic);
  ValidateTest<LoadTest2, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateLoadTest2SteensgaardRegionAware);

  ValidateTest<
    LoadFromUndefTest,
    Steensgaard,
    AgnosticMemoryNodeProvider
  >(ValidateLoadFromUndefSteensgaardAgnostic);
  ValidateTest<
    LoadFromUndefTest,
    Steensgaard,
    RegionAwareMemoryNodeProvider
  >(ValidateLoadFromUndefSteensgaardRegionAware);

  ValidateTest<CallTest1, Steensgaard, AgnosticMemoryNodeProvider>(ValidateCallTest1SteensgaardAgnostic);
  ValidateTest<CallTest1, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateCallTest1SteensgaardRegionAware);

  ValidateTest<CallTest2, Steensgaard, AgnosticMemoryNodeProvider>(ValidateCallTest2SteensgaardAgnostic);
  ValidateTest<CallTest2, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateCallTest2SteensgaardRegionAware);

  ValidateTest<
    IndirectCallTest1,
    Steensgaard,
    AgnosticMemoryNodeProvider
  >(ValidateIndirectCallTest1SteensgaardAgnostic);
  ValidateTest<
    IndirectCallTest1,
    Steensgaard,
    RegionAwareMemoryNodeProvider
  >(ValidateIndirectCallTest1SteensgaardRegionAware);

  ValidateTest<GammaTest, Steensgaard, AgnosticMemoryNodeProvider>(ValidateGammaTestSteensgaardAgnostic);
  ValidateTest<GammaTest, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateGammaTestSteensgaardRegionAware);

  ValidateTest<ThetaTest, Steensgaard, AgnosticMemoryNodeProvider>(ValidateThetaTestSteensgaardAgnostic);
  ValidateTest<ThetaTest, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateThetaTestSteensgaardRegionAware);

  ValidateTest<DeltaTest1, Steensgaard, AgnosticMemoryNodeProvider>(ValidateDeltaTest1SteensgaardAgnostic);
  ValidateTest<DeltaTest1, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateDeltaTest1SteensgaardRegionAware);

  ValidateTest<DeltaTest2, Steensgaard, AgnosticMemoryNodeProvider>(ValidateDeltaTest2SteensgaardAgnostic);
  ValidateTest<DeltaTest2, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateDeltaTest2SteensgaardRegionAware);

  ValidateTest<DeltaTest3, Steensgaard, AgnosticMemoryNodeProvider>(ValidateDeltaTest3SteensgaardAgnostic);
  ValidateTest<DeltaTest3, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateDeltaTest3SteensgaardRegionAware);

  ValidateTest<ImportTest, Steensgaard, AgnosticMemoryNodeProvider>(ValidateImportTestSteensgaardAgnostic);
  ValidateTest<ImportTest, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateImportTestSteensgaardRegionAware);

  ValidateTest<PhiTest1, Steensgaard, AgnosticMemoryNodeProvider>(ValidatePhiTestSteensgaardAgnostic);
  ValidateTest<PhiTest1, Steensgaard, RegionAwareMemoryNodeProvider>(ValidatePhiTestSteensgaardRegionAware);

  ValidateTest<MemcpyTest, Steensgaard, AgnosticMemoryNodeProvider>(ValidateMemcpySteensgaardAgnostic);
  ValidateTest<MemcpyTest, Steensgaard, RegionAwareMemoryNodeProvider>(ValidateMemcpySteensgaardRegionAware);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder", test)