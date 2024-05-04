/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/opt/alias-analyses/AgnosticMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/EliminatedMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/Operators.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownMemoryNodeEliminator.hpp>

#include <iostream>

using AgnosticTopDownMemoryNodeProvider = jlm::llvm::aa::EliminatedMemoryNodeProvider<
    jlm::llvm::aa::AgnosticMemoryNodeProvider,
    jlm::llvm::aa::TopDownMemoryNodeEliminator>;

template<class Test, class Analysis, class Provider>
static void
ValidateTest(std::function<void(const Test &)> validateEncoding)
{
  static_assert(
      std::is_base_of<jlm::tests::RvsdgTest, Test>::value,
      "Test should be derived from RvsdgTest class.");

  static_assert(
      std::is_base_of<jlm::llvm::aa::AliasAnalysis, Analysis>::value,
      "Analysis should be derived from AliasAnalysis class.");

  static_assert(
      std::is_base_of<jlm::llvm::aa::MemoryNodeProvider, Provider>::value,
      "Provider should be derived from MemoryNodeProvider class.");

  std::cout << "\n###\n";
  std::cout << "### Performing Test " << typeid(Test).name() << " using ["
            << typeid(Analysis).name() << ", " << typeid(Provider).name() << "]\n";
  std::cout << "###\n";

  Test test;
  auto & rvsdgModule = test.module();
  jlm::rvsdg::view(rvsdgModule.Rvsdg().root(), stdout);

  jlm::util::StatisticsCollector statisticsCollector;

  Analysis aliasAnalysis;
  auto pointsToGraph = aliasAnalysis.Analyze(rvsdgModule, statisticsCollector);
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  Provider provider;
  auto provisioning =
      provider.ProvisionMemoryNodes(rvsdgModule, *pointsToGraph, statisticsCollector);

  jlm::llvm::aa::MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *provisioning, statisticsCollector);
  jlm::rvsdg::view(rvsdgModule.Rvsdg().root(), stdout);

  validateEncoding(test);
}

template<class OP>
static bool
is(const jlm::rvsdg::node & node, size_t numInputs, size_t numOutputs)
{
  return jlm::rvsdg::is<OP>(&node) && node.ninputs() == numInputs && node.noutputs() == numOutputs;
}

static void
ValidateStoreTest1SteensgaardAgnostic(const jlm::tests::StoreTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 10);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

  assert(test.alloca_d->output(1)->nusers() == 1);
  assert(test.alloca_c->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_a->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_d->output(1)->begin())) == lambdaExitMerge);

  auto storeD = input_node(*test.alloca_c->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD, 3, 1));
  assert(storeD->input(0)->origin() == test.alloca_c->output(0));
  assert(storeD->input(1)->origin() == test.alloca_d->output(0));

  auto storeC = input_node(*test.alloca_b->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeC, 3, 1));
  assert(storeC->input(0)->origin() == test.alloca_b->output(0));
  assert(storeC->input(1)->origin() == test.alloca_c->output(0));

  auto storeB = input_node(*test.alloca_a->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_a->output(0));
  assert(storeB->input(1)->origin() == test.alloca_b->output(0));
}

static void
ValidateStoreTest1SteensgaardRegionAware(const jlm::tests::StoreTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 9);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 4, 1));

  assert(test.alloca_d->output(1)->nusers() == 1);
  assert(test.alloca_c->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_a->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_d->output(1)->begin())) == lambdaExitMerge);

  auto storeD = input_node(*test.alloca_c->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD, 3, 1));
  assert(storeD->input(0)->origin() == test.alloca_c->output(0));
  assert(storeD->input(1)->origin() == test.alloca_d->output(0));

  auto storeC = input_node(*test.alloca_b->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeC, 3, 1));
  assert(storeC->input(0)->origin() == test.alloca_b->output(0));
  assert(storeC->input(1)->origin() == test.alloca_c->output(0));

  auto storeB = input_node(*test.alloca_a->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_a->output(0));
  assert(storeB->input(1)->origin() == test.alloca_b->output(0));
}

static void
ValidateStoreTest1SteensgaardAgnosticTopDown(const jlm::tests::StoreTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 2);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
  assert(lambdaEntrySplit == jlm::rvsdg::node_output::node(lambdaExitMerge->input(1)->origin()));
}

static void
ValidateStoreTest2SteensgaardAgnostic(const jlm::tests::StoreTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 12);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(input_node((*test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = input_node(*test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = input_node(*test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::node_output::node(storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::node_output::node(storeB->input(3)->origin()) == storeA);

  auto storeX = input_node(*test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto storeY = input_node(*storeX->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  assert(storeY->input(0)->origin() == test.alloca_p->output(0));
  assert(storeY->input(1)->origin() == test.alloca_y->output(0));
}

static void
ValidateStoreTest2SteensgaardRegionAware(const jlm::tests::StoreTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 11);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(input_node((*test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = input_node(*test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = input_node(*test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::node_output::node(storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::node_output::node(storeB->input(3)->origin()) == storeA);

  auto storeX = input_node(*test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto storeY = input_node(*storeX->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  assert(storeY->input(0)->origin() == test.alloca_p->output(0));
  assert(storeY->input(1)->origin() == test.alloca_y->output(0));
}

static void
ValidateStoreTest2SteensgaardAgnosticTopDown(const jlm::tests::StoreTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 2);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
  assert(lambdaEntrySplit == jlm::rvsdg::node_output::node(lambdaExitMerge->input(1)->origin()));
}

static void
ValidateLoadTest1SteensgaardAgnostic(const jlm::tests::LoadTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = input_node(*test.lambda->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  auto loadX = jlm::rvsdg::node_output::node(loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  assert(jlm::rvsdg::node_output::node(loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->fctargument(0));
  assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);
}

static void
ValidateLoadTest1SteensgaardRegionAware(const jlm::tests::LoadTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = input_node(*test.lambda->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  auto loadX = jlm::rvsdg::node_output::node(loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  assert(jlm::rvsdg::node_output::node(loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->fctargument(0));
  assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);
}

static void
ValidateLoadTest1SteensgaardAgnosticTopDown(const jlm::tests::LoadTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = input_node(*test.lambda->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  auto loadX = jlm::rvsdg::node_output::node(loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  assert(jlm::rvsdg::node_output::node(loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->fctargument(0));
  assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);
}

static void
ValidateLoadTest2SteensgaardAgnostic(const jlm::tests::LoadTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 14);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(input_node((*test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(input_node((*test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = input_node(*test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = input_node(*test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::node_output::node(storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::node_output::node(storeB->input(3)->origin()) == storeA);

  auto storeX = input_node(*test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto loadP = input_node(*storeX->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadP, 2, 2));
  assert(loadP->input(0)->origin() == test.alloca_p->output(0));

  auto loadXY = input_node(*loadP->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadXY, 3, 3));
  assert(jlm::rvsdg::node_output::node(loadXY->input(1)->origin()) == storeB);
  assert(jlm::rvsdg::node_output::node(loadXY->input(2)->origin()) == storeB);

  auto storeY = input_node(*loadXY->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 4, 2));
  assert(storeY->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::node_output::node(storeY->input(2)->origin()) == loadXY);
  assert(jlm::rvsdg::node_output::node(storeY->input(3)->origin()) == loadXY);
}

static void
ValidateLoadTest2SteensgaardRegionAware(const jlm::tests::LoadTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 13);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  auto storeA = input_node(*test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = input_node(*test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::node_output::node(storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::node_output::node(storeB->input(3)->origin()) == storeA);

  auto storeX = input_node(*test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto loadP = input_node(*storeX->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadP, 2, 2));
  assert(loadP->input(0)->origin() == test.alloca_p->output(0));

  auto loadXY = input_node(*loadP->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadXY, 3, 3));
  assert(jlm::rvsdg::node_output::node(loadXY->input(1)->origin()) == storeB);
  assert(jlm::rvsdg::node_output::node(loadXY->input(2)->origin()) == storeB);

  auto storeY = input_node(*loadXY->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 4, 2));
  assert(storeY->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::node_output::node(storeY->input(2)->origin()) == loadXY);
  assert(jlm::rvsdg::node_output::node(storeY->input(3)->origin()) == loadXY);
}

static void
ValidateLoadTest2SteensgaardAgnosticTopDown(const jlm::tests::LoadTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 2);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
  assert(lambdaEntrySplit == jlm::rvsdg::node_output::node(lambdaExitMerge->input(1)->origin()));
}

static void
ValidateLoadFromUndefSteensgaardAgnostic(const jlm::tests::LoadFromUndefTest & test)
{
  using namespace jlm::llvm;

  assert(test.Lambda().subregion()->nnodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.Lambda().fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto load = jlm::rvsdg::node_output::node(test.Lambda().fctresult(0)->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));

  auto lambdaEntrySplit = input_node(*test.Lambda().fctargument(0)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateLoadFromUndefSteensgaardRegionAware(const jlm::tests::LoadFromUndefTest & test)
{
  using namespace jlm::llvm;

  assert(test.Lambda().subregion()->nnodes() == 3);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.Lambda().fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 0, 1));

  auto load = jlm::rvsdg::node_output::node(test.Lambda().fctresult(0)->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));
}

static void
ValidateLoadFromUndefSteensgaardAgnosticTopDown(const jlm::tests::LoadFromUndefTest & test)
{
  using namespace jlm::llvm;

  assert(test.Lambda().subregion()->nnodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.Lambda().fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto load = jlm::rvsdg::node_output::node(test.Lambda().fctresult(0)->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));

  auto lambdaEntrySplit = input_node(*test.Lambda().fctargument(0)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateCallTest1SteensgaardAgnostic(const jlm::tests::CallTest1 & test)
{
  using namespace jlm::llvm;

  /* validate f */
  {
    auto lambdaEntrySplit = input_node(*test.lambda_f->fctargument(3)->begin());
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_f->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_f->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_f->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadY->input(1)->origin()) == lambdaEntrySplit);
  }

  /* validate g */
  {
    auto lambdaEntrySplit = input_node(*test.lambda_g->fctargument(3)->begin());
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_g->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_g->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_g->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge = jlm::rvsdg::node_output::node(test.CallF().input(4)->origin());
    auto callExitSplit = input_node(*test.CallF().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 7, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 7));

    callEntryMerge = jlm::rvsdg::node_output::node(test.CallG().input(4)->origin());
    callExitSplit = input_node(*test.CallG().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 7, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 7));
  }
}

static void
ValidateCallTest1SteensgaardRegionAware(const jlm::tests::CallTest1 & test)
{
  using namespace jlm::llvm;

  /* validate f */
  {
    auto lambdaEntrySplit = input_node(*test.lambda_f->fctargument(3)->begin());
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_f->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_f->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_f->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadY->input(1)->origin()) == lambdaEntrySplit);
  }

  /* validate g */
  {
    auto lambdaEntrySplit = input_node(*test.lambda_g->fctargument(3)->begin());
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_g->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_g->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_g->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge = jlm::rvsdg::node_output::node(test.CallF().input(4)->origin());
    auto callExitSplit = input_node(*test.CallF().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 2, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 2));

    callEntryMerge = jlm::rvsdg::node_output::node(test.CallG().input(4)->origin());
    callExitSplit = input_node(*test.CallG().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));
  }
}

static void
ValidateCallTest1SteensgaardAgnosticTopDown(const jlm::tests::CallTest1 & test)
{
  using namespace jlm::llvm;

  // validate function f
  {
    auto lambdaEntrySplit = input_node(*test.lambda_f->fctargument(3)->begin());
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_f->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_f->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_f->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadY->input(1)->origin()) == lambdaEntrySplit);
  }

  // validate function g
  {
    auto lambdaEntrySplit = input_node(*test.lambda_g->fctargument(3)->begin());
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_g->fctresult(2)->origin());
    auto loadX = input_node(*test.lambda_g->fctargument(0)->begin());
    auto loadY = input_node(*test.lambda_g->fctargument(1)->begin());

    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 7, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::node_output::node(loadY->input(1)->origin()) == loadX);
  }

  // validate function h
  {
    auto callEntryMerge = jlm::rvsdg::node_output::node(test.CallF().input(4)->origin());
    auto callExitSplit = input_node(*test.CallF().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 7, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 7));

    callEntryMerge = jlm::rvsdg::node_output::node(test.CallG().input(4)->origin());
    callExitSplit = input_node(*test.CallG().output(2)->begin());

    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 7, 1));
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 7));
  }
}

static void
ValidateCallTest2SteensgaardAgnostic(const jlm::tests::CallTest2 & test)
{
  using namespace jlm::llvm;

  /* validate create function */
  {
    assert(test.lambda_create->subregion()->nnodes() == 7);

    auto stateMerge = input_node(*test.malloc->output(1)->begin());
    assert(is<MemStateMergeOperator>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(stateMerge->input(1)->origin());
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
ValidateCallTest2SteensgaardRegionAware(const jlm::tests::CallTest2 & test)
{
  using namespace jlm::llvm;

  /* validate create function */
  {
    assert(test.lambda_create->subregion()->nnodes() == 7);

    auto stateMerge = input_node(*test.malloc->output(1)->begin());
    assert(is<MemStateMergeOperator>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(stateMerge->input(1)->origin());
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
ValidateCallTest2SteensgaardAgnosticTopDown(const jlm::tests::CallTest2 & test)
{
  using namespace jlm::llvm;

  // validate create function
  {
    assert(test.lambda_create->subregion()->nnodes() == 7);

    auto stateMerge = input_node(*test.malloc->output(1)->begin());
    assert(is<MemStateMergeOperator>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(stateMerge->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

    auto lambdaExitMerge = input_node(*stateMerge->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));
  }

  // validate destroy function
  {
    assert(test.lambda_destroy->subregion()->nnodes() == 4);
  }

  // validate test function
  {
    assert(test.lambda_test->subregion()->nnodes() == 16);
  }
}

static void
ValidateIndirectCallTest1SteensgaardAgnostic(const jlm::tests::IndirectCallTest1 & test)
{
  using namespace jlm::llvm;

  /* validate indcall function */
  {
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 5);

    auto lambda_exit_mux =
        jlm::rvsdg::node_output::node(test.GetLambdaIndcall().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::node_output::node(lambda_exit_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto call_entry_mux = jlm::rvsdg::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::node_output::node(call_entry_mux->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 5));
  }

  /* validate test function */
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 9);

    auto lambda_exit_mux =
        jlm::rvsdg::node_output::node(test.GetLambdaTest().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::node_output::node(lambda_exit_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto call_entry_mux = jlm::rvsdg::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    call_exit_mux = jlm::rvsdg::node_output::node(call_entry_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    call = jlm::rvsdg::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    call_entry_mux = jlm::rvsdg::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::node_output::node(call_entry_mux->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 5));
  }
}

static void
ValidateIndirectCallTest1SteensgaardRegionAware(const jlm::tests::IndirectCallTest1 & test)
{
  using namespace jlm::llvm;

  /* validate indcall function */
  {
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 5);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaIndcall().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto call = jlm::rvsdg::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }

  /* validate test function */
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 9);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaTest().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto call = jlm::rvsdg::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto callEntryMerge = jlm::rvsdg::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    callExitSplit = jlm::rvsdg::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    call = jlm::rvsdg::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    callEntryMerge = jlm::rvsdg::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }
}

static void
ValidateIndirectCallTest1SteensgaardAgnosticTopDown(const jlm::tests::IndirectCallTest1 & test)
{
  using namespace jlm::llvm;

  // validate indcall function
  {
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 5);

    auto lambda_exit_mux =
        jlm::rvsdg::node_output::node(test.GetLambdaIndcall().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::node_output::node(lambda_exit_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto call_entry_mux = jlm::rvsdg::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::node_output::node(call_entry_mux->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 5));
  }

  // validate test function
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 9);

    auto lambda_exit_mux =
        jlm::rvsdg::node_output::node(test.GetLambdaTest().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::node_output::node(lambda_exit_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto call_entry_mux = jlm::rvsdg::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    call_exit_mux = jlm::rvsdg::node_output::node(call_entry_mux->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*call_exit_mux, 1, 5));

    call = jlm::rvsdg::node_output::node(call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    call_entry_mux = jlm::rvsdg::node_output::node(call->input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::node_output::node(call_entry_mux->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 5));
  }
}

static void
ValidateIndirectCallTest2SteensgaardAgnostic(const jlm::tests::IndirectCallTest2 & test)
{
  using namespace jlm::llvm;

  // validate function three()
  {
    assert(test.GetLambdaThree().subregion()->nnodes() == 3);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaThree().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = input_node(*test.GetLambdaThree().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 13));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->nnodes() == 3);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaFour().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = input_node(*test.GetLambdaFour().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 13));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->nnodes() == 5);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.GetLambdaI().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 13, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 13));

    auto callEntryMerge = jlm::rvsdg::node_output::node(test.GetIndirectCall().input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 13));
  }
}

static void
ValidateIndirectCallTest2SteensgaardRegionAware(const jlm::tests::IndirectCallTest2 & test)
{
  using namespace jlm::llvm;

  // validate function three()
  {
    assert(test.GetLambdaThree().subregion()->nnodes() == 2);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaThree().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 0, 1));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->nnodes() == 2);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaFour().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 0, 1));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->nnodes() == 5);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.GetLambdaI().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 6));

    auto callEntryMerge = jlm::rvsdg::node_output::node(test.GetIndirectCall().input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 6, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 6));
  }

  // validate function x()
  {
    assert(test.GetLambdaX().subregion()->nnodes() == 7);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.GetLambdaX().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 6));

    auto callEntryMerge =
        jlm::rvsdg::node_output::node(test.GetCallIWithThree().input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 6, 1));

    const jlm::rvsdg::node * storeNode = nullptr;
    const jlm::rvsdg::node * lambdaEntrySplit = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::node_output::node(callEntryMerge->input(n)->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeNode = node;
      }
      else if (is<aa::LambdaEntryMemStateOperator>(node))
      {
        lambdaEntrySplit = node;
      }
      else
      {
        assert(0 && "This should not have happened!");
      }
    }
    assert(storeNode && lambdaEntrySplit);
    assert(is<StoreNonVolatileOperation>(*storeNode, 4, 2));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 6));
  }

  // validate function y()
  {
    assert(test.GetLambdaY().subregion()->nnodes() == 7);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.GetLambdaY().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 6));

    auto callEntryMerge = jlm::rvsdg::node_output::node(test.GetCallIWithFour().input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 6, 1));

    const jlm::rvsdg::node * storeNode = nullptr;
    const jlm::rvsdg::node * lambdaEntrySplit = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::node_output::node(callEntryMerge->input(n)->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeNode = node;
      }
      else if (is<aa::LambdaEntryMemStateOperator>(node))
      {
        lambdaEntrySplit = node;
      }
      else
      {
        assert(0 && "This should not have happened!");
      }
    }
    assert(storeNode && lambdaEntrySplit);
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 6));
  }

  // validate function test()
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 16);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaTest().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

    auto loadG1 = input_node(*test.GetLambdaTest().cvargument(2)->begin());
    assert(is<LoadNonVolatileOperation>(*loadG1, 2, 2));

    auto loadG2 = input_node(*test.GetLambdaTest().cvargument(3)->begin());
    assert(is<LoadNonVolatileOperation>(*loadG2, 2, 2));

    auto lambdaEntrySplit = input_node(*test.GetLambdaTest().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 6));
  }

  // validate function test2()
  {
    assert(test.GetLambdaTest2().subregion()->nnodes() == 7);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaTest2().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 6, 1));

    auto lambdaEntrySplit = input_node(*test.GetLambdaTest().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 6));
  }
}

static void
ValidateIndirectCallTest2SteensgaardAgnosticTopDown(const jlm::tests::IndirectCallTest2 & test)
{
  using namespace jlm::llvm;

  // validate function three()
  {
    assert(test.GetLambdaThree().subregion()->nnodes() == 3);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaThree().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = input_node(*test.GetLambdaThree().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 13));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->nnodes() == 3);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaFour().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = input_node(*test.GetLambdaFour().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 13));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->nnodes() == 5);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.GetLambdaI().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 13, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 13));

    auto callEntryMerge = jlm::rvsdg::node_output::node(test.GetIndirectCall().input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 13));
  }

  // validate function x()
  {
    assert(test.GetLambdaX().subregion()->nnodes() == 7);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.GetLambdaX().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 13, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 13));

    auto callEntryMerge =
        jlm::rvsdg::node_output::node(test.GetCallIWithThree().input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 13, 1));

    const jlm::rvsdg::node * storeNode = nullptr;
    const jlm::rvsdg::node * lambdaEntrySplit = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::node_output::node(callEntryMerge->input(n)->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeNode = node;
      }
      else if (is<aa::LambdaEntryMemStateOperator>(node))
      {
        lambdaEntrySplit = node;
      }
      else
      {
        assert(0 && "This should not have happened!");
      }
    }
    assert(storeNode && lambdaEntrySplit);
    assert(is<StoreNonVolatileOperation>(*storeNode, 4, 2));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 13));
  }

  // validate function y()
  {
    assert(test.GetLambdaY().subregion()->nnodes() == 8);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.GetLambdaY().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 12, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 13));

    auto callEntryMerge = jlm::rvsdg::node_output::node(test.GetCallIWithFour().input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 13, 1));

    jlm::rvsdg::node * undefNode = nullptr;
    const jlm::rvsdg::node * storeNode = nullptr;
    const jlm::rvsdg::node * lambdaEntrySplit = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::node_output::node(callEntryMerge->input(n)->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        assert(storeNode == nullptr);
        storeNode = node;
      }
      else if (is<aa::LambdaEntryMemStateOperator>(node))
      {
        lambdaEntrySplit = node;
      }
      else if (is<UndefValueOperation>(node))
      {
        assert(undefNode == nullptr);
        undefNode = node;
      }
      else
      {
        assert(0 && "This should not have happened!");
      }
    }
    assert(storeNode && lambdaEntrySplit && undefNode);
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 12));
  }

  // validate function test()
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 17);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaTest().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 10, 1));

    auto loadG1 = input_node(*test.GetLambdaTest().cvargument(2)->begin());
    assert(is<LoadNonVolatileOperation>(*loadG1, 2, 2));

    auto callXEntryMerge = jlm::rvsdg::node_output::node(test.GetTestCallX().input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callXEntryMerge, 13, 1));

    auto callXExitSplit = input_node(*test.GetTestCallX().output(2)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callXExitSplit, 1, 13));

    jlm::rvsdg::node * undefNode = nullptr;
    for (auto & node : test.GetLambdaTest().subregion()->nodes)
    {
      if (is<UndefValueOperation>(&node))
      {
        undefNode = &node;
        break;
      }
    }
    assert(undefNode != nullptr);
    assert(undefNode->output(0)->nusers() == 1);
    assert(input_node(*undefNode->output(0)->begin()) == callXEntryMerge);

    auto loadG2 = input_node(*test.GetLambdaTest().cvargument(3)->begin());
    assert(is<LoadNonVolatileOperation>(*loadG2, 2, 2));

    auto lambdaEntrySplit = input_node(*test.GetLambdaTest().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 10));
  }

  // validate function test2()
  {
    assert(test.GetLambdaTest2().subregion()->nnodes() == 8);

    auto lambdaExitMerge =
        jlm::rvsdg::node_output::node(test.GetLambdaTest2().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 10, 1));

    auto callXEntryMerge = jlm::rvsdg::node_output::node(test.GetTest2CallX().input(3)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callXEntryMerge, 13, 1));

    auto callXExitSplit = input_node(*test.GetTest2CallX().output(2)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callXExitSplit, 1, 13));

    jlm::rvsdg::node * undefNode = nullptr;
    for (auto & node : test.GetLambdaTest2().subregion()->nodes)
    {
      if (is<UndefValueOperation>(&node))
      {
        undefNode = &node;
        break;
      }
    }
    assert(undefNode != nullptr);
    assert(undefNode->output(0)->nusers() == 2);
    for (auto & user : *undefNode->output(0))
    {
      assert(input_node(user) == callXEntryMerge);
    }

    auto lambdaEntrySplit = input_node(*test.GetLambdaTest2().fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 10));
  }
}

static void
ValidateGammaTestSteensgaardAgnostic(const jlm::tests::GammaTest & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::node_output::node(loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto gamma = jlm::rvsdg::node_output::node(loadTmp1->input(1)->origin());
  assert(gamma == test.gamma);
}

static void
ValidateGammaTestSteensgaardRegionAware(const jlm::tests::GammaTest & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::node_output::node(loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto lambdaEntrySplit = jlm::rvsdg::node_output::node(loadTmp1->input(1)->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateGammaTestSteensgaardAgnosticTopDown(const jlm::tests::GammaTest & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::node_output::node(loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto gamma = jlm::rvsdg::node_output::node(loadTmp1->input(1)->origin());
  assert(gamma == test.gamma);
}

static void
ValidateThetaTestSteensgaardAgnostic(const jlm::tests::ThetaTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambda_exit_mux = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 2, 1));

  auto thetaOutput =
      jlm::util::AssertedCast<jlm::rvsdg::theta_output>(lambda_exit_mux->input(0)->origin());
  auto theta = jlm::rvsdg::node_output::node(thetaOutput);
  assert(theta == test.theta);

  auto storeStateOutput = thetaOutput->result()->origin();
  auto store = jlm::rvsdg::node_output::node(storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index() + 2)->origin() == thetaOutput->argument());

  auto lambda_entry_mux = jlm::rvsdg::node_output::node(thetaOutput->input()->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 2));
}

static void
ValidateThetaTestSteensgaardRegionAware(const jlm::tests::ThetaTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto thetaOutput =
      jlm::util::AssertedCast<jlm::rvsdg::theta_output>(lambdaExitMerge->input(0)->origin());
  auto theta = jlm::rvsdg::node_output::node(thetaOutput);
  assert(theta == test.theta);

  auto storeStateOutput = thetaOutput->result()->origin();
  auto store = jlm::rvsdg::node_output::node(storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index() + 2)->origin() == thetaOutput->argument());

  auto lambdaEntrySplit = jlm::rvsdg::node_output::node(thetaOutput->input()->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateThetaTestSteensgaardAgnosticTopDown(const jlm::tests::ThetaTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambda_exit_mux = jlm::rvsdg::node_output::node(test.lambda->fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambda_exit_mux, 2, 1));

  auto thetaOutput =
      jlm::util::AssertedCast<jlm::rvsdg::theta_output>(lambda_exit_mux->input(0)->origin());
  auto theta = jlm::rvsdg::node_output::node(thetaOutput);
  assert(theta == test.theta);

  auto storeStateOutput = thetaOutput->result()->origin();
  auto store = jlm::rvsdg::node_output::node(storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index() + 2)->origin() == thetaOutput->argument());

  auto lambda_entry_mux = jlm::rvsdg::node_output::node(thetaOutput->input()->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambda_entry_mux, 1, 2));
}

static void
ValidateDeltaTest1SteensgaardAgnostic(const jlm::tests::DeltaTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit = input_node(*test.lambda_h->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 4));

  auto storeF = input_node(*test.constantFive->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(jlm::rvsdg::node_output::node(storeF->input(2)->origin()) == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = input_node(*test.lambda_g->fctargument(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

static void
ValidateDeltaTest1SteensgaardRegionAware(const jlm::tests::DeltaTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit = input_node(*test.lambda_h->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));

  auto storeF = input_node(*test.constantFive->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(jlm::rvsdg::node_output::node(storeF->input(2)->origin()) == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = input_node(*test.lambda_g->fctargument(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

static void
ValidateDeltaTest1SteensgaardAgnosticTopDown(const jlm::tests::DeltaTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit = input_node(*test.lambda_h->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 4));

  auto storeF = input_node(*test.constantFive->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(jlm::rvsdg::node_output::node(storeF->input(2)->origin()) == lambdaEntrySplit);

  auto loadF = input_node(*test.lambda_g->fctargument(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
}

static void
ValidateDeltaTest2SteensgaardAgnostic(const jlm::tests::DeltaTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
}

static void
ValidateDeltaTest2SteensgaardRegionAware(const jlm::tests::DeltaTest2 & test)
{
  using namespace jlm::llvm;

  /* Validate f1() */
  {
    assert(test.lambda_f1->subregion()->nnodes() == 4);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_f1->fctresult(1)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto storeNode = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(storeNode->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = input_node(*test.lambda_f2->cvargument(0)->begin());
    assert(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    assert(jlm::rvsdg::node_output::node(storeD1->input(2)->origin()) == lambdaEntrySplit);

    auto storeD2 = input_node(*test.lambda_f2->cvargument(1)->begin());
    assert(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    assert(jlm::rvsdg::node_output::node(storeD2->input(2)->origin()) == lambdaEntrySplit);

    auto callEntryMerge = input_node(*storeD1->output(0)->begin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto callF1 = input_node(*callEntryMerge->output(0)->begin());
    assert(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit = input_node(*callF1->output(1)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto lambdaExitMerge = input_node(*callExitSplit->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidateDeltaTest2SteensgaardAgnosticTopDown(const jlm::tests::DeltaTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
}

static void
ValidateDeltaTest3SteensgaardAgnostic(const jlm::tests::DeltaTest3 & test)
{
  using namespace jlm::llvm;

  /* validate f() */
  {
    assert(test.LambdaF().subregion()->nnodes() == 6);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto truncNode = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<trunc_op>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::node_output::node(truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(loadG1Node->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

    jlm::rvsdg::node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::node_output::node(input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jlm::rvsdg::node_output::node(storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::node_output::node(loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

static void
ValidateDeltaTest3SteensgaardRegionAware(const jlm::tests::DeltaTest3 & test)
{
  using namespace jlm::llvm;

  /* validate f() */
  {
    assert(test.LambdaF().subregion()->nnodes() == 6);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

    auto truncNode = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<trunc_op>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::node_output::node(truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(loadG1Node->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

    jlm::rvsdg::node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::node_output::node(input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jlm::rvsdg::node_output::node(storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::node_output::node(loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

static void
ValidateDeltaTest3SteensgaardAgnosticTopDown(const jlm::tests::DeltaTest3 & test)
{
  using namespace jlm::llvm;

  // validate f()
  {
    assert(test.LambdaF().subregion()->nnodes() == 6);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto truncNode = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<trunc_op>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::node_output::node(truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(loadG1Node->input(1)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

    jlm::rvsdg::node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::node_output::node(input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jlm::rvsdg::node_output::node(storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::node_output::node(loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

static void
ValidateImportTestSteensgaardAgnostic(const jlm::tests::ImportTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
}

static void
ValidateImportTestSteensgaardRegionAware(const jlm::tests::ImportTest & test)
{
  using namespace jlm::llvm;

  /* Validate f1() */
  {
    assert(test.lambda_f1->subregion()->nnodes() == 4);

    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_f1->fctresult(1)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

    auto storeNode = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(storeNode->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = input_node(*test.lambda_f2->cvargument(0)->begin());
    assert(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    assert(jlm::rvsdg::node_output::node(storeD1->input(2)->origin()) == lambdaEntrySplit);

    auto storeD2 = input_node(*test.lambda_f2->cvargument(1)->begin());
    assert(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    assert(jlm::rvsdg::node_output::node(storeD2->input(2)->origin()) == lambdaEntrySplit);

    auto callEntryMerge = input_node(*storeD1->output(0)->begin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 1, 1));

    auto callF1 = input_node(*callEntryMerge->output(0)->begin());
    assert(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit = input_node(*callF1->output(1)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 1));

    auto lambdaExitMerge = input_node(*callExitSplit->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidateImportTestSteensgaardAgnosticTopDown(const jlm::tests::ImportTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit = input_node(*test.lambda_f2->fctargument(1)->begin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = input_node(*test.lambda_f2->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::node_output::node(storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  assert(storeD1InF2->output(0)->nusers() == 1);
  auto d1StateIndexEntry = (*storeD1InF2->output(0)->begin())->index();

  auto storeD1InF1 = input_node(*test.lambda_f1->cvargument(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));
  assert(d1StateIndexEntry == storeD1InF1->input(2)->origin()->index());
  assert(storeD1InF1->output(0)->nusers() == 1);
  auto d1StateIndexExit = (*storeD1InF1->output(0)->begin())->index();

  auto storeD2InF2 = input_node(*test.lambda_f2->cvargument(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndexExit != storeD2InF2->input(2)->origin()->index());
}

static void
ValidatePhiTestSteensgaardAgnostic(const jlm::tests::PhiTest1 & test)
{
  using namespace jlm::llvm;

  auto arrayStateIndex = (*test.alloca->output(1)->begin())->index();

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_fib->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 4, 1));

  auto store = jlm::rvsdg::node_output::node(lambdaExitMerge->input(arrayStateIndex)->origin());
  assert(is<StoreNonVolatileOperation>(*store, 3, 1));

  auto gamma = jlm::rvsdg::node_output::node(store->input(2)->origin());
  assert(gamma == test.gamma);

  auto gammaStateIndex = store->input(2)->origin()->index();

  auto load1 =
      jlm::rvsdg::node_output::node(test.gamma->exitvar(gammaStateIndex)->result(0)->origin());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::node_output::node(load1->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*load2, 2, 2));

  assert(load2->input(1)->origin()->index() == arrayStateIndex);
}

static void
ValidatePhiTestSteensgaardRegionAware(const jlm::tests::PhiTest1 & test)
{
  using namespace jlm::llvm;

  auto arrayStateIndex = (*test.alloca->output(1)->begin())->index();

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_fib->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 1, 1));

  auto store = jlm::rvsdg::node_output::node(lambdaExitMerge->input(arrayStateIndex)->origin());
  assert(is<StoreNonVolatileOperation>(*store, 3, 1));

  auto gamma = jlm::rvsdg::node_output::node(store->input(2)->origin());
  assert(gamma == test.gamma);

  auto gammaStateIndex = store->input(2)->origin()->index();

  auto load1 =
      jlm::rvsdg::node_output::node(test.gamma->exitvar(gammaStateIndex)->result(0)->origin());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::node_output::node(load1->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*load2, 2, 2));

  assert(load2->input(1)->origin()->index() == arrayStateIndex);
}

static void
ValidatePhiTestSteensgaardAgnosticTopDown(const jlm::tests::PhiTest1 & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.lambda_fib->fctresult(1)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 4, 1));

  const StoreNonVolatileNode * storeNode = nullptr;
  const jlm::rvsdg::gamma_node * gammaNode = nullptr;
  for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
  {
    auto node = jlm::rvsdg::node_output::node(lambdaExitMerge->input(n)->origin());
    if (auto castedStoreNode = dynamic_cast<const StoreNonVolatileNode *>(node))
    {
      storeNode = castedStoreNode;
    }
    else if (auto castedGammaNode = dynamic_cast<const jlm::rvsdg::gamma_node *>(node))
    {
      gammaNode = castedGammaNode;
    }
    else
    {
      assert(0 && "This should not have happened!");
    }
  }
  assert(gammaNode != nullptr && storeNode != nullptr);

  assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

  auto gammaStateIndex = storeNode->input(2)->origin()->index();

  auto load1 =
      jlm::rvsdg::node_output::node(test.gamma->exitvar(gammaStateIndex)->result(0)->origin());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::node_output::node(load1->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*load2, 2, 2));
}

static void
ValidateMemcpySteensgaardAgnostic(const jlm::tests::MemcpyTest & test)
{
  using namespace jlm::llvm;

  /*
   * Validate function f
   */
  {
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto load = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<LoadNonVolatileOperation>(*load, 3, 3));

    auto store = jlm::rvsdg::node_output::node(load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 4, 2));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(store->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));
  }

  /*
   * Validate function g
   */
  {
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaG().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 5));

    auto call = jlm::rvsdg::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 5, 1));

    jlm::rvsdg::node * memcpy = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::node_output::node(callEntryMerge->input(n)->origin());
      if (is<MemCpyNonVolatileOperation>(node))
        memcpy = node;
    }
    assert(memcpy != nullptr);
    assert(is<MemCpyNonVolatileOperation>(*memcpy, 7, 4));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(memcpy->input(5)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));
  }
}

static void
ValidateMemcpySteensgaardRegionAware(const jlm::tests::MemcpyTest & test)
{
  using namespace jlm::llvm;

  /*
   * Validate function f
   */
  {
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

    auto load = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<LoadNonVolatileOperation>(*load, 3, 3));

    auto store = jlm::rvsdg::node_output::node(load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 4, 2));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(store->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
  }

  /*
   * Validate function g
   */
  {
    auto callNode = input_node(*test.LambdaG().cvargument(2)->begin());
    assert(is<CallOperation>(*callNode, 3, 3));

    auto callEntryMerge = jlm::rvsdg::node_output::node(callNode->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 2, 1));

    auto callExitSplit = input_node(*callNode->output(2)->begin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 2));

    auto memcpyNode = jlm::rvsdg::node_output::node(callEntryMerge->input(0)->origin());
    assert(is<MemCpyNonVolatileOperation>(*memcpyNode, 7, 4));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(memcpyNode->input(4)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
    assert(jlm::rvsdg::node_output::node(memcpyNode->input(5)->origin()) == lambdaEntrySplit);

    auto lambdaExitMerge = input_node(*callExitSplit->output(0)->begin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidateMemcpyTestSteensgaardAgnosticTopDown(const jlm::tests::MemcpyTest & test)
{
  using namespace jlm::llvm;

  // Validate function f
  {
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto load = jlm::rvsdg::node_output::node(test.LambdaF().fctresult(0)->origin());
    assert(is<LoadNonVolatileOperation>(*load, 3, 3));

    auto store = jlm::rvsdg::node_output::node(load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 4, 2));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(store->input(2)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));
  }

  // Validate function g
  {
    auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaG().fctresult(2)->origin());
    assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 5, 1));

    auto callExitSplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
    assert(is<aa::CallExitMemStateOperator>(*callExitSplit, 1, 5));

    auto call = jlm::rvsdg::node_output::node(callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::node_output::node(call->input(2)->origin());
    assert(is<aa::CallEntryMemStateOperator>(*callEntryMerge, 5, 1));

    jlm::rvsdg::node * memcpy = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::node_output::node(callEntryMerge->input(n)->origin());
      if (is<MemCpyNonVolatileOperation>(node))
        memcpy = node;
    }
    assert(memcpy != nullptr);
    assert(is<MemCpyNonVolatileOperation>(*memcpy, 7, 4));

    auto lambdaEntrySplit = jlm::rvsdg::node_output::node(memcpy->input(5)->origin());
    assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 5));
  }
}

static void
ValidateFreeNullTestSteensgaardAgnostic(const jlm::tests::FreeNullTest & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge = jlm::rvsdg::node_output::node(test.LambdaMain().fctresult(0)->origin());
  assert(is<aa::LambdaExitMemStateOperator>(*lambdaExitMerge, 2, 1));

  auto free = jlm::rvsdg::node_output::node(test.LambdaMain().fctresult(1)->origin());
  assert(is<FreeOperation>(*free, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::node_output::node(lambdaExitMerge->input(0)->origin());
  assert(is<aa::LambdaEntryMemStateOperator>(*lambdaEntrySplit, 1, 2));
}

static int
TestMemoryStateEncoder()
{
  using namespace jlm::llvm::aa;

  ValidateTest<jlm::tests::StoreTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateStoreTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::StoreTest1, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateStoreTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::StoreTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateStoreTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::StoreTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateStoreTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::StoreTest2, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateStoreTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::StoreTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateStoreTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::LoadTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateLoadTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::LoadTest1, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateLoadTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::LoadTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateLoadTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::LoadTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateLoadTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::LoadTest2, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateLoadTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::LoadTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateLoadTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::LoadFromUndefTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateLoadFromUndefSteensgaardAgnostic);
  ValidateTest<jlm::tests::LoadFromUndefTest, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateLoadFromUndefSteensgaardRegionAware);
  ValidateTest<jlm::tests::LoadFromUndefTest, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateLoadFromUndefSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::CallTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateCallTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::CallTest1, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateCallTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::CallTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateCallTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::CallTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateCallTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::CallTest2, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateCallTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::CallTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateCallTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::IndirectCallTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateIndirectCallTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::IndirectCallTest1, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateIndirectCallTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::IndirectCallTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateIndirectCallTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::IndirectCallTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateIndirectCallTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::IndirectCallTest2, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateIndirectCallTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::IndirectCallTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateIndirectCallTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::GammaTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateGammaTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::GammaTest, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateGammaTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::GammaTest, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateGammaTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::ThetaTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateThetaTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::ThetaTest, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateThetaTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::ThetaTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateThetaTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::DeltaTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateDeltaTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::DeltaTest1, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateDeltaTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::DeltaTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateDeltaTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::DeltaTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateDeltaTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::DeltaTest2, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateDeltaTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::DeltaTest2, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateDeltaTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::DeltaTest3, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateDeltaTest3SteensgaardAgnostic);
  ValidateTest<jlm::tests::DeltaTest3, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateDeltaTest3SteensgaardRegionAware);
  ValidateTest<jlm::tests::DeltaTest3, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateDeltaTest3SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::ImportTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateImportTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::ImportTest, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateImportTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::ImportTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateImportTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::PhiTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidatePhiTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::PhiTest1, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidatePhiTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::PhiTest1, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidatePhiTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::MemcpyTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateMemcpySteensgaardAgnostic);
  ValidateTest<jlm::tests::MemcpyTest, Steensgaard, RegionAwareMemoryNodeProvider>(
      ValidateMemcpySteensgaardRegionAware);
  ValidateTest<jlm::tests::MemcpyTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateMemcpyTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::FreeNullTest, Steensgaard, AgnosticMemoryNodeProvider>(
      ValidateFreeNullTestSteensgaardAgnostic);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder", TestMemoryStateEncoder)
