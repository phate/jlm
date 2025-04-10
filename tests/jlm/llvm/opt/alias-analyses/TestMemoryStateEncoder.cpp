/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "TestRvsdgs.hpp"

#include <test-registry.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/EliminatedModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownModRefEliminator.hpp>

#include <iostream>

using AgnosticTopDownMemoryNodeProvider = jlm::llvm::aa::EliminatedModRefSummarizer<
    jlm::llvm::aa::AgnosticModRefSummarizer,
    jlm::llvm::aa::TopDownModRefEliminator>;

template<class Test, class Analysis, class TModRefSummarizer>
static void
ValidateTest(std::function<void(const Test &)> validateEncoding)
{
  static_assert(
      std::is_base_of<jlm::tests::RvsdgTest, Test>::value,
      "Test should be derived from RvsdgTest class.");

  static_assert(
      std::is_base_of_v<jlm::llvm::aa::PointsToAnalysis, Analysis>,
      "Analysis should be derived from PointsToAnalysis class.");

  static_assert(
      std::is_base_of_v<jlm::llvm::aa::ModRefSummarizer, TModRefSummarizer>,
      "TModRefSummarizer should be derived from ModRefSummarizer class.");

  std::cout << "\n###\n";
  std::cout << "### Performing Test " << typeid(Test).name() << " using ["
            << typeid(Analysis).name() << ", " << typeid(TModRefSummarizer).name() << "]\n";
  std::cout << "###\n";

  Test test;
  auto & rvsdgModule = test.module();
  jlm::rvsdg::view(&rvsdgModule.Rvsdg().GetRootRegion(), stdout);

  jlm::util::StatisticsCollector statisticsCollector;

  Analysis aliasAnalysis;
  auto pointsToGraph = aliasAnalysis.Analyze(rvsdgModule, statisticsCollector);
  std::cout << jlm::llvm::aa::PointsToGraph::ToDot(*pointsToGraph);

  TModRefSummarizer summarizer;
  auto modRefSummary =
      summarizer.SummarizeModRefs(rvsdgModule, *pointsToGraph, statisticsCollector);

  jlm::llvm::aa::MemoryStateEncoder encoder;
  encoder.Encode(rvsdgModule, *modRefSummary, statisticsCollector);
  jlm::rvsdg::view(&rvsdgModule.Rvsdg().GetRootRegion(), stdout);

  validateEncoding(test);
}

template<class OP>
static bool
is(const jlm::rvsdg::Node & node, size_t numInputs, size_t numOutputs)
{
  return jlm::rvsdg::is<OP>(&node) && node.ninputs() == numInputs && node.noutputs() == numOutputs;
}

static void
ValidateStoreTest1SteensgaardAgnostic(const jlm::tests::StoreTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 10);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 6, 1));

  assert(test.alloca_d->output(1)->nusers() == 1);
  assert(test.alloca_c->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_a->output(1)->nusers() == 1);

  assert(jlm::rvsdg::input::GetNode((**test.alloca_d->output(1)->begin())) == lambdaExitMerge);

  auto storeD = jlm::rvsdg::input::GetNode(**test.alloca_c->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD, 3, 1));
  assert(storeD->input(0)->origin() == test.alloca_c->output(0));
  assert(storeD->input(1)->origin() == test.alloca_d->output(0));

  auto storeC = jlm::rvsdg::input::GetNode(**test.alloca_b->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeC, 3, 1));
  assert(storeC->input(0)->origin() == test.alloca_b->output(0));
  assert(storeC->input(1)->origin() == test.alloca_c->output(0));

  auto storeB = jlm::rvsdg::input::GetNode(**test.alloca_a->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_a->output(0));
  assert(storeB->input(1)->origin() == test.alloca_b->output(0));
}

static void
ValidateStoreTest1SteensgaardRegionAware(const jlm::tests::StoreTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 9);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 4, 1));

  assert(test.alloca_d->output(1)->nusers() == 1);
  assert(test.alloca_c->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_a->output(1)->nusers() == 1);

  assert(jlm::rvsdg::input::GetNode((**test.alloca_d->output(1)->begin())) == lambdaExitMerge);

  auto storeD = jlm::rvsdg::input::GetNode(**test.alloca_c->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeD, 3, 1));
  assert(storeD->input(0)->origin() == test.alloca_c->output(0));
  assert(storeD->input(1)->origin() == test.alloca_d->output(0));

  auto storeC = jlm::rvsdg::input::GetNode(**test.alloca_b->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeC, 3, 1));
  assert(storeC->input(0)->origin() == test.alloca_b->output(0));
  assert(storeC->input(1)->origin() == test.alloca_c->output(0));

  auto storeB = jlm::rvsdg::input::GetNode(**test.alloca_a->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_a->output(0));
  assert(storeB->input(1)->origin() == test.alloca_b->output(0));
}

static void
ValidateStoreTest1SteensgaardAgnosticTopDown(const jlm::tests::StoreTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 2);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
  assert(lambdaEntrySplit == jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(1)->origin()));
}

static void
ValidateStoreTest2SteensgaardAgnostic(const jlm::tests::StoreTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 12);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(jlm::rvsdg::input::GetNode((**test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(jlm::rvsdg::input::GetNode((**test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = jlm::rvsdg::input::GetNode(**test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = jlm::rvsdg::input::GetNode(**test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::output::GetNode(*storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::output::GetNode(*storeB->input(3)->origin()) == storeA);

  auto storeX = jlm::rvsdg::input::GetNode(**test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto storeY = jlm::rvsdg::input::GetNode(**storeX->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  assert(storeY->input(0)->origin() == test.alloca_p->output(0));
  assert(storeY->input(1)->origin() == test.alloca_y->output(0));
}

static void
ValidateStoreTest2SteensgaardRegionAware(const jlm::tests::StoreTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 11);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(jlm::rvsdg::input::GetNode((**test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(jlm::rvsdg::input::GetNode((**test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = jlm::rvsdg::input::GetNode(**test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = jlm::rvsdg::input::GetNode(**test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::output::GetNode(*storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::output::GetNode(*storeB->input(3)->origin()) == storeA);

  auto storeX = jlm::rvsdg::input::GetNode(**test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto storeY = jlm::rvsdg::input::GetNode(**storeX->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  assert(storeY->input(0)->origin() == test.alloca_p->output(0));
  assert(storeY->input(1)->origin() == test.alloca_y->output(0));
}

static void
ValidateStoreTest2SteensgaardAgnosticTopDown(const jlm::tests::StoreTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 2);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
  assert(lambdaEntrySplit == jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(1)->origin()));
}

static void
ValidateLoadTest1SteensgaardAgnostic(const jlm::tests::LoadTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  auto loadX = jlm::rvsdg::output::GetNode(*loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  assert(jlm::rvsdg::output::GetNode(*loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->GetFunctionArguments()[0]);
  assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);
}

static void
ValidateLoadTest1SteensgaardRegionAware(const jlm::tests::LoadTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  auto loadX = jlm::rvsdg::output::GetNode(*loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  assert(jlm::rvsdg::output::GetNode(*loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->GetFunctionArguments()[0]);
  assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);
}

static void
ValidateLoadTest1SteensgaardAgnosticTopDown(const jlm::tests::LoadTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  auto loadX = jlm::rvsdg::output::GetNode(*loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  assert(jlm::rvsdg::output::GetNode(*loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->GetFunctionArguments()[0]);
  assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);
}

static void
ValidateLoadTest2SteensgaardAgnostic(const jlm::tests::LoadTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 14);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  assert(jlm::rvsdg::input::GetNode((**test.alloca_a->output(1)->begin())) == lambdaExitMerge);
  assert(jlm::rvsdg::input::GetNode((**test.alloca_b->output(1)->begin())) == lambdaExitMerge);

  auto storeA = jlm::rvsdg::input::GetNode(**test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = jlm::rvsdg::input::GetNode(**test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::output::GetNode(*storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::output::GetNode(*storeB->input(3)->origin()) == storeA);

  auto storeX = jlm::rvsdg::input::GetNode(**test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto loadP = jlm::rvsdg::input::GetNode(**storeX->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadP, 2, 2));
  assert(loadP->input(0)->origin() == test.alloca_p->output(0));

  auto loadXY = jlm::rvsdg::input::GetNode(**loadP->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadXY, 3, 3));
  assert(jlm::rvsdg::output::GetNode(*loadXY->input(1)->origin()) == storeB);
  assert(jlm::rvsdg::output::GetNode(*loadXY->input(2)->origin()) == storeB);

  auto storeY = jlm::rvsdg::input::GetNode(**loadXY->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 4, 2));
  assert(storeY->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::output::GetNode(*storeY->input(2)->origin()) == loadXY);
  assert(jlm::rvsdg::output::GetNode(*storeY->input(3)->origin()) == loadXY);
}

static void
ValidateLoadTest2SteensgaardRegionAware(const jlm::tests::LoadTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 13);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

  assert(test.alloca_a->output(1)->nusers() == 1);
  assert(test.alloca_b->output(1)->nusers() == 1);
  assert(test.alloca_x->output(1)->nusers() == 1);
  assert(test.alloca_y->output(1)->nusers() == 1);
  assert(test.alloca_p->output(1)->nusers() == 1);

  auto storeA = jlm::rvsdg::input::GetNode(**test.alloca_a->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeA, 4, 2));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));

  auto storeB = jlm::rvsdg::input::GetNode(**test.alloca_b->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeB, 4, 2));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::output::GetNode(*storeB->input(2)->origin()) == storeA);
  assert(jlm::rvsdg::output::GetNode(*storeB->input(3)->origin()) == storeA);

  auto storeX = jlm::rvsdg::input::GetNode(**test.alloca_p->output(1)->begin());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));

  auto loadP = jlm::rvsdg::input::GetNode(**storeX->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadP, 2, 2));
  assert(loadP->input(0)->origin() == test.alloca_p->output(0));

  auto loadXY = jlm::rvsdg::input::GetNode(**loadP->output(0)->begin());
  assert(is<LoadNonVolatileOperation>(*loadXY, 3, 3));
  assert(jlm::rvsdg::output::GetNode(*loadXY->input(1)->origin()) == storeB);
  assert(jlm::rvsdg::output::GetNode(*loadXY->input(2)->origin()) == storeB);

  auto storeY = jlm::rvsdg::input::GetNode(**loadXY->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeY, 4, 2));
  assert(storeY->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::output::GetNode(*storeY->input(2)->origin()) == loadXY);
  assert(jlm::rvsdg::output::GetNode(*storeY->input(3)->origin()) == loadXY);
}

static void
ValidateLoadTest2SteensgaardAgnosticTopDown(const jlm::tests::LoadTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 2);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
  assert(lambdaEntrySplit == jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(1)->origin()));
}

static void
ValidateLoadFromUndefSteensgaardAgnostic(const jlm::tests::LoadFromUndefTest & test)
{
  using namespace jlm::llvm;

  assert(test.Lambda().subregion()->nnodes() == 4);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.Lambda().GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto load = jlm::rvsdg::output::GetNode(*test.Lambda().GetFunctionResults()[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.Lambda().GetFunctionArguments()[0]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateLoadFromUndefSteensgaardRegionAware(const jlm::tests::LoadFromUndefTest & test)
{
  using namespace jlm::llvm;

  assert(test.Lambda().subregion()->nnodes() == 3);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.Lambda().GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

  auto load = jlm::rvsdg::output::GetNode(*test.Lambda().GetFunctionResults()[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));
}

static void
ValidateLoadFromUndefSteensgaardAgnosticTopDown(const jlm::tests::LoadFromUndefTest & test)
{
  using namespace jlm::llvm;

  assert(test.Lambda().subregion()->nnodes() == 4);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.Lambda().GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto load = jlm::rvsdg::output::GetNode(*test.Lambda().GetFunctionResults()[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.Lambda().GetFunctionArguments()[0]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateCallTest1SteensgaardAgnostic(const jlm::tests::CallTest1 & test)
{
  using namespace jlm::llvm;

  /* validate f */
  {
    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[3]->begin());
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_f->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[0]->begin());
    auto loadY = jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[1]->begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadY->input(1)->origin()) == lambdaEntrySplit);
  }

  /* validate g */
  {
    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[3]->begin());
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_g->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[0]->begin());
    auto loadY = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[1]->begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.CallF().input(4)->origin());
    auto callExitSplit = jlm::rvsdg::input::GetNode(**test.CallF().output(2)->begin());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));

    callEntryMerge = jlm::rvsdg::output::GetNode(*test.CallG().input(4)->origin());
    callExitSplit = jlm::rvsdg::input::GetNode(**test.CallG().output(2)->begin());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));
  }
}

static void
ValidateCallTest1SteensgaardRegionAware(const jlm::tests::CallTest1 & test)
{
  using namespace jlm::llvm;

  /* validate f */
  {
    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[3]->begin());
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_f->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[0]->begin());
    auto loadY = jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[1]->begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadY->input(1)->origin()) == lambdaEntrySplit);
  }

  /* validate g */
  {
    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[3]->begin());
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_g->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[0]->begin());
    auto loadY = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[1]->begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.CallF().input(4)->origin());
    auto callExitSplit = jlm::rvsdg::input::GetNode(**test.CallF().output(2)->begin());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 2, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 2));

    callEntryMerge = jlm::rvsdg::output::GetNode(*test.CallG().input(4)->origin());
    callExitSplit = jlm::rvsdg::input::GetNode(**test.CallG().output(2)->begin());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));
  }
}

static void
ValidateCallTest1SteensgaardAgnosticTopDown(const jlm::tests::CallTest1 & test)
{
  using namespace jlm::llvm;

  // validate function f
  {
    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[3]->begin());
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_f->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[0]->begin());
    auto loadY = jlm::rvsdg::input::GetNode(**test.lambda_f->GetFunctionArguments()[1]->begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadY->input(1)->origin()) == lambdaEntrySplit);
  }

  // validate function g
  {
    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[3]->begin());
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_g->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[0]->begin());
    auto loadY = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[1]->begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadX->input(1)->origin()) == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::output::GetNode(*loadY->input(1)->origin()) == loadX);
  }

  // validate function h
  {
    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.CallF().input(4)->origin());
    auto callExitSplit = jlm::rvsdg::input::GetNode(**test.CallF().output(2)->begin());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));

    callEntryMerge = jlm::rvsdg::output::GetNode(*test.CallG().input(4)->origin());
    callExitSplit = jlm::rvsdg::input::GetNode(**test.CallG().output(2)->begin());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));
  }
}

static void
ValidateCallTest2SteensgaardAgnostic(const jlm::tests::CallTest2 & test)
{
  using namespace jlm::llvm;

  /* validate create function */
  {
    assert(test.lambda_create->subregion()->nnodes() == 7);

    auto stateMerge = jlm::rvsdg::input::GetNode(**test.malloc->output(1)->begin());
    assert(is<MemoryStateMergeOperation>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*stateMerge->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

    auto lambdaExitMerge = jlm::rvsdg::input::GetNode(**stateMerge->output(0)->begin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

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

    auto stateMerge = jlm::rvsdg::input::GetNode(**test.malloc->output(1)->begin());
    assert(is<MemoryStateMergeOperation>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*stateMerge->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

    auto lambdaExitMerge = jlm::rvsdg::input::GetNode(**stateMerge->output(0)->begin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

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

    auto stateMerge = jlm::rvsdg::input::GetNode(**test.malloc->output(1)->begin());
    assert(is<MemoryStateMergeOperation>(*stateMerge, 2, 1));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*stateMerge->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

    auto lambdaExitMerge = jlm::rvsdg::input::GetNode(**stateMerge->output(0)->begin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));
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
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 6);

    auto lambda_exit_mux =
        jlm::rvsdg::output::GetNode(*test.GetLambdaIndcall().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::output::GetNode(*lambda_exit_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::output::GetNode(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto call_entry_mux = jlm::rvsdg::output::GetNode(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::output::GetNode(*call_entry_mux->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
  }

  /* validate test function */
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 9);

    auto lambda_exit_mux =
        jlm::rvsdg::output::GetNode(*test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::output::GetNode(*lambda_exit_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::output::GetNode(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto call_entry_mux = jlm::rvsdg::output::GetNode(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    call_exit_mux = jlm::rvsdg::output::GetNode(*call_entry_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    call = jlm::rvsdg::output::GetNode(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    call_entry_mux = jlm::rvsdg::output::GetNode(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::output::GetNode(*call_entry_mux->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
  }
}

static void
ValidateIndirectCallTest1SteensgaardRegionAware(const jlm::tests::IndirectCallTest1 & test)
{
  using namespace jlm::llvm;

  /* validate indcall function */
  {
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 4);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaIndcall().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto call =
        jlm::rvsdg::output::GetNode(*test.GetLambdaIndcall().GetFunctionResults()[0]->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  /* validate test function */
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 6);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto add = jlm::rvsdg::output::GetNode(*test.GetLambdaTest().GetFunctionResults()[0]->origin());
    assert(is<jlm::rvsdg::BinaryOperation>(*add, 2, 1));

    auto call = jlm::rvsdg::output::GetNode(*add->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));

    call = jlm::rvsdg::output::GetNode(*add->input(1)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    callEntryMerge = jlm::rvsdg::output::GetNode(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }
}

static void
ValidateIndirectCallTest1SteensgaardAgnosticTopDown(const jlm::tests::IndirectCallTest1 & test)
{
  using namespace jlm::llvm;

  // validate indcall function
  {
    assert(test.GetLambdaIndcall().subregion()->nnodes() == 6);

    auto lambda_exit_mux =
        jlm::rvsdg::output::GetNode(*test.GetLambdaIndcall().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::output::GetNode(*lambda_exit_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::output::GetNode(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto call_entry_mux = jlm::rvsdg::output::GetNode(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::output::GetNode(*call_entry_mux->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
  }

  // validate test function
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 9);

    auto lambda_exit_mux =
        jlm::rvsdg::output::GetNode(*test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux = jlm::rvsdg::output::GetNode(*lambda_exit_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::output::GetNode(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto call_entry_mux = jlm::rvsdg::output::GetNode(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    call_exit_mux = jlm::rvsdg::output::GetNode(*call_entry_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    call = jlm::rvsdg::output::GetNode(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    call_entry_mux = jlm::rvsdg::output::GetNode(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux = jlm::rvsdg::output::GetNode(*call_entry_mux->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
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
        jlm::rvsdg::output::GetNode(*test.GetLambdaThree().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaThree().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->nnodes() == 3);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaFour().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaFour().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->nnodes() == 6);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaI().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto callExitSplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 13));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.GetIndirectCall().input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*callEntryMerge->input(0)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
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
        jlm::rvsdg::output::GetNode(*test.GetLambdaThree().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->nnodes() == 2);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaFour().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->nnodes() == 4);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaI().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.GetIndirectCall().input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function x()
  {
    assert(test.GetLambdaX().subregion()->nnodes() == 7);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaX().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.GetCallIWithThree().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function y()
  {
    assert(test.GetLambdaY().subregion()->nnodes() == 7);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaY().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.GetCallIWithFour().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function test()
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 16);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto loadG1 =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest().GetContextVars()[2].inner->begin());
    assert(is<LoadNonVolatileOperation>(*loadG1, 2, 2));

    auto loadG2 =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest().GetContextVars()[3].inner->begin());
    assert(is<LoadNonVolatileOperation>(*loadG2, 2, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }

  // validate function test2()
  {
    assert(test.GetLambdaTest2().subregion()->nnodes() == 7);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaTest2().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest2().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
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
        jlm::rvsdg::output::GetNode(*test.GetLambdaThree().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaThree().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->nnodes() == 3);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaFour().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaFour().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->nnodes() == 6);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaI().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto callExitSplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 13));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.GetIndirectCall().input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*callEntryMerge->input(0)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function x()
  {
    assert(test.GetLambdaX().subregion()->nnodes() == 8);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaX().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto callExitSplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 13));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.GetCallIWithThree().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 13, 1));

    const jlm::rvsdg::Node * storeNode = nullptr;
    const jlm::rvsdg::Node * lambdaEntrySplit = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::output::GetNode(*callEntryMerge->input(n)->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeNode = node;
      }
      else if (is<LambdaEntryMemoryStateSplitOperation>(node))
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
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function y()
  {
    assert(test.GetLambdaY().subregion()->nnodes() == 9);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaY().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 12, 1));

    auto callExitSplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 13));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*test.GetCallIWithFour().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 13, 1));

    jlm::rvsdg::Node * undefNode = nullptr;
    const jlm::rvsdg::Node * storeNode = nullptr;
    const jlm::rvsdg::Node * lambdaEntrySplit = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::output::GetNode(*callEntryMerge->input(n)->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        assert(storeNode == nullptr);
        storeNode = node;
      }
      else if (is<LambdaEntryMemoryStateSplitOperation>(node))
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
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 12));
  }

  // validate function test()
  {
    assert(test.GetLambdaTest().subregion()->nnodes() == 17);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 10, 1));

    auto loadG1 =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest().GetContextVars()[2].inner->begin());
    assert(is<LoadNonVolatileOperation>(*loadG1, 2, 2));

    auto callXEntryMerge = jlm::rvsdg::output::GetNode(*test.GetTestCallX().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callXEntryMerge, 13, 1));

    auto callXExitSplit = jlm::rvsdg::input::GetNode(**test.GetTestCallX().output(2)->begin());
    assert(is<CallExitMemoryStateSplitOperation>(*callXExitSplit, 1, 13));

    jlm::rvsdg::Node * undefNode = nullptr;
    for (auto & node : test.GetLambdaTest().subregion()->Nodes())
    {
      if (is<UndefValueOperation>(&node))
      {
        undefNode = &node;
        break;
      }
    }
    assert(undefNode != nullptr);
    assert(undefNode->output(0)->nusers() == 1);
    assert(jlm::rvsdg::input::GetNode(**undefNode->output(0)->begin()) == callXEntryMerge);

    auto loadG2 =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest().GetContextVars()[3].inner->begin());
    assert(is<LoadNonVolatileOperation>(*loadG2, 2, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 10));
  }

  // validate function test2()
  {
    assert(test.GetLambdaTest2().subregion()->nnodes() == 8);

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.GetLambdaTest2().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 10, 1));

    auto callXEntryMerge = jlm::rvsdg::output::GetNode(*test.GetTest2CallX().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callXEntryMerge, 13, 1));

    auto callXExitSplit = jlm::rvsdg::input::GetNode(**test.GetTest2CallX().output(2)->begin());
    assert(is<CallExitMemoryStateSplitOperation>(*callXExitSplit, 1, 13));

    jlm::rvsdg::Node * undefNode = nullptr;
    for (auto & node : test.GetLambdaTest2().subregion()->Nodes())
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
      assert(jlm::rvsdg::input::GetNode(*user) == callXEntryMerge);
    }

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.GetLambdaTest2().GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 10));
  }
}

static void
ValidateGammaTestSteensgaardAgnostic(const jlm::tests::GammaTest & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::output::GetNode(*loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto gamma = jlm::rvsdg::output::GetNode(*loadTmp1->input(1)->origin());
  assert(gamma == test.gamma);
}

static void
ValidateGammaTestSteensgaardRegionAware(const jlm::tests::GammaTest & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::output::GetNode(*loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*loadTmp1->input(1)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateGammaTestSteensgaardAgnosticTopDown(const jlm::tests::GammaTest & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::output::GetNode(*loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto gamma = jlm::rvsdg::output::GetNode(*loadTmp1->input(1)->origin());
  assert(gamma == test.gamma);
}

static void
ValidateThetaTestSteensgaardAgnostic(const jlm::tests::ThetaTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambda_exit_mux =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 2, 1));

  auto thetaOutput = lambda_exit_mux->input(0)->origin();
  auto theta = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*thetaOutput);
  assert(theta == test.theta);

  auto loopvar = theta->MapOutputLoopVar(*thetaOutput);
  auto storeStateOutput = loopvar.post->origin();
  auto store = jlm::rvsdg::output::GetNode(*storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index() + 2)->origin() == loopvar.pre);

  auto lambda_entry_mux = jlm::rvsdg::output::GetNode(*loopvar.input->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 2));
}

static void
ValidateThetaTestSteensgaardRegionAware(const jlm::tests::ThetaTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto thetaOutput = lambdaExitMerge->input(0)->origin();
  auto theta = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*thetaOutput);
  assert(theta == test.theta);
  auto loopvar = theta->MapOutputLoopVar(*thetaOutput);

  auto storeStateOutput = loopvar.post->origin();
  auto store = jlm::rvsdg::output::GetNode(*storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index() + 2)->origin() == loopvar.pre);

  auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*loopvar.input->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

static void
ValidateThetaTestSteensgaardAgnosticTopDown(const jlm::tests::ThetaTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda->subregion()->nnodes() == 4);

  auto lambda_exit_mux =
      jlm::rvsdg::output::GetNode(*test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 2, 1));

  auto thetaOutput = lambda_exit_mux->input(0)->origin();
  auto theta = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*thetaOutput);
  assert(theta == test.theta);
  auto loopvar = theta->MapOutputLoopVar(*thetaOutput);

  auto storeStateOutput = loopvar.post->origin();
  auto store = jlm::rvsdg::output::GetNode(*storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index() + 2)->origin() == loopvar.pre);

  auto lambda_entry_mux = jlm::rvsdg::output::GetNode(*loopvar.input->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 2));
}

static void
ValidateDeltaTest1SteensgaardAgnostic(const jlm::tests::DeltaTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda_h->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 4));

  auto storeF = jlm::rvsdg::input::GetNode(**test.constantFive->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(jlm::rvsdg::output::GetNode(*storeF->input(2)->origin()) == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[0]->begin());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

static void
ValidateDeltaTest1SteensgaardRegionAware(const jlm::tests::DeltaTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda_h->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

  auto storeF = jlm::rvsdg::input::GetNode(**test.constantFive->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(jlm::rvsdg::output::GetNode(*storeF->input(2)->origin()) == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[0]->begin());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

static void
ValidateDeltaTest1SteensgaardAgnosticTopDown(const jlm::tests::DeltaTest1 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_h->subregion()->nnodes() == 7);

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda_h->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 4));

  auto storeF = jlm::rvsdg::input::GetNode(**test.constantFive->output(0)->begin());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(jlm::rvsdg::output::GetNode(*storeF->input(2)->origin()) == lambdaEntrySplit);

  auto loadF = jlm::rvsdg::input::GetNode(**test.lambda_g->GetFunctionArguments()[0]->begin());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
}

static void
ValidateDeltaTest2SteensgaardAgnostic(const jlm::tests::DeltaTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::output::GetNode(*storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 =
      jlm::rvsdg::input::GetNode(**test.lambda_f1->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[1].inner->begin());
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

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_f1->GetFunctionResults()[1]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto storeNode = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*storeNode->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_f2->GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[0].inner->begin());
    assert(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    assert(jlm::rvsdg::output::GetNode(*storeD1->input(2)->origin()) == lambdaEntrySplit);

    auto storeD2 = jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[1].inner->begin());
    assert(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    assert(jlm::rvsdg::output::GetNode(*storeD2->input(2)->origin()) == lambdaEntrySplit);

    auto callEntryMerge = jlm::rvsdg::input::GetNode(**storeD1->output(0)->begin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callF1 = jlm::rvsdg::input::GetNode(**callEntryMerge->output(0)->begin());
    assert(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit = jlm::rvsdg::input::GetNode(**callF1->output(1)->begin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto lambdaExitMerge = jlm::rvsdg::input::GetNode(**callExitSplit->output(0)->begin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidateDeltaTest2SteensgaardAgnosticTopDown(const jlm::tests::DeltaTest2 & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::output::GetNode(*storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 =
      jlm::rvsdg::input::GetNode(**test.lambda_f1->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  auto storeD2InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[1].inner->begin());
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

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto truncNode = jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<TruncOperation>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::output::GetNode(*truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*loadG1Node->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

    jlm::rvsdg::Node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::output::GetNode(*input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jlm::rvsdg::output::GetNode(*storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::output::GetNode(*loadG2Node->input(1)->origin());
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

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto truncNode = jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<TruncOperation>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::output::GetNode(*truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*loadG1Node->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    jlm::rvsdg::Node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::output::GetNode(*input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jlm::rvsdg::output::GetNode(*storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::output::GetNode(*loadG2Node->input(1)->origin());
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

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto truncNode = jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<TruncOperation>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::output::GetNode(*truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*loadG1Node->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

    jlm::rvsdg::Node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::output::GetNode(*input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node = jlm::rvsdg::output::GetNode(*storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::output::GetNode(*loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

static void
ValidateImportTestSteensgaardAgnostic(const jlm::tests::ImportTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::output::GetNode(*storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 =
      jlm::rvsdg::input::GetNode(**test.lambda_f1->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[1].inner->begin());
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

    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.lambda_f1->GetFunctionResults()[1]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto storeNode = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*storeNode->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->nnodes() == 9);

    auto lambdaEntrySplit =
        jlm::rvsdg::input::GetNode(**test.lambda_f2->GetFunctionArguments()[1]->begin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[0].inner->begin());
    assert(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    assert(jlm::rvsdg::output::GetNode(*storeD1->input(2)->origin()) == lambdaEntrySplit);

    auto storeD2 = jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[1].inner->begin());
    assert(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    assert(jlm::rvsdg::output::GetNode(*storeD2->input(2)->origin()) == lambdaEntrySplit);

    auto callEntryMerge = jlm::rvsdg::input::GetNode(**storeD1->output(0)->begin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callF1 = jlm::rvsdg::input::GetNode(**callEntryMerge->output(0)->begin());
    assert(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit = jlm::rvsdg::input::GetNode(**callF1->output(1)->begin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto lambdaExitMerge = jlm::rvsdg::input::GetNode(**callExitSplit->output(0)->begin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidateImportTestSteensgaardAgnosticTopDown(const jlm::tests::ImportTest & test)
{
  using namespace jlm::llvm;

  assert(test.lambda_f2->subregion()->nnodes() == 9);

  auto lambdaEntrySplit =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetFunctionArguments()[1]->begin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(jlm::rvsdg::output::GetNode(*storeD1InF2->input(2)->origin()) == lambdaEntrySplit);

  assert(storeD1InF2->output(0)->nusers() == 1);
  auto d1StateIndexEntry = (*storeD1InF2->output(0)->begin())->index();

  auto storeD1InF1 =
      jlm::rvsdg::input::GetNode(**test.lambda_f1->GetContextVars()[0].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));
  assert(d1StateIndexEntry == storeD1InF1->input(2)->origin()->index());
  assert(storeD1InF1->output(0)->nusers() == 1);
  auto d1StateIndexExit = (*storeD1InF1->output(0)->begin())->index();

  auto storeD2InF2 =
      jlm::rvsdg::input::GetNode(**test.lambda_f2->GetContextVars()[1].inner->begin());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndexExit != storeD2InF2->input(2)->origin()->index());
}

static void
ValidatePhiTestSteensgaardAgnostic(const jlm::tests::PhiTest1 & test)
{
  using namespace jlm::llvm;

  auto arrayStateIndex = (*test.alloca->output(1)->begin())->index();

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda_fib->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 4, 1));

  auto store = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(arrayStateIndex)->origin());
  assert(is<StoreNonVolatileOperation>(*store, 3, 1));

  auto gamma = jlm::rvsdg::output::GetNode(*store->input(2)->origin());
  assert(gamma == test.gamma);

  auto gammaStateIndex = store->input(2)->origin()->index();

  auto load1 = jlm::rvsdg::output::GetNode(
      *test.gamma->GetExitVars()[gammaStateIndex].branchResult[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::output::GetNode(*load1->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*load2, 2, 2));

  assert(load2->input(1)->origin()->index() == arrayStateIndex);
}

static void
ValidatePhiTestSteensgaardRegionAware(const jlm::tests::PhiTest1 & test)
{
  using namespace jlm::llvm;

  auto arrayStateIndex = (*test.alloca->output(1)->begin())->index();

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda_fib->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto store = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(arrayStateIndex)->origin());
  assert(is<StoreNonVolatileOperation>(*store, 3, 1));

  auto gamma = jlm::rvsdg::output::GetNode(*store->input(2)->origin());
  assert(gamma == test.gamma);

  auto gammaStateIndex = store->input(2)->origin()->index();

  auto load1 = jlm::rvsdg::output::GetNode(
      *test.gamma->GetExitVars()[gammaStateIndex].branchResult[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::output::GetNode(*load1->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*load2, 2, 2));

  assert(load2->input(1)->origin()->index() == arrayStateIndex);
}

static void
ValidatePhiTestSteensgaardAgnosticTopDown(const jlm::tests::PhiTest1 & test)
{
  using namespace jlm::llvm;

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*test.lambda_fib->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 4, 1));

  const StoreNonVolatileNode * storeNode = nullptr;
  const jlm::rvsdg::GammaNode * gammaNode = nullptr;
  for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
  {
    auto node = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(n)->origin());
    if (auto castedStoreNode = dynamic_cast<const StoreNonVolatileNode *>(node))
    {
      storeNode = castedStoreNode;
    }
    else if (auto castedGammaNode = dynamic_cast<const jlm::rvsdg::GammaNode *>(node))
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

  auto load1 = jlm::rvsdg::output::GetNode(
      *test.gamma->GetExitVars()[gammaStateIndex].branchResult[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::output::GetNode(*load1->input(1)->origin());
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
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto load = jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<LoadNonVolatileOperation>(*load, 3, 3));

    auto store = jlm::rvsdg::output::GetNode(*load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 4, 2));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*store->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }

  /*
   * Validate function g
   */
  {
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaG().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto callExitSplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 5));

    auto call = jlm::rvsdg::output::GetNode(*callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 5, 1));

    jlm::rvsdg::Node * memcpy = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::output::GetNode(*callEntryMerge->input(n)->origin());
      if (is<MemCpyNonVolatileOperation>(node))
        memcpy = node;
    }
    assert(memcpy != nullptr);
    assert(is<MemCpyNonVolatileOperation>(*memcpy, 7, 4));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*memcpy->input(5)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
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
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto load = jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<LoadNonVolatileOperation>(*load, 3, 3));

    auto store = jlm::rvsdg::output::GetNode(*load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 4, 2));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*store->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
  }

  /*
   * Validate function g
   */
  {
    auto callNode = jlm::rvsdg::input::GetNode(**test.LambdaG().GetContextVars()[2].inner->begin());
    assert(is<CallOperation>(*callNode, 3, 3));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*callNode->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 2, 1));

    auto callExitSplit = jlm::rvsdg::input::GetNode(**callNode->output(2)->begin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 2));

    auto memcpyNode = jlm::rvsdg::output::GetNode(*callEntryMerge->input(0)->origin());
    assert(is<MemCpyNonVolatileOperation>(*memcpyNode, 7, 4));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*memcpyNode->input(4)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
    assert(jlm::rvsdg::output::GetNode(*memcpyNode->input(5)->origin()) == lambdaEntrySplit);

    auto lambdaExitMerge = jlm::rvsdg::input::GetNode(**callExitSplit->output(0)->begin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

static void
ValidateMemcpyTestSteensgaardAgnosticTopDown(const jlm::tests::MemcpyTest & test)
{
  using namespace jlm::llvm;

  // Validate function f
  {
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto load = jlm::rvsdg::output::GetNode(*test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<LoadNonVolatileOperation>(*load, 3, 3));

    auto store = jlm::rvsdg::output::GetNode(*load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 4, 2));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*store->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }

  // Validate function g
  {
    auto lambdaExitMerge =
        jlm::rvsdg::output::GetNode(*test.LambdaG().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto callExitSplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 5));

    auto call = jlm::rvsdg::output::GetNode(*callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::output::GetNode(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 5, 1));

    jlm::rvsdg::Node * memcpy = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node = jlm::rvsdg::output::GetNode(*callEntryMerge->input(n)->origin());
      if (is<MemCpyNonVolatileOperation>(node))
        memcpy = node;
    }
    assert(memcpy != nullptr);
    assert(is<MemCpyNonVolatileOperation>(*memcpy, 7, 4));

    auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*memcpy->input(5)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }
}

static void
ValidateFreeNullTestSteensgaardAgnostic(const jlm::tests::FreeNullTest & test)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto lambdaExitMerge =
      jlm::rvsdg::output::GetNode(*GetMemoryStateRegionResult(test.LambdaMain()).origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto free = jlm::rvsdg::output::GetNode(*test.LambdaMain().GetFunctionResults()[0]->origin());
  assert(is<FreeOperation>(*free, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::output::GetNode(*lambdaExitMerge->input(0)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

static int
TestMemoryStateEncoder()
{
  using namespace jlm::llvm::aa;

  ValidateTest<jlm::tests::StoreTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidateStoreTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::StoreTest1, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateStoreTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::StoreTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateStoreTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::StoreTest2, Steensgaard, AgnosticModRefSummarizer>(
      ValidateStoreTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::StoreTest2, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateStoreTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::StoreTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateStoreTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::LoadTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidateLoadTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::LoadTest1, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateLoadTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::LoadTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateLoadTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::LoadTest2, Steensgaard, AgnosticModRefSummarizer>(
      ValidateLoadTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::LoadTest2, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateLoadTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::LoadTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateLoadTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::LoadFromUndefTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateLoadFromUndefSteensgaardAgnostic);
  ValidateTest<jlm::tests::LoadFromUndefTest, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateLoadFromUndefSteensgaardRegionAware);
  ValidateTest<jlm::tests::LoadFromUndefTest, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateLoadFromUndefSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::CallTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidateCallTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::CallTest1, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateCallTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::CallTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateCallTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::CallTest2, Steensgaard, AgnosticModRefSummarizer>(
      ValidateCallTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::CallTest2, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateCallTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::CallTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateCallTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::IndirectCallTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidateIndirectCallTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::IndirectCallTest1, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateIndirectCallTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::IndirectCallTest1, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateIndirectCallTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::IndirectCallTest2, Steensgaard, AgnosticModRefSummarizer>(
      ValidateIndirectCallTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::IndirectCallTest2, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateIndirectCallTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::IndirectCallTest2, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateIndirectCallTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::GammaTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateGammaTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::GammaTest, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateGammaTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::GammaTest, Steensgaard, AgnosticTopDownMemoryNodeProvider>(
      ValidateGammaTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::ThetaTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateThetaTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::ThetaTest, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateThetaTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::ThetaTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateThetaTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::DeltaTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidateDeltaTest1SteensgaardAgnostic);
  ValidateTest<jlm::tests::DeltaTest1, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateDeltaTest1SteensgaardRegionAware);
  ValidateTest<jlm::tests::DeltaTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidateDeltaTest1SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::DeltaTest2, Steensgaard, AgnosticModRefSummarizer>(
      ValidateDeltaTest2SteensgaardAgnostic);
  ValidateTest<jlm::tests::DeltaTest2, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateDeltaTest2SteensgaardRegionAware);
  ValidateTest<jlm::tests::DeltaTest2, Steensgaard, AgnosticModRefSummarizer>(
      ValidateDeltaTest2SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::DeltaTest3, Steensgaard, AgnosticModRefSummarizer>(
      ValidateDeltaTest3SteensgaardAgnostic);
  ValidateTest<jlm::tests::DeltaTest3, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateDeltaTest3SteensgaardRegionAware);
  ValidateTest<jlm::tests::DeltaTest3, Steensgaard, AgnosticModRefSummarizer>(
      ValidateDeltaTest3SteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::ImportTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateImportTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::ImportTest, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateImportTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::ImportTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateImportTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::PhiTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidatePhiTestSteensgaardAgnostic);
  ValidateTest<jlm::tests::PhiTest1, Steensgaard, RegionAwareModRefSummarizer>(
      ValidatePhiTestSteensgaardRegionAware);
  ValidateTest<jlm::tests::PhiTest1, Steensgaard, AgnosticModRefSummarizer>(
      ValidatePhiTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::MemcpyTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateMemcpySteensgaardAgnostic);
  ValidateTest<jlm::tests::MemcpyTest, Steensgaard, RegionAwareModRefSummarizer>(
      ValidateMemcpySteensgaardRegionAware);
  ValidateTest<jlm::tests::MemcpyTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateMemcpyTestSteensgaardAgnosticTopDown);

  ValidateTest<jlm::tests::FreeNullTest, Steensgaard, AgnosticModRefSummarizer>(
      ValidateFreeNullTestSteensgaardAgnostic);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder", TestMemoryStateEncoder)
