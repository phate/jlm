/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/EliminatedModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/rvsdg/view.hpp>

template<class Analysis, class TModRefSummarizer>
static void
encodeStates(jlm::rvsdg::RvsdgModule & rvsdgModule)
{
  static_assert(
      std::is_base_of_v<jlm::llvm::aa::PointsToAnalysis, Analysis>,
      "Analysis should be derived from PointsToAnalysis class.");

  static_assert(
      std::is_base_of_v<jlm::llvm::aa::ModRefSummarizer, TModRefSummarizer>,
      "TModRefSummarizer should be derived from ModRefSummarizer class.");

  jlm::rvsdg::view(&rvsdgModule.Rvsdg().GetRootRegion(), stdout);

  jlm::util::StatisticsCollector statisticsCollector;

  Analysis aliasAnalysis;
  auto pointsToGraph = aliasAnalysis.Analyze(rvsdgModule, statisticsCollector);
  std::cout << jlm::llvm::aa::PointsToGraph::dumpDot(*pointsToGraph);

  TModRefSummarizer summarizer;
  auto modRefSummary =
      summarizer.SummarizeModRefs(rvsdgModule, *pointsToGraph, statisticsCollector);

  jlm::llvm::aa::MemoryStateEncoder encoder;
  std::cout << "run encoder\n";
  encoder.Encode(rvsdgModule, *modRefSummary, statisticsCollector);
  jlm::rvsdg::view(&rvsdgModule.Rvsdg().GetRootRegion(), stdout);
}

template<class OP>
static bool
is(const jlm::rvsdg::Node & node, size_t numInputs, size_t numOutputs)
{
  return jlm::rvsdg::is<OP>(&node) && node.ninputs() == numInputs && node.noutputs() == numOutputs;
}

static void
storeTest1AndersenAgnostic()
{
  using namespace jlm::llvm;

  StoreTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 14);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 6, 1));

  // Agnostic ModRef summaries lead to Join operations for all allocas
  auto [aJoinNode, aJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_a->output(1)->SingleUser());
  auto [bJoinNode, bJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_b->output(1)->SingleUser());
  auto [cJoinNode, cJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_c->output(1)->SingleUser());
  auto [dJoinNode, dJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_d->output(1)->SingleUser());
  assert(aJoinOp && aJoinNode->output(0)->nusers() == 1);
  assert(bJoinOp && bJoinNode->output(0)->nusers() == 1);
  assert(cJoinOp && cJoinNode->output(0)->nusers() == 1);
  assert(dJoinOp && dJoinNode->output(0)->nusers() == 1);

  // the d alloca is not used by any operation, and goes straight to the call exit
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(dJoinNode->output(0)->SingleUser())
      == lambdaExitMerge);

  auto storeD = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(cJoinNode->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeD, 3, 1));
  assert(storeD->input(0)->origin() == test.alloca_c->output(0));
  assert(storeD->input(1)->origin() == test.alloca_d->output(0));

  auto storeC = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(bJoinNode->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeC, 3, 1));
  assert(storeC->input(0)->origin() == test.alloca_b->output(0));
  assert(storeC->input(1)->origin() == test.alloca_c->output(0));

  auto storeB = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(aJoinNode->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_a->output(0));
  assert(storeB->input(1)->origin() == test.alloca_b->output(0));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-storeTest1AndersenAgnostic",
    storeTest1AndersenAgnostic)

static void
storeTest1AndersenRegionAware()
{
  using namespace jlm::llvm;

  StoreTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 1);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-storeTest1AndersenRegionAware",
    storeTest1AndersenRegionAware)

static void
storeTest2AndersenAgnostic()
{
  using namespace jlm::llvm;

  StoreTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 17);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));

  // Agnostic ModRef summaries lead to Join operations for all allocas
  auto [aJoinNode, aJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_a->output(1)->SingleUser());
  auto [bJoinNode, bJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_b->output(1)->SingleUser());
  auto [xJoinNode, xJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_x->output(1)->SingleUser());
  auto [yJoinNode, yJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_y->output(1)->SingleUser());
  auto [pJoinNode, pJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_p->output(1)->SingleUser());
  assert(aJoinOp && aJoinNode->output(0)->nusers() == 1);
  assert(bJoinOp && bJoinNode->output(0)->nusers() == 1);
  assert(xJoinOp && xJoinNode->output(0)->nusers() == 1);
  assert(yJoinOp && yJoinNode->output(0)->nusers() == 1);
  assert(pJoinOp && pJoinNode->output(0)->nusers() == 1);

  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(aJoinNode->output(0)->SingleUser())
      == lambdaExitMerge);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(bJoinNode->output(0)->SingleUser())
      == lambdaExitMerge);

  auto storeA =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_a->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeA, 3, 1));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));
  assert(storeA->input(1)->origin() == test.alloca_a->output(0));
  assert(jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeA->input(2)->origin()));

  auto storeB =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_b->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(storeB->input(1)->origin() == test.alloca_b->output(0));
  assert(jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeB->input(2)->origin()));

  auto storeX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(pJoinNode->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));
  assert(jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeX->input(2)->origin()));

  auto storeY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeX->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  assert(storeY->input(0)->origin() == test.alloca_p->output(0));
  assert(storeY->input(1)->origin() == test.alloca_y->output(0));
  assert(storeY->input(2)->origin() == storeX->output(0));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-storeTest2AndersenAgnostic",
    storeTest2AndersenAgnostic)

static void
storeTest2AndersenRegionAware()
{
  using namespace jlm::llvm;

  StoreTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 1);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-storeTest2AndersenRegionAware",
    storeTest2AndersenRegionAware)

static void
loadTest1AndersenAgnostic()
{
  using namespace jlm::llvm;

  LoadTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda->GetFunctionArguments()[1]->SingleUser());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  assert(loadX->input(0)->origin() == test.lambda->GetFunctionArguments()[0]);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin())
      == lambdaEntrySplit);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-loadTest1AndersenAgnostic",
    loadTest1AndersenAgnostic)

static void
loadTest1AndersenRegionAware()
{
  using namespace jlm::llvm;

  LoadTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda->GetFunctionArguments()[1]->SingleUser());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

  auto loadA = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(0)->origin());

  assert(is<LoadNonVolatileOperation>(*loadA, 2, 2));
  assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(1)->origin()) == loadX);

  assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
  assert(loadX->input(0)->origin() == test.lambda->GetFunctionArguments()[0]);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin())
      == lambdaEntrySplit);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-loadTest1AndersenRegionAware",
    loadTest1AndersenRegionAware)

static void
loadTest2AndersenAgnostic()
{
  using namespace jlm::llvm;
  LoadTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 19);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));

  // Agnostic ModRef summaries lead to Join operations for all allocas
  auto [aJoinNode, aJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_a->output(1)->SingleUser());
  auto [bJoinNode, bJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_b->output(1)->SingleUser());
  auto [xJoinNode, xJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_x->output(1)->SingleUser());
  auto [yJoinNode, yJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_y->output(1)->SingleUser());
  auto [pJoinNode, pJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_p->output(1)->SingleUser());
  assert(aJoinOp && aJoinNode->output(0)->nusers() == 1);
  assert(bJoinOp && bJoinNode->output(0)->nusers() == 1);
  assert(xJoinOp && xJoinNode->output(0)->nusers() == 1);
  assert(yJoinOp && yJoinNode->output(0)->nusers() == 1);
  assert(pJoinOp && pJoinNode->output(0)->nusers() == 1);

  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(aJoinNode->output(0)->SingleUser())
      == lambdaExitMerge);
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(bJoinNode->output(0)->SingleUser())
      == lambdaExitMerge);

  auto storeA =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_a->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeA, 3, 1));
  assert(storeA->input(0)->origin() == test.alloca_x->output(0));
  assert(jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeA->input(2)->origin()));

  auto storeB =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_b->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  assert(storeB->input(0)->origin() == test.alloca_y->output(0));
  assert(jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeB->input(2)->origin()));

  auto storeX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(pJoinNode->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  assert(storeX->input(0)->origin() == test.alloca_p->output(0));
  assert(storeX->input(1)->origin() == test.alloca_x->output(0));
  assert(jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeX->input(2)->origin()));

  auto load1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeX->output(0)->SingleUser());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));
  assert(load1->input(0)->origin() == test.alloca_p->output(0));
  assert(load1->input(1)->origin() == storeX->output(0));

  auto load2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(load1->output(0)->SingleUser());
  assert(is<LoadNonVolatileOperation>(*load2, 2, 2));
  assert(load2->input(1)->origin() == storeA->output(0));

  auto storeY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(load2->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  assert(storeY->input(0)->origin() == test.alloca_y->output(0));
  assert(storeY->input(2)->origin() == storeB->output(0));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-loadTest2AndersenAgnostic",
    loadTest2AndersenAgnostic)

static void
loadTest2AndersenRegionAware()
{
  using namespace jlm::llvm;

  LoadTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 1);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-loadTest2AndersenRegionAware",
    loadTest2AndersenRegionAware)

static void
loadFromUndefAndersenAgnostic()
{
  using namespace jlm::llvm;

  LoadFromUndefTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());
  assert(test.Lambda().subregion()->numNodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.Lambda().GetFunctionArguments()[0]->SingleUser());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-loadFromUndefAndersenAgnostic",
    loadFromUndefAndersenAgnostic)

static void
loadFromUndefAndersenRegionAware()
{
  using namespace jlm::llvm;

  LoadFromUndefTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  assert(test.Lambda().subregion()->numNodes() == 3);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

  auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load, 1, 1));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-loadFromUndefAndersenRegionAware",
    loadFromUndefAndersenRegionAware)

static void
callTest1AndersenAgnostic()
{
  using namespace jlm::llvm;

  CallTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /* validate f */
  {
    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f->GetFunctionArguments()[3]->Users().begin());
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f->GetFunctionArguments()[0]->Users().begin());
    auto loadY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f->GetFunctionArguments()[1]->Users().begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin())
        == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin())
        == lambdaEntrySplit);
  }

  /* validate g */
  {
    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_g->GetFunctionArguments()[3]->Users().begin());
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_g->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_g->GetFunctionArguments()[0]->Users().begin());
    auto loadY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_g->GetFunctionArguments()[1]->Users().begin());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin())
        == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallF().input(4)->origin());
    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.CallF().output(2)->SingleUser());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));

    callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallG().input(4)->origin());
    callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.CallG().output(2)->SingleUser());

    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-callTest1AndersenAgnostic",
    callTest1AndersenAgnostic)

static void
callTest1AndersenRegionAware()
{
  using namespace jlm::llvm;

  CallTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* validate f */
  {
    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f->GetFunctionArguments()[3]->SingleUser());
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f->GetFunctionArguments()[0]->SingleUser());
    auto loadY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f->GetFunctionArguments()[1]->SingleUser());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin())
        == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin())
        == lambdaEntrySplit);
  }

  /* validate g */
  {
    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_g->GetFunctionArguments()[3]->SingleUser());
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_g->GetFunctionResults()[2]->origin());
    auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_g->GetFunctionArguments()[0]->SingleUser());
    auto loadY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_g->GetFunctionArguments()[1]->SingleUser());

    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

    assert(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin())
        == lambdaEntrySplit);

    assert(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    assert(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallF().input(4)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 2, 1));
    // There is no call exit split, as it has been removed by dead node elimination
    assert(test.CallF().output(2)->nusers() == 0);

    callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallG().input(4)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));
    assert(test.CallG().output(2)->nusers() == 0);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-callTest1AndersenRegionAware",
    callTest1AndersenRegionAware)

static void
callTest2AndersenAgnostic()
{
  using namespace jlm::llvm;
  CallTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /* validate create function */
  {
    assert(test.lambda_create->subregion()->numNodes() == 7);

    auto stateJoin =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.malloc->output(1)->SingleUser());
    assert(is<MemoryStateJoinOperation>(*stateJoin, 2, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*stateJoin->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(stateJoin->output(0)->SingleUser());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto mallocStateLambdaEntryIndex = stateJoin->input(1)->origin()->index();
    auto mallocStateLambdaExitIndex = stateJoin->output(0)->SingleUser().index();
    assert(mallocStateLambdaEntryIndex == mallocStateLambdaExitIndex);
  }

  /* validate destroy function */
  {
    assert(test.lambda_destroy->subregion()->numNodes() == 4);
  }

  /* validate test function */
  {
    assert(test.lambda_test->subregion()->numNodes() == 16);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-callTest2AndersenAgnostic",
    callTest2AndersenAgnostic)

static void
callTest2AndersenRegionAware()
{
  using namespace jlm::llvm;

  CallTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* validate create function */
  {
    assert(test.lambda_create->subregion()->numNodes() == 7);

    auto stateJoin =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.malloc->output(1)->SingleUser());
    assert(is<MemoryStateJoinOperation>(*stateJoin, 2, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*stateJoin->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(stateJoin->output(0)->SingleUser());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto mallocStateLambdaEntryIndex = stateJoin->input(1)->origin()->index();
    auto mallocStateLambdaExitIndex = stateJoin->output(0)->SingleUser().index();
    assert(mallocStateLambdaEntryIndex == mallocStateLambdaExitIndex);
  }

  /* validate destroy function */
  {
    assert(test.lambda_destroy->subregion()->numNodes() == 4);
  }

  /* validate test function */
  {
    assert(test.lambda_test->subregion()->numNodes() == 16);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-callTest2AndersenRegionAware",
    callTest2AndersenRegionAware)

static void
indirectCallTest1AndersenAgnostic()
{
  using namespace jlm::llvm;

  IndirectCallTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /* validate indcall function */
  {
    assert(test.GetLambdaIndcall().subregion()->numNodes() == 6);

    auto lambda_exit_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaIndcall().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambda_exit_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto call_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_entry_mux->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
  }

  /* validate test function */
  {
    assert(test.GetLambdaTest().subregion()->numNodes() == 9);

    auto lambda_exit_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambda_exit_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto call_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    call_exit_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_entry_mux->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_exit_mux->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    call_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_entry_mux->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-indirectCallTest1AndersenAgnostic",
    indirectCallTest1AndersenAgnostic)

static void
indirectCallTest1AndersenRegionAware()
{
  using namespace jlm::llvm;

  IndirectCallTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* validate indcall function */
  {
    assert(test.GetLambdaIndcall().subregion()->numNodes() == 4);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaIndcall().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaIndcall().GetFunctionResults()[0]->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  /* validate test function */
  {
    assert(test.GetLambdaTest().subregion()->numNodes() == 6);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto add = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[0]->origin());
    assert(is<jlm::rvsdg::BinaryOperation>(*add, 2, 1));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*add->input(0)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    auto callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));

    call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*add->input(1)->origin());
    assert(is<CallOperation>(*call, 4, 3));

    callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-indirectCallTest1AndersenRegionAware",
    indirectCallTest1AndersenRegionAware)

static void
indirectCallTest2AndersenAgnostic()
{
  using namespace jlm::llvm;

  IndirectCallTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  // validate function three()
  {
    assert(test.GetLambdaThree().subregion()->numNodes() == 3);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaThree().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaThree().GetFunctionArguments()[1]->SingleUser());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->numNodes() == 3);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaFour().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaFour().GetFunctionArguments()[1]->SingleUser());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->numNodes() == 6);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaI().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 13));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetIndirectCall().input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 13, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callEntryMerge->input(0)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-indirectCallTest2AndersenAgnostic",
    indirectCallTest2AndersenAgnostic)

static void
indirectCallTest2AndersenRegionAware()
{
  using namespace jlm::llvm;

  IndirectCallTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  // validate function three()
  {
    assert(test.GetLambdaThree().subregion()->numNodes() == 2);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaThree().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
  }

  // validate function four()
  {
    assert(test.GetLambdaFour().subregion()->numNodes() == 2);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaFour().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
  }

  // validate function i()
  {
    assert(test.GetLambdaI().subregion()->numNodes() == 4);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaI().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetIndirectCall().input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function x()
  {
    assert(test.GetLambdaX().subregion()->numNodes() == 7);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaX().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetCallIWithThree().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function y()
  {
    assert(test.GetLambdaY().subregion()->numNodes() == 7);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaY().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetCallIWithFour().input(3)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function test()
  {
    assert(test.GetLambdaTest().subregion()->numNodes() == 14);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto loadG1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaTest().GetContextVars()[2].inner->SingleUser());
    assert(is<LoadNonVolatileOperation>(*loadG1, 2, 2));

    auto loadG2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaTest().GetContextVars()[3].inner->SingleUser());
    assert(is<LoadNonVolatileOperation>(*loadG2, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaTest().GetFunctionArguments()[1]->SingleUser());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
  }

  // validate function test2()
  {
    assert(test.GetLambdaTest2().subregion()->numNodes() == 5);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest2().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    // The entry memory state is unused
    assert(test.GetLambdaTest2().GetFunctionArguments()[1]->nusers() == 0);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-indirectCallTest2AndersenRegionAware",
    indirectCallTest2AndersenRegionAware)

static void
gammaTestAndersenAgnostic()
{
  using namespace jlm::llvm;

  GammaTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto gamma = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp1->input(1)->origin());
  assert(gamma == test.gamma);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-gammaTestAndersenAgnostic",
    gammaTestAndersenAgnostic)

static void
gammaTestAndersenRegionAware()
{
  using namespace jlm::llvm;

  GammaTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto loadTmp2 =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp2, 2, 2));

  auto loadTmp1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp2->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*loadTmp1, 2, 2));

  auto lambdaEntrySplit =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp1->input(1)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-gammaTestAndersenRegionAware",
    gammaTestAndersenRegionAware)

static void
thetaTestAndersenAgnostic()
{
  using namespace jlm::llvm;

  ThetaTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 4);

  auto lambda_exit_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 2, 1));

  auto thetaOutput = lambda_exit_mux->input(0)->origin();
  auto theta = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*thetaOutput);
  assert(theta == test.theta);

  auto loopvar = theta->MapOutputLoopVar(*thetaOutput);
  auto storeStateOutput = loopvar.post->origin();
  auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 4, 2));
  assert(store->input(storeStateOutput->index() + 2)->origin() == loopvar.pre);

  auto lambda_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loopvar.input->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-thetaTestAndersenAgnostic",
    thetaTestAndersenAgnostic)

static void
thetaTestAndersenRegionAware()
{
  using namespace jlm::llvm;

  ThetaTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  assert(test.lambda->subregion()->numNodes() == 4);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto thetaOutput = lambdaExitMerge->input(0)->origin();
  auto theta = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*thetaOutput);
  assert(theta == test.theta);
  auto loopvar = theta->MapOutputLoopVar(*thetaOutput);

  auto storeStateOutput = loopvar.post->origin();
  auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeStateOutput);
  assert(is<StoreNonVolatileOperation>(*store, 3, 1));
  assert(store->input(storeStateOutput->index() + 2)->origin() == loopvar.pre);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loopvar.input->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-thetaTestAndersenRegionAware",
    thetaTestAndersenRegionAware)

static void
deltaTest1AndersenAgnostic()
{
  using namespace jlm::llvm;

  DeltaTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda_h->subregion()->numNodes() == 7);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_h->GetFunctionArguments()[1]->SingleUser());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 4));

  auto storeF =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.constantFive->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeF->input(2)->origin())
      == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_g->GetFunctionArguments()[0]->SingleUser());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-deltaTest1AndersenAgnostic",
    deltaTest1AndersenAgnostic)

static void
deltaTest1AndersenRegionAware()
{
  using namespace jlm::llvm;

  DeltaTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  assert(test.lambda_h->subregion()->numNodes() == 7);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_h->GetFunctionArguments()[1]->SingleUser());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

  auto storeF =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.constantFive->output(0)->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeF->input(2)->origin())
      == lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_g->GetFunctionArguments()[0]->SingleUser());
  assert(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  assert(loadF->input(1)->origin()->index() == deltaStateIndex);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-deltaTest1AndersenRegionAware",
    deltaTest1AndersenRegionAware)

static void
deltaTest2AndersenAgnostic()
{
  using namespace jlm::llvm;

  DeltaTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda_f2->subregion()->numNodes() == 9);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[0].inner->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1InF2->input(2)->origin())
      == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f1->GetContextVars()[0].inner->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[1].inner->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-deltaTest2AndersenAgnostic",
    deltaTest2AndersenAgnostic)

static void
deltaTest2AndersenRegionAware()
{
  using namespace jlm::llvm;

  DeltaTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* Validate f1() */
  {
    assert(test.lambda_f1->subregion()->numNodes() == 4);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f1->GetFunctionResults()[1]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto storeNode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeNode->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->numNodes() == 9);

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[0].inner->SingleUser());
    assert(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1->input(2)->origin())
        == lambdaEntrySplit);

    auto storeD2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[1].inner->SingleUser());
    assert(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD2->input(2)->origin())
        == lambdaEntrySplit);

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeD1->output(0)->SingleUser());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callF1 =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callEntryMerge->output(0)->SingleUser());
    assert(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callF1->output(1)->SingleUser());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callExitSplit->output(0)->SingleUser());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-deltaTest2AndersenRegionAware",
    deltaTest2AndersenRegionAware)

static void
deltaTest3AndersenAgnostic()
{
  using namespace jlm::llvm;

  DeltaTest3 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /* validate f() */
  {
    assert(test.LambdaF().subregion()->numNodes() == 6);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto truncNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<TruncOperation>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG1Node->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

    jlm::rvsdg::Node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-deltaTest3AndersenAgnostic",
    deltaTest3AndersenAgnostic)

static void
deltaTest3AndersenRegionAware()
{
  using namespace jlm::llvm;

  DeltaTest3 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* validate f() */
  {
    assert(test.LambdaF().subregion()->numNodes() == 6);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto truncNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<TruncOperation>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*truncNode->input(0)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG1Node->input(1)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    jlm::rvsdg::Node * storeG2Node = nullptr;
    for (size_t n = 0; n < lambdaExitMerge->ninputs(); n++)
    {
      auto input = lambdaExitMerge->input(n);
      auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*input->origin());
      if (is<StoreNonVolatileOperation>(node))
      {
        storeG2Node = node;
        break;
      }
    }
    assert(storeG2Node != nullptr);

    auto loadG2Node =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeG2Node->input(2)->origin());
    assert(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG2Node->input(1)->origin());
    assert(node == lambdaEntrySplit);
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-deltaTest3AndersenRegionAware",
    deltaTest3AndersenRegionAware)

static void
importTestAndersenAgnostic()
{
  using namespace jlm::llvm;

  ImportTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  assert(test.lambda_f2->subregion()->numNodes() == 9);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[0].inner->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  assert(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1InF2->input(2)->origin())
      == lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f1->GetContextVars()[0].inner->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  assert(d1StateIndex == storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[1].inner->SingleUser());
  assert(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  assert(d1StateIndex != storeD2InF2->input(2)->origin()->index());
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-importTestAndersenAgnostic",
    importTestAndersenAgnostic)

static void
importTestAndersenRegionAware()
{
  using namespace jlm::llvm;

  ImportTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* Validate f1() */
  {
    assert(test.lambda_f1->subregion()->numNodes() == 4);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f1->GetFunctionResults()[1]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto storeNode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    assert(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeNode->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    assert(test.lambda_f2->subregion()->numNodes() == 9);

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[0].inner->SingleUser());
    assert(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1->input(2)->origin())
        == lambdaEntrySplit);

    auto storeD2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[1].inner->SingleUser());
    assert(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD2->input(2)->origin())
        == lambdaEntrySplit);

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeD1->output(0)->SingleUser());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callF1 =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callEntryMerge->output(0)->SingleUser());
    assert(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callF1->output(1)->SingleUser());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callExitSplit->output(0)->SingleUser());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-importTestAndersenRegionAware",
    importTestAndersenRegionAware)

static void
phiTest1AndersenAgnostic()
{
  using namespace jlm::llvm;

  PhiTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  auto [joinNode, joinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca->output(1)->SingleUser());
  assert(joinNode && joinOp);
  auto arrayStateIndex = joinNode->output(0)->SingleUser().index();

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda_fib->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 4, 1));

  auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *lambdaExitMerge->input(arrayStateIndex)->origin());
  assert(is<StoreNonVolatileOperation>(*store, 3, 1));

  auto gamma = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*store->input(2)->origin());
  assert(gamma == test.gamma);

  auto gammaStateIndex = store->input(2)->origin()->index();

  auto load1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.gamma->GetExitVars()[gammaStateIndex].branchResult[0]->origin());
  assert(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*load1->input(1)->origin());
  assert(is<LoadNonVolatileOperation>(*load2, 2, 2));

  assert(load2->input(1)->origin()->index() == arrayStateIndex);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-phiTest1AndersenAgnostic",
    phiTest1AndersenAgnostic)

static void
phiTest1AndersenRegionAware()
{
  using namespace jlm::llvm;

  PhiTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda_fib->GetFunctionResults()[1]->origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto gamma = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda_fib->GetFunctionResults()[0]->origin());
  assert(gamma == test.gamma);

  // In the region aware, we know that the alloca is non-reentrant, so there is no Join
  auto [node, op] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca->output(1)->SingleUser());
  assert(!op);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-phiTest1AndersenRegionAware",
    phiTest1AndersenRegionAware)

static void
memCpyTestAndersenAgnostic()
{
  using namespace jlm::llvm;

  MemcpyTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /*
   * Validate function f
   */
  {
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<LoadNonVolatileOperation>(*load, 2, 2));

    auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*store->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }

  /*
   * Validate function g
   */
  {
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaG().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 5));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callExitSplit->input(0)->origin());
    assert(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 5, 1));

    jlm::rvsdg::Node * memcpy = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node =
          jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callEntryMerge->input(n)->origin());
      if (is<MemCpyNonVolatileOperation>(node))
        memcpy = node;
    }
    assert(memcpy != nullptr);
    assert(is<MemCpyNonVolatileOperation>(*memcpy, 5, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*memcpy->input(4)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-memCpyTestAndersenAgnostic",
    memCpyTestAndersenAgnostic)

static void
memCpyAndersenRegionAware()
{
  using namespace jlm::llvm;

  MemcpyTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /*
   * Validate function f
   */
  {
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[2]->origin());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    assert(is<LoadNonVolatileOperation>(*load, 2, 2));

    auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*load->input(1)->origin());
    assert(is<StoreNonVolatileOperation>(*store, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*store->input(2)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /*
   * Validate function g
   */
  {
    auto callNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.LambdaG().GetContextVars()[2].inner->SingleUser());
    assert(is<CallOperation>(*callNode, 3, 3));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callNode->input(2)->origin());
    assert(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callNode->output(2)->SingleUser());
    assert(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto memcpyNode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callEntryMerge->input(0)->origin());
    assert(is<MemCpyNonVolatileOperation>(*memcpyNode, 5, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*memcpyNode->input(3)->origin());
    assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
    assert(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*memcpyNode->input(4)->origin())
        == lambdaEntrySplit);

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callExitSplit->output(0)->SingleUser());
    assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-memCpyAndersenRegionAware",
    memCpyAndersenRegionAware)

static void
freeNullTestAndersenAgnostic()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  FreeNullTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *GetMemoryStateRegionResult(test.LambdaMain()).origin());
  assert(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto free = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.LambdaMain().GetFunctionResults()[0]->origin());
  assert(is<FreeOperation>(*free, 2, 1));

  auto lambdaEntrySplit =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
  assert(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/alias-analyses/TestMemoryStateEncoder-freeNullTestAndersenAgnostic",
    freeNullTestAndersenAgnostic)
