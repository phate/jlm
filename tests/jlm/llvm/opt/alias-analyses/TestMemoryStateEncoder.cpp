/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
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

TEST(MemoryStateEncoderTests, storeTest1AndersenAgnostic)
{
  using namespace jlm::llvm;

  StoreTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 14u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 6, 1));

  // Agnostic ModRef summaries lead to Join operations for all allocas
  auto [aJoinNode, aJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_a->output(1)->SingleUser());
  auto [bJoinNode, bJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_b->output(1)->SingleUser());
  auto [cJoinNode, cJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_c->output(1)->SingleUser());
  auto [dJoinNode, dJoinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca_d->output(1)->SingleUser());
  EXPECT_TRUE(aJoinOp && aJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(bJoinOp && bJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(cJoinOp && cJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(dJoinOp && dJoinNode->output(0)->nusers() == 1);

  // the d alloca is not used by any operation, and goes straight to the call exit
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(dJoinNode->output(0)->SingleUser()),
      lambdaExitMerge);

  auto storeD = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(cJoinNode->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD, 3, 1));
  EXPECT_EQ(storeD->input(0)->origin(), test.alloca_c->output(0));
  EXPECT_EQ(storeD->input(1)->origin(), test.alloca_d->output(0));

  auto storeC = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(bJoinNode->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeC, 3, 1));
  EXPECT_EQ(storeC->input(0)->origin(), test.alloca_b->output(0));
  EXPECT_EQ(storeC->input(1)->origin(), test.alloca_c->output(0));

  auto storeB = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(aJoinNode->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  EXPECT_EQ(storeB->input(0)->origin(), test.alloca_a->output(0));
  EXPECT_EQ(storeB->input(1)->origin(), test.alloca_b->output(0));
}

TEST(MemoryStateEncoderTests, storeTest1AndersenRegionAware)
{
  using namespace jlm::llvm;

  StoreTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 1u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
}

TEST(MemoryStateEncoderTests, storeTest2AndersenAgnostic)
{
  using namespace jlm::llvm;

  StoreTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 17u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));

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
  EXPECT_TRUE(aJoinOp && aJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(bJoinOp && bJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(xJoinOp && xJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(yJoinOp && yJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(pJoinOp && pJoinNode->output(0)->nusers() == 1);

  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(aJoinNode->output(0)->SingleUser()),
      lambdaExitMerge);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(bJoinNode->output(0)->SingleUser()),
      lambdaExitMerge);

  auto storeA =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_a->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeA, 3, 1));
  EXPECT_EQ(storeA->input(0)->origin(), test.alloca_x->output(0));
  EXPECT_EQ(storeA->input(1)->origin(), test.alloca_a->output(0));
  EXPECT_TRUE(
      jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeA->input(2)->origin()));

  auto storeB =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_b->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  EXPECT_EQ(storeB->input(0)->origin(), test.alloca_y->output(0));
  EXPECT_EQ(storeB->input(1)->origin(), test.alloca_b->output(0));
  EXPECT_TRUE(
      jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeB->input(2)->origin()));

  auto storeX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(pJoinNode->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  EXPECT_EQ(storeX->input(0)->origin(), test.alloca_p->output(0));
  EXPECT_EQ(storeX->input(1)->origin(), test.alloca_x->output(0));
  EXPECT_TRUE(
      jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeX->input(2)->origin()));

  auto storeY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeX->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  EXPECT_EQ(storeY->input(0)->origin(), test.alloca_p->output(0));
  EXPECT_EQ(storeY->input(1)->origin(), test.alloca_y->output(0));
  EXPECT_EQ(storeY->input(2)->origin(), storeX->output(0));
}

TEST(MemoryStateEncoderTests, storeTest2AndersenRegionAware)
{
  using namespace jlm::llvm;

  StoreTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 1u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
}

TEST(MemoryStateEncoderTests, loadTest1AndersenAgnostic)
{
  using namespace jlm::llvm;

  LoadTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 4u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda->GetFunctionArguments()[1]->SingleUser());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

  auto loadA = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(0)->origin());

  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadA, 3, 3));
  EXPECT_EQ(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(1)->origin()), loadX);

  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadX, 3, 3));
  EXPECT_EQ(loadX->input(0)->origin(), test.lambda->GetFunctionArguments()[0]);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin()),
      lambdaEntrySplit);
}

TEST(MemoryStateEncoderTests, loadTest1AndersenRegionAware)
{
  using namespace jlm::llvm;

  LoadTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 4u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda->GetFunctionArguments()[1]->SingleUser());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

  auto loadA = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  auto loadX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(0)->origin());

  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadA, 2, 2));
  EXPECT_EQ(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadA->input(1)->origin()), loadX);

  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadX, 2, 2));
  EXPECT_EQ(loadX->input(0)->origin(), test.lambda->GetFunctionArguments()[0]);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin()),
      lambdaEntrySplit);
}

TEST(MemoryStateEncoderTests, loadTest2AndersenAgnostic)
{
  using namespace jlm::llvm;
  LoadTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 19u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));

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
  EXPECT_TRUE(aJoinOp && aJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(bJoinOp && bJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(xJoinOp && xJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(yJoinOp && yJoinNode->output(0)->nusers() == 1);
  EXPECT_TRUE(pJoinOp && pJoinNode->output(0)->nusers() == 1);

  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(aJoinNode->output(0)->SingleUser()),
      lambdaExitMerge);
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(bJoinNode->output(0)->SingleUser()),
      lambdaExitMerge);

  auto storeA =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_a->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeA, 3, 1));
  EXPECT_EQ(storeA->input(0)->origin(), test.alloca_x->output(0));
  EXPECT_TRUE(
      jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeA->input(2)->origin()));

  auto storeB =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.alloca_b->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeB, 3, 1));
  EXPECT_EQ(storeB->input(0)->origin(), test.alloca_y->output(0));
  EXPECT_TRUE(
      jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeB->input(2)->origin()));

  auto storeX = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(pJoinNode->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeX, 3, 1));
  EXPECT_EQ(storeX->input(0)->origin(), test.alloca_p->output(0));
  EXPECT_EQ(storeX->input(1)->origin(), test.alloca_x->output(0));
  EXPECT_TRUE(
      jlm::rvsdg::IsOwnerNodeOperation<MemoryStateJoinOperation>(*storeX->input(2)->origin()));

  auto load1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeX->output(0)->SingleUser());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*load1, 2, 2));
  EXPECT_EQ(load1->input(0)->origin(), test.alloca_p->output(0));
  EXPECT_EQ(load1->input(1)->origin(), storeX->output(0));

  auto load2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(load1->output(0)->SingleUser());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*load2, 2, 2));
  EXPECT_EQ(load2->input(1)->origin(), storeA->output(0));

  auto storeY = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(load2->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeY, 3, 1));
  EXPECT_EQ(storeY->input(0)->origin(), test.alloca_y->output(0));
  EXPECT_EQ(storeY->input(2)->origin(), storeB->output(0));
}

TEST(MemoryStateEncoderTests, loadTest2AndersenRegionAware)
{
  using namespace jlm::llvm;

  LoadTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 1u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
}

TEST(MemoryStateEncoderTests, loadFromUndefAndersenAgnostic)
{
  using namespace jlm::llvm;

  LoadFromUndefTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());
  EXPECT_EQ(test.Lambda().subregion()->numNodes(), 4u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*load, 1, 1));

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.Lambda().GetFunctionArguments()[0]->SingleUser());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

TEST(MemoryStateEncoderTests, loadFromUndefAndersenRegionAware)
{
  using namespace jlm::llvm;

  LoadFromUndefTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  EXPECT_EQ(test.Lambda().subregion()->numNodes(), 3u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

  auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.Lambda().GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*load, 1, 1));
}

TEST(MemoryStateEncoderTests, callTest1AndersenAgnostic)
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

    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin()),
        lambdaEntrySplit);

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin()),
        lambdaEntrySplit);
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

    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 7, 1));
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 7));

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin()),
        lambdaEntrySplit);

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    EXPECT_TRUE(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin()) == loadX);
  }

  /* validate h */
  {
    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallF().input(4)->origin());
    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.CallF().output(2)->SingleUser());

    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));

    callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallG().input(4)->origin());
    callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.CallG().output(2)->SingleUser());

    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 7, 1));
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 7));
  }
}

TEST(MemoryStateEncoderTests, callTest1AndersenRegionAware)
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

    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin()),
        lambdaEntrySplit);

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin()),
        lambdaEntrySplit);
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

    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadX, 2, 2));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadX->input(1)->origin()),
        lambdaEntrySplit);

    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadY, 2, 2));
    EXPECT_EQ(jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadY->input(1)->origin()), loadX);
  }

  /* validate h */
  {
    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallF().input(4)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 2, 1));
    // There is no call exit split, as it has been removed by dead node elimination
    EXPECT_EQ(test.CallF().output(2)->nusers(), 0u);

    callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.CallG().input(4)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));
    EXPECT_EQ(test.CallG().output(2)->nusers(), 0u);
  }
}

TEST(MemoryStateEncoderTests, callTest2AndersenAgnostic)
{
  using namespace jlm::llvm;
  CallTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /* validate create function */
  {
    EXPECT_EQ(test.lambda_create->subregion()->numNodes(), 7u);

    auto stateJoin = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        MallocOperation::memoryStateOutput(*test.malloc).SingleUser());
    EXPECT_TRUE(is<MemoryStateJoinOperation>(*stateJoin, 2, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*stateJoin->input(1)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(stateJoin->output(0)->SingleUser());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto mallocStateLambdaEntryIndex = stateJoin->input(1)->origin()->index();
    auto mallocStateLambdaExitIndex = stateJoin->output(0)->SingleUser().index();
    EXPECT_EQ(mallocStateLambdaEntryIndex, mallocStateLambdaExitIndex);
  }

  /* validate destroy function */
  {
    EXPECT_EQ(test.lambda_destroy->subregion()->numNodes(), 4u);
  }

  /* validate test function */
  {
    EXPECT_EQ(test.lambda_test->subregion()->numNodes(), 16u);
  }
}

TEST(MemoryStateEncoderTests, callTest2AndersenRegionAware)
{
  using namespace jlm::llvm;

  CallTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* validate create function */
  {
    EXPECT_EQ(test.lambda_create->subregion()->numNodes(), 7u);

    auto stateJoin = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        MallocOperation::memoryStateOutput(*test.malloc).SingleUser());
    EXPECT_TRUE(is<MemoryStateJoinOperation>(*stateJoin, 2, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*stateJoin->input(1)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(stateJoin->output(0)->SingleUser());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto mallocStateLambdaEntryIndex = stateJoin->input(1)->origin()->index();
    auto mallocStateLambdaExitIndex = stateJoin->output(0)->SingleUser().index();
    EXPECT_EQ(mallocStateLambdaEntryIndex, mallocStateLambdaExitIndex);
  }

  /* validate destroy function */
  {
    EXPECT_EQ(test.lambda_destroy->subregion()->numNodes(), 4u);
  }

  /* validate test function */
  {
    EXPECT_EQ(test.lambda_test->subregion()->numNodes(), 16u);
  }
}

TEST(MemoryStateEncoderTests, indirectCallTest1AndersenAgnostic)
{
  using namespace jlm::llvm;

  IndirectCallTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /* validate indcall function */
  {
    EXPECT_EQ(test.GetLambdaIndcall().subregion()->numNodes(), 6u);

    auto lambda_exit_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaIndcall().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambda_exit_mux->input(0)->origin());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_exit_mux->input(0)->origin());
    EXPECT_TRUE(is<CallOperation>(*call, 3, 3));

    auto call_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(2)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_entry_mux->input(2)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
  }

  /* validate test function */
  {
    EXPECT_EQ(test.GetLambdaTest().subregion()->numNodes(), 9u);

    auto lambda_exit_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 5, 1));

    auto call_exit_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambda_exit_mux->input(0)->origin());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_exit_mux->input(0)->origin());
    EXPECT_TRUE(is<CallOperation>(*call, 4, 3));

    auto call_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    call_exit_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_entry_mux->input(0)->origin());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*call_exit_mux, 1, 5));

    call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_exit_mux->input(0)->origin());
    EXPECT_TRUE(is<CallOperation>(*call, 4, 3));

    call_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*call_entry_mux, 5, 1));

    auto lambda_entry_mux =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call_entry_mux->input(2)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 5));
  }
}

TEST(MemoryStateEncoderTests, indirectCallTest1AndersenRegionAware)
{
  using namespace jlm::llvm;

  IndirectCallTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* validate indcall function */
  {
    EXPECT_EQ(test.GetLambdaIndcall().subregion()->numNodes(), 4u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaIndcall().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaIndcall().GetFunctionResults()[0]->origin());
    EXPECT_TRUE(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(2)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  /* validate test function */
  {
    EXPECT_EQ(test.GetLambdaTest().subregion()->numNodes(), 6u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto add = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[0]->origin());
    EXPECT_TRUE(is<jlm::rvsdg::BinaryOperation>(*add, 2, 1));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*add->input(0)->origin());
    EXPECT_TRUE(is<CallOperation>(*call, 4, 3));

    auto callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));

    call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*add->input(1)->origin());
    EXPECT_TRUE(is<CallOperation>(*call, 4, 3));

    callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(3)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }
}

TEST(MemoryStateEncoderTests, indirectCallTest2AndersenAgnostic)
{
  using namespace jlm::llvm;

  IndirectCallTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  // validate function three()
  {
    EXPECT_EQ(test.GetLambdaThree().subregion()->numNodes(), 3u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaThree().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaThree().GetFunctionArguments()[1]->SingleUser());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function four()
  {
    EXPECT_EQ(test.GetLambdaFour().subregion()->numNodes(), 3u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaFour().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaFour().GetFunctionArguments()[1]->SingleUser());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }

  // validate function i()
  {
    EXPECT_EQ(test.GetLambdaI().subregion()->numNodes(), 6u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaI().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 13, 1));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 13));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetIndirectCall().input(2)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 13, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callEntryMerge->input(0)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 13));
  }
}

TEST(MemoryStateEncoderTests, indirectCallTest2AndersenRegionAware)
{
  using namespace jlm::llvm;

  IndirectCallTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  // validate function three()
  {
    EXPECT_EQ(test.GetLambdaThree().subregion()->numNodes(), 2u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaThree().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
  }

  // validate function four()
  {
    EXPECT_EQ(test.GetLambdaFour().subregion()->numNodes(), 2u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaFour().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));
  }

  // validate function i()
  {
    EXPECT_EQ(test.GetLambdaI().subregion()->numNodes(), 4u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaI().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetIndirectCall().input(2)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function x()
  {
    EXPECT_EQ(test.GetLambdaX().subregion()->numNodes(), 7u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaX().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetCallIWithThree().input(3)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function y()
  {
    EXPECT_EQ(test.GetLambdaY().subregion()->numNodes(), 7u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaY().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*test.GetCallIWithFour().input(3)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 0, 1));
  }

  // validate function test()
  {
    EXPECT_EQ(test.GetLambdaTest().subregion()->numNodes(), 14u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto loadG1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaTest().GetContextVars()[2].inner->SingleUser());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadG1, 2, 2));

    auto loadG2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaTest().GetContextVars()[3].inner->SingleUser());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadG2, 2, 2));

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.GetLambdaTest().GetFunctionArguments()[1]->SingleUser());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
  }

  // validate function test2()
  {
    EXPECT_EQ(test.GetLambdaTest2().subregion()->numNodes(), 5u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.GetLambdaTest2().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 0, 1));

    // The entry memory state is unused
    EXPECT_EQ(test.GetLambdaTest2().GetFunctionArguments()[1]->nusers(), 0u);
  }
}

TEST(MemoryStateEncoderTests, gammaTestAndersenAgnostic)
{
  using namespace jlm::llvm;

  GammaTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto loadTmp2 =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadTmp2, 3, 3));

  auto loadTmp1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp2->input(1)->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadTmp1, 3, 3));

  auto gamma = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp1->input(1)->origin());
  EXPECT_EQ(gamma, test.gamma);
}

TEST(MemoryStateEncoderTests, gammaTestAndersenRegionAware)
{
  using namespace jlm::llvm;

  GammaTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto loadTmp2 =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadTmp2, 2, 2));

  auto loadTmp1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp2->input(1)->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadTmp1, 2, 2));

  auto lambdaEntrySplit =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadTmp1->input(1)->origin());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
}

TEST(MemoryStateEncoderTests, thetaTestAndersenAgnostic)
{
  using namespace jlm::llvm;

  ThetaTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 4u);

  auto lambda_exit_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambda_exit_mux, 2, 1));

  auto thetaOutput = lambda_exit_mux->input(0)->origin();
  auto theta = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*thetaOutput);
  EXPECT_EQ(theta, test.theta);

  auto loopvar = theta->MapOutputLoopVar(*thetaOutput);
  auto storeStateOutput = loopvar.post->origin();
  auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeStateOutput);
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*store, 4, 2));
  EXPECT_EQ(store->input(storeStateOutput->index() + 2)->origin(), loopvar.pre);

  auto lambda_entry_mux = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loopvar.input->origin());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambda_entry_mux, 1, 2));
}

TEST(MemoryStateEncoderTests, thetaTestAndersenRegionAware)
{
  using namespace jlm::llvm;

  ThetaTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda->subregion()->numNodes(), 4u);

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda->GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto thetaOutput = lambdaExitMerge->input(0)->origin();
  auto theta = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::ThetaNode>(*thetaOutput);
  EXPECT_EQ(theta, test.theta);
  auto loopvar = theta->MapOutputLoopVar(*thetaOutput);

  auto storeStateOutput = loopvar.post->origin();
  auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeStateOutput);
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*store, 3, 1));
  EXPECT_EQ(store->input(storeStateOutput->index() + 2)->origin(), loopvar.pre);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loopvar.input->origin());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
}

TEST(MemoryStateEncoderTests, deltaTest1AndersenAgnostic)
{
  using namespace jlm::llvm;

  DeltaTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda_h->subregion()->numNodes(), 7u);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_h->GetFunctionArguments()[1]->SingleUser());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 4));

  auto storeF =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.constantFive->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeF->input(2)->origin()),
      lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_g->GetFunctionArguments()[0]->SingleUser());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  EXPECT_EQ(loadF->input(1)->origin()->index(), deltaStateIndex);
}

TEST(MemoryStateEncoderTests, deltaTest1AndersenRegionAware)
{
  using namespace jlm::llvm;

  DeltaTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda_h->subregion()->numNodes(), 7u);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_h->GetFunctionArguments()[1]->SingleUser());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

  auto storeF =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(test.constantFive->output(0)->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeF, 3, 1));
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeF->input(2)->origin()),
      lambdaEntrySplit);

  auto deltaStateIndex = storeF->input(2)->origin()->index();

  auto loadF = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_g->GetFunctionArguments()[0]->SingleUser());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadF, 2, 2));
  EXPECT_EQ(loadF->input(1)->origin()->index(), deltaStateIndex);
}

TEST(MemoryStateEncoderTests, deltaTest2AndersenAgnostic)
{
  using namespace jlm::llvm;

  DeltaTest2 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda_f2->subregion()->numNodes(), 9u);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[0].inner->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1InF2->input(2)->origin()),
      lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f1->GetContextVars()[0].inner->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  EXPECT_EQ(d1StateIndex, storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[1].inner->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  EXPECT_NE(d1StateIndex, storeD2InF2->input(2)->origin()->index());
}

TEST(MemoryStateEncoderTests, deltaTest2AndersenRegionAware)
{
  using namespace jlm::llvm;

  DeltaTest2 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* Validate f1() */
  {
    EXPECT_EQ(test.lambda_f1->subregion()->numNodes(), 4u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f1->GetFunctionResults()[1]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto storeNode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeNode->input(2)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    EXPECT_EQ(test.lambda_f2->subregion()->numNodes(), 9u);

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[0].inner->SingleUser());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1->input(2)->origin()),
        lambdaEntrySplit);

    auto storeD2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[1].inner->SingleUser());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD2->input(2)->origin()),
        lambdaEntrySplit);

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeD1->output(0)->SingleUser());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callF1 =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callEntryMerge->output(0)->SingleUser());
    EXPECT_TRUE(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callF1->output(1)->SingleUser());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callExitSplit->output(0)->SingleUser());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

TEST(MemoryStateEncoderTests, deltaTest3AndersenAgnostic)
{
  using namespace jlm::llvm;

  DeltaTest3 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  /* validate f() */
  {
    EXPECT_EQ(test.LambdaF().subregion()->numNodes(), 6u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto truncNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    EXPECT_TRUE(is<TruncOperation>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*truncNode->input(0)->origin());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG1Node->input(1)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

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
    EXPECT_NE(storeG2Node, nullptr);

    auto loadG2Node =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeG2Node->input(2)->origin());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG2Node->input(1)->origin());
    EXPECT_EQ(node, lambdaEntrySplit);
  }
}

TEST(MemoryStateEncoderTests, deltaTest3AndersenRegionAware)
{
  using namespace jlm::llvm;

  DeltaTest3 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* validate f() */
  {
    EXPECT_EQ(test.LambdaF().subregion()->numNodes(), 6u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

    auto truncNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    EXPECT_TRUE(is<TruncOperation>(*truncNode, 1, 1));

    auto loadG1Node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*truncNode->input(0)->origin());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadG1Node, 2, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG1Node->input(1)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

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
    EXPECT_NE(storeG2Node, nullptr);

    auto loadG2Node =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeG2Node->input(2)->origin());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadG2Node, 2, 2));

    auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*loadG2Node->input(1)->origin());
    EXPECT_EQ(node, lambdaEntrySplit);
  }
}

TEST(MemoryStateEncoderTests, importTestAndersenAgnostic)
{
  using namespace jlm::llvm;

  ImportTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  EXPECT_EQ(test.lambda_f2->subregion()->numNodes(), 9u);

  auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));

  auto storeD1InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[0].inner->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));
  EXPECT_EQ(
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1InF2->input(2)->origin()),
      lambdaEntrySplit);

  auto d1StateIndex = storeD1InF2->input(2)->origin()->index();

  auto storeD1InF1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f1->GetContextVars()[0].inner->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1InF1, 3, 1));

  EXPECT_EQ(d1StateIndex, storeD1InF1->input(2)->origin()->index());

  auto storeD2InF2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      test.lambda_f2->GetContextVars()[1].inner->SingleUser());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1InF2, 3, 1));

  EXPECT_NE(d1StateIndex, storeD2InF2->input(2)->origin()->index());
}

TEST(MemoryStateEncoderTests, importTestAndersenRegionAware)
{
  using namespace jlm::llvm;

  ImportTest test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  /* Validate f1() */
  {
    EXPECT_EQ(test.lambda_f1->subregion()->numNodes(), 4u);

    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.lambda_f1->GetFunctionResults()[1]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto storeNode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeNode, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeNode->input(2)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /* Validate f2() */
  {
    EXPECT_EQ(test.lambda_f2->subregion()->numNodes(), 9u);

    auto lambdaEntrySplit = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetFunctionArguments()[1]->SingleUser());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));

    auto storeD1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[0].inner->SingleUser());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD1, 3, 1));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD1->input(2)->origin()),
        lambdaEntrySplit);

    auto storeD2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.lambda_f2->GetContextVars()[1].inner->SingleUser());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*storeD2, 3, 1));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*storeD2->input(2)->origin()),
        lambdaEntrySplit);

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(storeD1->output(0)->SingleUser());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callF1 =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callEntryMerge->output(0)->SingleUser());
    EXPECT_TRUE(is<CallOperation>(*callF1, 3, 2));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callF1->output(1)->SingleUser());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callExitSplit->output(0)->SingleUser());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

TEST(MemoryStateEncoderTests, phiTest1AndersenAgnostic)
{
  using namespace jlm::llvm;

  PhiTest1 test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  auto [joinNode, joinOp] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca->output(1)->SingleUser());
  EXPECT_TRUE(joinNode && joinOp);
  auto arrayStateIndex = joinNode->output(0)->SingleUser().index();

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda_fib->GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 4, 1));

  auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *lambdaExitMerge->input(arrayStateIndex)->origin());
  EXPECT_TRUE(is<StoreNonVolatileOperation>(*store, 3, 1));

  auto gamma = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*store->input(2)->origin());
  EXPECT_EQ(gamma, test.gamma);

  auto gammaStateIndex = store->input(2)->origin()->index();

  auto load1 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.gamma->GetExitVars()[gammaStateIndex].branchResult[0]->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*load1, 2, 2));

  auto load2 = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*load1->input(1)->origin());
  EXPECT_TRUE(is<LoadNonVolatileOperation>(*load2, 2, 2));

  EXPECT_EQ(load2->input(1)->origin()->index(), arrayStateIndex);
}

TEST(MemoryStateEncoderTests, phiTest1AndersenRegionAware)
{
  using namespace jlm::llvm;

  PhiTest1 test;
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda_fib->GetFunctionResults()[1]->origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

  auto gamma = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.lambda_fib->GetFunctionResults()[0]->origin());
  EXPECT_EQ(gamma, test.gamma);

  // In the region aware, we know that the alloca is non-reentrant, so there is no Join
  auto [node, op] = jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(
      test.alloca->output(1)->SingleUser());
  EXPECT_EQ(op, nullptr);
}

TEST(MemoryStateEncoderTests, memCpyTestAndersenAgnostic)
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
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*load, 2, 2));

    auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*load->input(1)->origin());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*store, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*store->input(2)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }

  /*
   * Validate function g
   */
  {
    auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaG().GetFunctionResults()[2]->origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 5, 1));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 5));

    auto call = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callExitSplit->input(0)->origin());
    EXPECT_TRUE(is<CallOperation>(*call, 3, 3));

    auto callEntryMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*call->input(2)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 5, 1));

    jlm::rvsdg::Node * memcpy = nullptr;
    for (size_t n = 0; n < callEntryMerge->ninputs(); n++)
    {
      auto node =
          jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callEntryMerge->input(n)->origin());
      if (is<MemCpyNonVolatileOperation>(node))
        memcpy = node;
    }
    EXPECT_NE(memcpy, nullptr);
    EXPECT_TRUE(is<MemCpyNonVolatileOperation>(*memcpy, 5, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*memcpy->input(4)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 5));
  }
}

TEST(MemoryStateEncoderTests, memCpyAndersenRegionAware)
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
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto load = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        *test.LambdaF().GetFunctionResults()[0]->origin());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*load, 2, 2));

    auto store = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*load->input(1)->origin());
    EXPECT_TRUE(is<StoreNonVolatileOperation>(*store, 3, 1));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*store->input(2)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));
  }

  /*
   * Validate function g
   */
  {
    auto callNode = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
        test.LambdaG().GetContextVars()[2].inner->SingleUser());
    EXPECT_TRUE(is<CallOperation>(*callNode, 3, 3));

    auto callEntryMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callNode->input(2)->origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMerge, 1, 1));

    auto callExitSplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callNode->output(2)->SingleUser());
    EXPECT_TRUE(is<CallExitMemoryStateSplitOperation>(*callExitSplit, 1, 1));

    auto memcpyNode =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*callEntryMerge->input(0)->origin());
    EXPECT_TRUE(is<MemCpyNonVolatileOperation>(*memcpyNode, 5, 2));

    auto lambdaEntrySplit =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*memcpyNode->input(3)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
    EXPECT_EQ(
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*memcpyNode->input(4)->origin()),
        lambdaEntrySplit);

    auto lambdaExitMerge =
        jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(callExitSplit->output(0)->SingleUser());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));
  }
}

TEST(MemoryStateEncoderTests, freeNullTestAndersenAgnostic)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  FreeNullTest test;
  encodeStates<aa::Andersen, aa::AgnosticModRefSummarizer>(test.module());

  auto lambdaExitMerge = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *GetMemoryStateRegionResult(test.LambdaMain()).origin());
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 2, 1));

  auto free = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(
      *test.LambdaMain().GetFunctionResults()[0]->origin());
  EXPECT_TRUE(is<FreeOperation>(*free, 2, 1));

  auto lambdaEntrySplit =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::Node>(*lambdaExitMerge->input(0)->origin());
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 2));
}

TEST(MemoryStateEncoderTests, LambdaMemoryStateArgumentMultipleUsers)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto bitType32 = BitType::Create(32);
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto pointerType = PointerType::Create();
  auto functionTypeOne = FunctionType::Create(
      { ioStateType, memoryStateType },
      { bitType32, ioStateType, memoryStateType });
  auto functionTypeMain = FunctionType::Create(
      { pointerType, ioStateType, memoryStateType },
      { bitType32, ioStateType, memoryStateType });

  jlm::llvm::LlvmRvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  LambdaNode * lambdaOne = nullptr;
  {
    lambdaOne = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeOne, "one", Linkage::privateLinkage));
    auto ioStateArgument = lambdaOne->GetFunctionArguments()[0];
    auto memoryStateArgument = lambdaOne->GetFunctionArguments()[1];

    auto & one = IntegerConstantOperation::Create(*lambdaOne->subregion(), 32, 1);

    lambdaOne->finalize({ one.output(0), ioStateArgument, memoryStateArgument });
  }

  LambdaNode * lambdaMain = nullptr;
  {
    lambdaMain = LambdaNode::Create(
        rvsdg.GetRootRegion(),
        LlvmLambdaOperation::Create(functionTypeMain, "main", Linkage::externalLinkage));
    auto pointerArgument = lambdaMain->GetFunctionArguments()[0];
    auto ioStateArgument = lambdaMain->GetFunctionArguments()[1];
    auto memoryStateArgument = lambdaMain->GetFunctionArguments()[2];
    auto ctxVarOne = lambdaMain->AddContextVar(*lambdaOne->output());

    auto callResults = CallOperation::Create(
        ctxVarOne.inner,
        functionTypeOne,
        { ioStateArgument, memoryStateArgument });

    auto & loadNode = LoadNonVolatileOperation::CreateNode(
        *pointerArgument,
        { memoryStateArgument },
        bitType32,
        4);

    auto & addNode = CreateOpNode<IntegerAddOperation>({ callResults[0], loadNode.output(0) }, 32);

    lambdaMain->finalize({ addNode.output(0), callResults[1], loadNode.output(1) });
  }

  GraphExport::Create(*lambdaMain->output(), "main");

  view(rvsdg, stdout);

  // Act
  encodeStates<aa::Andersen, aa::RegionAwareModRefSummarizer>(rvsdgModule);

  view(rvsdg, stdout);

  // Assert
  {
    auto lambdaExitMerge =
        TryGetOwnerNode<SimpleNode>(*GetMemoryStateRegionResult(*lambdaMain).origin());
    EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(*lambdaExitMerge, 1, 1));

    auto loadNode = TryGetOwnerNode<SimpleNode>(*lambdaExitMerge->input(0)->origin());
    EXPECT_TRUE(is<LoadNonVolatileOperation>(*loadNode, 2, 2));

    auto lambdaEntrySplit = TryGetOwnerNode<SimpleNode>(*loadNode->input(1)->origin());
    EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(*lambdaEntrySplit, 1, 1));

    auto addNode = TryGetOwnerNode<SimpleNode>(*lambdaMain->GetFunctionResults()[0]->origin());
    EXPECT_TRUE(is<IntegerAddOperation>(*addNode, 2, 1));

    auto callNode = TryGetOwnerNode<SimpleNode>(*addNode->input(0)->origin());
    EXPECT_TRUE(is<CallOperation>(*callNode, 3, 3));
    EXPECT_TRUE(CallOperation::GetMemoryStateOutput(*callNode).IsDead());

    auto callEntryMergeNode =
        TryGetOwnerNode<SimpleNode>(*CallOperation::GetMemoryStateInput(*callNode).origin());
    EXPECT_TRUE(is<CallEntryMemoryStateMergeOperation>(*callEntryMergeNode, 0, 1));
  }
}
