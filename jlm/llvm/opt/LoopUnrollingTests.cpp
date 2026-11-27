/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/LoopUnrolling.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

#include <optional>
#include <unordered_map>

static jlm::util::StatisticsCollector statisticsCollector;

static size_t
nthetas(jlm::rvsdg::Region * region)
{
  size_t n = 0;
  for (const auto & node : region->Nodes())
  {
    if (dynamic_cast<const jlm::rvsdg::ThetaNode *>(&node))
      n++;
  }

  return n;
}

static jlm::rvsdg::ThetaNode *
create_theta(
    const jlm::llvm::IntegerBinaryOperation & cop,
    const jlm::llvm::IntegerBinaryOperation & aop,
    jlm::rvsdg::Output * init,
    jlm::rvsdg::Output * step,
    jlm::rvsdg::Output * end)
{
  using namespace jlm::rvsdg;

  auto graph = init->region()->graph();

  auto theta = ThetaNode::create(&graph->GetRootRegion());
  auto subregion = theta->subregion();
  auto idv = theta->AddLoopVar(init);
  auto lvs = theta->AddLoopVar(step);
  auto lve = theta->AddLoopVar(end);

  auto arm = SimpleNode::Create(*subregion, aop.copy(), { idv.pre, lvs.pre }).output(0);
  auto cmp = SimpleNode::Create(*subregion, cop.copy(), { arm, lve.pre }).output(0);
  auto & matchNode = MatchOperation::CreateNode(*cmp, { { 1, 1 } }, 0, 2);

  idv.post->divert_to(arm);
  theta->set_predicate(matchNode.output(0));

  return theta;
}

TEST(LoopUnrollingTests, test_unrollinfo)
{
  auto bt32 = jlm::rvsdg::BitType::Create(32);
  jlm::llvm::IntegerSltOperation slt(32);
  jlm::llvm::IntegerUltOperation ult(32);
  jlm::llvm::IntegerUleOperation ule(32);
  jlm::llvm::IntegerUgtOperation ugt(32);
  jlm::llvm::IntegerSgeOperation sge(32);
  jlm::llvm::IntegerEqOperation eq(32);

  jlm::llvm::IntegerAddOperation add(32);
  jlm::llvm::IntegerSubOperation sub(32);

  {
    jlm::rvsdg::Graph graph;
    auto x = &jlm::rvsdg::GraphImport::Create(graph, bt32, "x");
    auto theta = create_theta(slt, add, x, x, x);
    auto ui = jlm::llvm::LoopUnrollInfo::create(theta);

    EXPECT_NE(ui, nullptr);
    EXPECT_TRUE(ui->is_additive());
    EXPECT_FALSE(ui->is_subtractive());
    EXPECT_FALSE(ui->is_known());
    EXPECT_FALSE(ui->niterations());
    EXPECT_EQ(ui->theta(), theta);
    EXPECT_EQ(theta->MapPreLoopVar(*ui->idv()).input->origin(), x);
  }

  {
    jlm::rvsdg::Graph graph;

    auto init0 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto init1 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto initm1 = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0xFFFFFFFF)
                      .output(0);

    auto step1 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto step0 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto stepm1 = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0xFFFFFFFF)
                      .output(0);
    auto step2 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2).output(0);

    auto end100 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);

    auto theta = create_theta(ult, add, init0, step1, end100);
    auto ui = jlm::llvm::LoopUnrollInfo::create(theta);
    EXPECT_TRUE(ui && *ui->niterations() == 100);

    theta = create_theta(ule, add, init0, step1, end100);
    ui = jlm::llvm::LoopUnrollInfo::create(theta);
    EXPECT_TRUE(ui && *ui->niterations() == 101);

    theta = create_theta(ugt, sub, end100, stepm1, init0);
    ui = jlm::llvm::LoopUnrollInfo::create(theta);
    EXPECT_TRUE(ui && *ui->niterations() == 100);

    theta = create_theta(sge, sub, end100, stepm1, init0);
    ui = jlm::llvm::LoopUnrollInfo::create(theta);
    EXPECT_TRUE(ui && *ui->niterations() == 101);

    theta = create_theta(ult, add, init0, step0, end100);
    ui = jlm::llvm::LoopUnrollInfo::create(theta);
    EXPECT_TRUE(ui && !ui->niterations());

    theta = create_theta(eq, add, initm1, step1, end100);
    ui = jlm::llvm::LoopUnrollInfo::create(theta);
    EXPECT_TRUE(ui && *ui->niterations() == 101);

    theta = create_theta(eq, add, init1, step2, end100);
    ui = jlm::llvm::LoopUnrollInfo::create(theta);
    EXPECT_TRUE(ui && !ui->niterations());
  }
}

TEST(LoopUnrollingTests, test_known_boundaries)
{
  jlm::llvm::IntegerUltOperation ult(32);
  jlm::llvm::IntegerSgtOperation sgt(32);
  jlm::llvm::IntegerAddOperation add(32);
  jlm::llvm::IntegerSubOperation sub(32);

  {
    jlm::rvsdg::Graph graph;

    auto init = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 4).output(0);

    auto theta = create_theta(ult, add, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 4);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is greater than or equal the number of iterations.
      The loop should be fully unrolled and the theta removed.
    */
    EXPECT_EQ(nthetas(&graph.GetRootRegion()), 0u);
  }

  {
    jlm::rvsdg::Graph graph;

    auto init = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);

    auto theta = create_theta(ult, add, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 2);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is a multiple of the number of iterations.
      We should only find one (unrolled) theta.
    */
    EXPECT_EQ(nthetas(&graph.GetRootRegion()), 1u);
  }

  {
    jlm::rvsdg::Graph graph;

    auto init = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);

    auto theta = create_theta(ult, add, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 3);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is NOT a multiple of the number of iterations
      and we have one remaining iteration. We should find only the
      unrolled theta and the body of the old theta as epilogue.
    */
    EXPECT_EQ(nthetas(&graph.GetRootRegion()), 1u);
  }

  {
    jlm::rvsdg::Graph graph;

    auto init =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);
    auto step =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, -1).output(0);
    auto end = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);

    auto theta = create_theta(sgt, sub, init, step, end);
    //		jlm::rvsdg::view(graph, stdout);
    jlm::llvm::unroll(theta, 6);
    //		jlm::rvsdg::view(graph, stdout);
    /*
      The unroll factor is NOT a multiple of the number of iterations
      and we have four remaining iterations. We should find two thetas:
      one unrolled theta and one theta for the residual iterations.
    */
    EXPECT_EQ(nthetas(&graph.GetRootRegion()), 2u);
  }
}

TEST(LoopUnrollingTests, test_unknown_boundaries)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bt = jlm::rvsdg::BitType::Create(32);
  TestOperation op({ bt }, { bt });

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, bt, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, bt, "y");

  auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto lv1 = theta->AddLoopVar(x);
  auto lv2 = theta->AddLoopVar(y);

  auto one = IntegerConstantOperation::Create(*theta->subregion(), 32, 1).output(0);
  auto add = IntegerAddOperation::createNode(32, *lv1.pre, *one).output(0);
  auto cmp = IntegerUltOperation::createNode(32, *add, *lv2.pre).output(0);
  auto & matchNode = MatchOperation::CreateNode(*cmp, { { 1, 0 } }, 1, 2);

  lv1.post->divert_to(add);

  theta->set_predicate(matchNode.output(0));

  auto & ex1 = GraphExport::Create(*lv1.output, "x");

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::LoopUnrolling loopunroll(2);
  loopunroll.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);

  auto node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*ex1.origin());
  EXPECT_NE(node, nullptr);
  node = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::GammaNode>(*node->input(1)->origin());
  EXPECT_NE(node, nullptr);

  /* Create cleaner output */
  DeadNodeElimination dne;
  dne.Run(rm, statisticsCollector);
  //	jlm::rvsdg::view(graph, stdout);
}

static std::vector<jlm::rvsdg::ThetaNode *>
find_thetas(jlm::rvsdg::Region * region)
{
  std::vector<jlm::rvsdg::ThetaNode *> thetas;
  for (auto & node : jlm::rvsdg::TopDownTraverser(region))
  {
    if (auto theta = dynamic_cast<jlm::rvsdg::ThetaNode *>(node))
      thetas.push_back(theta);
  }

  return thetas;
}

TEST(LoopUnrollingTests, test_nested_theta)
{
  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto init = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
  auto step = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
  auto end = jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 97).output(0);

  /* Outer loop */
  auto otheta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());

  auto lvo_init = otheta->AddLoopVar(init);
  auto lvo_step = otheta->AddLoopVar(step);
  auto lvo_end = otheta->AddLoopVar(end);

  auto add = jlm::llvm::IntegerAddOperation::createNode(32, *lvo_init.pre, *lvo_step.pre).output(0);
  auto compare = jlm::llvm::IntegerUltOperation::createNode(32, *add, *lvo_end.pre).output(0);
  auto & matchNode = jlm::rvsdg::MatchOperation::CreateNode(*compare, { { 1, 1 } }, 0, 2);
  otheta->set_predicate(matchNode.output(0));
  lvo_init.post->divert_to(add);

  /* First inner loop in the original loop */
  auto inner_theta = jlm::rvsdg::ThetaNode::create(otheta->subregion());

  auto inner_init =
      jlm::llvm::IntegerConstantOperation::Create(*otheta->subregion(), 32, 0).output(0);
  auto lvi_init = inner_theta->AddLoopVar(inner_init);
  auto lvi_step = inner_theta->AddLoopVar(lvo_step.pre);
  auto lvi_end = inner_theta->AddLoopVar(lvo_end.pre);

  auto inner_add =
      jlm::llvm::IntegerAddOperation::createNode(32, *lvi_init.pre, *lvi_step.pre).output(0);
  auto inner_compare =
      jlm::llvm::IntegerUltOperation::createNode(32, *inner_add, *lvi_end.pre).output(0);
  auto & innerMatchNode =
      jlm::rvsdg::MatchOperation::CreateNode(*inner_compare, { { 1, 1 } }, 0, 2);
  inner_theta->set_predicate(innerMatchNode.output(0));
  lvi_init.post->divert_to(inner_add);

  /* Nested inner loop */
  auto inner_nested_theta = jlm::rvsdg::ThetaNode::create(inner_theta->subregion());

  auto inner_nested_init =
      jlm::llvm::IntegerConstantOperation::Create(*inner_theta->subregion(), 32, 0).output(0);
  auto lvi_nested_init = inner_nested_theta->AddLoopVar(inner_nested_init);
  auto lvi_nested_step = inner_nested_theta->AddLoopVar(lvi_step.pre);
  auto lvi_nested_end = inner_nested_theta->AddLoopVar(lvi_end.pre);

  auto inner_nested_add =
      jlm::llvm::IntegerAddOperation::createNode(32, *lvi_nested_init.pre, *lvi_nested_step.pre)
          .output(0);
  auto inner_nested_compare =
      jlm::llvm::IntegerUltOperation::createNode(32, *inner_nested_add, *lvi_nested_end.pre)
          .output(0);
  auto & innerNestedMatchNode =
      jlm::rvsdg::MatchOperation::CreateNode(*inner_nested_compare, { { 1, 1 } }, 0, 2);
  inner_nested_theta->set_predicate(innerNestedMatchNode.output(0));
  lvi_nested_init.post->divert_to(inner_nested_add);

  /* Second inner loop in the original loop */
  auto inner2_theta = jlm::rvsdg::ThetaNode::create(otheta->subregion());

  auto inner2_init =
      jlm::llvm::IntegerConstantOperation::Create(*otheta->subregion(), 32, 0).output(0);
  auto lvi2_init = inner2_theta->AddLoopVar(inner2_init);
  auto lvi2_step = inner2_theta->AddLoopVar(lvo_step.pre);
  auto lvi2_end = inner2_theta->AddLoopVar(lvo_end.pre);

  auto inner2_add =
      jlm::llvm::IntegerAddOperation::createNode(32, *lvi2_init.pre, *lvi2_step.pre).output(0);
  auto inner2_compare =
      jlm::llvm::IntegerUltOperation::createNode(32, *inner2_add, *lvi2_end.pre).output(0);
  auto & inner2MatchNode =
      jlm::rvsdg::MatchOperation::CreateNode(*inner2_compare, { { 1, 1 } }, 0, 2);
  inner2_theta->set_predicate(inner2MatchNode.output(0));
  lvi2_init.post->divert_to(inner2_add);

  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::LoopUnrolling loopunroll(4);
  loopunroll.Run(rm, statisticsCollector);
  /*
    The outher theta should contain two inner thetas
  */
  EXPECT_EQ(nthetas(otheta->subregion()), 2u);
  /*
    The outer theta should not be unrolled and since the
    original graph contains 7 nodes and the unroll factor
    is 4 an unrolled theta should have around 28 nodes. So
    we check for less than 20 nodes in case an updated
    unroll algorithm would hoist code from the inner
    thetas.
  */
  EXPECT_LE(otheta->subregion()->numNodes(), 20u);
  /*
    The inner theta should not be unrolled and since the
    original graph contains 5 nodes and the unroll factor
    is 4 an unrolled theta should have around 20 nodes. So
    we check for less than 15 nodes in case an updated
    unroll algorithm would hoist code from the inner
    thetas.
  */
  EXPECT_LE(inner_theta->subregion()->numNodes(), 15u);
  /*
    The innermost theta should be unrolled and since the
    original graph contains 3 nodes and the unroll factor
    is 4 an unrolled theta should have around 12 nodes. So
    we check for more than 7 nodes in case an updated
    unroll algorithm would hoist code from the inner
    thetas.
  */
  auto thetas = find_thetas(inner_theta->subregion());
  EXPECT_EQ(thetas.size(), 1u);
  EXPECT_GE(thetas[0]->subregion()->numNodes(), 7u);
  /*
    The second inner theta should be unrolled and since
    the original graph contains 3 nodes and the unroll
    factor is 4 an unrolled theta should have around 12
    nodes. So we check for less than 7 nodes in case an
    updated unroll algorithm would hoist code from the
    innner thetas.
  */
  thetas = find_thetas(otheta->subregion());
  EXPECT_EQ(thetas.size(), 2u);
  EXPECT_GE(thetas[1]->subregion()->numNodes(), 7u);
  //	jlm::rvsdg::view(graph, stdout);
  jlm::llvm::unroll(otheta, 4);
  //	jlm::rvsdg::view(graph, stdout);
  /*
    After unrolling the outher theta four times it should
    now contain 8 thetas.
  */
  thetas = find_thetas(&graph.GetRootRegion());
  EXPECT_EQ(thetas.size(), 3u);
  EXPECT_EQ(nthetas(thetas[0]->subregion()), 8u);
}

TEST(LoopUnrollingTests, test_zero_iterations_is_rejected_any_factor)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  for (const size_t factor : { 2u, 3u, 4u, 8u })
  {
    Graph graph;
    auto & region = graph.GetRootRegion();
    auto init = IntegerConstantOperation::Create(region, 32, 7).output(0);
    auto step = IntegerConstantOperation::Create(region, 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(region, 32, 7).output(0);
    auto theta = create_theta(IntegerUltOperation(32), IntegerAddOperation(32), init, step, end);
    auto ui = jlm::llvm::LoopUnrollInfo::create(theta);
    ASSERT_NE(ui, nullptr);
    ASSERT_NE(ui->niterations(), nullptr);
    EXPECT_EQ(*ui->niterations(), IntegerValueRepresentation(32, 0));
    const auto nodesBefore = region.numNodes();

    jlm::llvm::unroll(theta, factor);

    EXPECT_EQ(nthetas(&region), 1u) << "factor=" << factor;
    EXPECT_EQ(region.numNodes(), nodesBefore) << "factor=" << factor;
  }
}

static std::optional<jlm::llvm::IntegerValueRepresentation>
evaluate_guard(
    jlm::rvsdg::Output * output,
    const std::unordered_map<jlm::rvsdg::Output *, jlm::llvm::IntegerValueRepresentation> & inputs)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;
  const auto it = inputs.find(output);
  if (it != inputs.end())
    return it->second;
  auto node = TryGetOwnerNode<SimpleNode>(*output);
  if (!node)
    return std::nullopt;
  if (auto constant = dynamic_cast<const IntegerConstantOperation *>(&node->GetOperation()))
    return constant->Representation();
  if (node->ninputs() != 2)
    return std::nullopt;
  auto a = evaluate_guard(node->input(0)->origin(), inputs);
  auto b = evaluate_guard(node->input(1)->origin(), inputs);
  if (!a || !b)
    return std::nullopt;
  if (is<IntegerAddOperation>(node))
    return a->add(*b);
  if (is<IntegerSubOperation>(node))
    return a->sub(*b);
  if (is<IntegerMulOperation>(node))
    return a->mul(*b);
  if (is<IntegerAndOperation>(node))
    return a->land(*b);
  if (is<IntegerOrOperation>(node))
    return a->lor(*b);
  if (is<IntegerXorOperation>(node))
    return a->lxor(*b);
  if (is<IntegerUltOperation>(node))
    return IntegerValueRepresentation(a->ult(*b));
  return std::nullopt;
}

TEST(LoopUnrollingTests, test_runtime_batch_guard_matches_loop_semantics)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;
  jlm::llvm::LlvmRvsdgModule module(jlm::util::FilePath(""), "", "");
  auto & graph = module.Rvsdg();
  auto type = BitType::Create(8);
  auto init = &GraphImport::Create(graph, type, "init");
  auto step = &GraphImport::Create(graph, type, "step");
  auto end = &GraphImport::Create(graph, type, "end");
  auto theta = create_theta(IntegerUltOperation(8), IntegerAddOperation(8), init, step, end);
  auto & result = GraphExport::Create(*theta->GetLoopVars()[0].output, "result");
  jlm::llvm::LoopUnrolling(4).Run(module, statisticsCollector);

  auto residual = TryGetOwnerNode<GammaNode>(*result.origin());
  ASSERT_NE(residual, nullptr);
  GammaNode * batch = nullptr;
  for (auto & node : graph.GetRootRegion().Nodes())
    if (auto gamma = dynamic_cast<GammaNode *>(&node); gamma && gamma != residual)
      batch = gamma;
  ASSERT_NE(batch, nullptr);
  auto match = TryGetOwnerNode<SimpleNode>(*batch->predicate()->origin());
  ASSERT_NE(match, nullptr);
  ASSERT_TRUE(is<MatchOperation>(match));

  for (int initial : { 0, 1, 200 })
    for (int increment : { 1, 3, 30, 128 })
      for (int bound : { 0, 1, 4, 100, 255 })
      {
        IntegerValueRepresentation value(8, initial), stride(8, increment), limit(8, bound);
        bool expected = true;
        for (size_t copy = 1; copy < 4; ++copy)
        {
          value = value.add(stride);
          expected = expected && value.ult(limit) == '1';
        }
        auto actual = evaluate_guard(
            match->input(0)->origin(),
            { { init, { 8, initial } }, { step, stride }, { end, limit } });
        ASSERT_TRUE(actual.has_value());
        EXPECT_EQ(actual->to_uint(), expected ? 1u : 0u)
            << "init=" << initial << " step=" << increment << " end=" << bound;
      }
}

TEST(LoopUnrollingTests, test_residual_guard_includes_batch_decision)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;
  jlm::llvm::LlvmRvsdgModule module(jlm::util::FilePath(""), "", "");
  auto & graph = module.Rvsdg();
  auto type = BitType::Create(8);
  auto init = &GraphImport::Create(graph, type, "init");
  auto step = &GraphImport::Create(graph, type, "step");
  auto end = &GraphImport::Create(graph, type, "end");
  auto theta = create_theta(IntegerUltOperation(8), IntegerAddOperation(8), init, step, end);
  auto & result = GraphExport::Create(*theta->GetLoopVars()[0].output, "result");
  jlm::llvm::LoopUnrolling(2).Run(module, statisticsCollector);

  auto residual = TryGetOwnerNode<GammaNode>(*result.origin());
  ASSERT_NE(residual, nullptr);
  auto match = TryGetOwnerNode<SimpleNode>(*residual->predicate()->origin());
  ASSERT_NE(match, nullptr);
  ASSERT_TRUE(is<MatchOperation>(match));
  auto condition = TryGetOwnerNode<SimpleNode>(*match->input(0)->origin());
  ASSERT_NE(condition, nullptr);
  EXPECT_TRUE(is<IntegerOrOperation>(condition))
      << "The residual guard must combine the negated batch decision with the loop comparison";
}
