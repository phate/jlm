/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/unroll.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/TestOperations.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

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

/**
 * Builds a theta equivalent to \code while (idv < end) { idv += step; } \endcode using the LLVM
 * integer operation family. This mirrors what LlvmModuleConversion produces and is therefore the
 * shape the loop unrolling pass has to handle in practice.
 */
static jlm::rvsdg::ThetaNode *
createIntegerTheta(
    const size_t numBits,
    jlm::rvsdg::Output * init,
    jlm::rvsdg::Output * step,
    jlm::rvsdg::Output * end,
    const bool subtract = false)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto graph = init->region()->graph();

  auto theta = ThetaNode::create(&graph->GetRootRegion());
  auto idv = theta->AddLoopVar(init);
  auto lvs = theta->AddLoopVar(step);
  auto lve = theta->AddLoopVar(end);

  auto & arm = subtract ? IntegerSubOperation::createNode(numBits, *idv.pre, *lvs.pre)
                        : IntegerAddOperation::createNode(numBits, *idv.pre, *lvs.pre);
  auto & cmp = IntegerUltOperation::createNode(numBits, *arm.output(0), *lve.pre);
  auto & matchNode = MatchOperation::CreateNode(*cmp.output(0), { { 1, 1 } }, 0, 2);

  idv.post->divert_to(arm.output(0));
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
    auto initm1 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, -1).output(0);

    auto step1 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto step0 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto stepm1 =
        jlm::llvm::IntegerConstantOperation::Create(graph.GetRootRegion(), 32, -1).output(0);
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
  using namespace jlm::llvm;

  jlm::llvm::IntegerUltOperation ult(32);
  jlm::llvm::IntegerSgtOperation sgt(32);
  jlm::llvm::IntegerAddOperation add(32);
  jlm::llvm::IntegerSubOperation sub(32);

  {
    jlm::rvsdg::Graph graph;

    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 4).output(0);

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

    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);

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

    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);

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

    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, -1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);

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
  auto add = jlm::llvm::IntegerAddOperation::createNode(32, *lv1.pre, *one).output(0);
  auto cmp = jlm::llvm::IntegerUltOperation::createNode(32, *add, *lv2.pre).output(0);
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
  using namespace jlm::llvm;

  jlm::llvm::LlvmRvsdgModule rm(jlm::util::FilePath(""), "", "");
  auto & graph = rm.Rvsdg();

  auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
  auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
  auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 97).output(0);

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

  auto inner_init = IntegerConstantOperation::Create(*otheta->subregion(), 32, 0).output(0);
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
      IntegerConstantOperation::Create(*inner_theta->subregion(), 32, 0).output(0);
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

  auto inner2_init = IntegerConstantOperation::Create(*otheta->subregion(), 32, 0).output(0);
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

/**
 * Builds the loop shape found in test-factorial.c, using the LLVM integer operation family:
 * \code while (idv > 1) { idv += -1; } \endcode
 * The bound and the step are constants defined inside the loop body, so recognizing this loop
 * requires the analysis to see through them.
 */
static jlm::rvsdg::ThetaNode *
createDecrementingTheta(const size_t numBits, jlm::rvsdg::Output * init)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto theta = ThetaNode::create(init->region());
  auto subregion = theta->subregion();
  auto idv = theta->AddLoopVar(init);

  auto & one = IntegerConstantOperation::Create(*subregion, numBits, 1);
  auto & minusOne = IntegerConstantOperation::Create(*subregion, numBits, -1);
  auto & arm = IntegerAddOperation::createNode(numBits, *idv.pre, *minusOne.output(0));
  auto & cmp = IntegerUgtOperation::createNode(numBits, *arm.output(0), *one.output(0));
  auto & matchNode = MatchOperation::CreateNode(*cmp.output(0), { { 1, 1 } }, 0, 2);

  idv.post->divert_to(arm.output(0));
  theta->set_predicate(matchNode.output(0));

  return theta;
}

/**
 * Resolves the comparison that feeds a theta or gamma predicate, i.e. predicate -> Match -> cmp.
 */
static jlm::rvsdg::SimpleNode *
findPredicateComparison(jlm::rvsdg::Output * predicate)
{
  using namespace jlm::rvsdg;

  auto matchNode = TryGetOwnerNode<SimpleNode>(*predicate);
  if (!matchNode || !is<MatchOperation>(matchNode))
    return nullptr;

  return TryGetOwnerNode<SimpleNode>(*matchNode->input(0)->origin());
}

static std::vector<jlm::rvsdg::GammaNode *>
findGammas(jlm::rvsdg::Region * region)
{
  std::vector<jlm::rvsdg::GammaNode *> gammas;
  for (const auto & node : region->Nodes())
  {
    if (auto gamma = dynamic_cast<const jlm::rvsdg::GammaNode *>(&node))
      gammas.push_back(const_cast<jlm::rvsdg::GammaNode *>(gamma));
  }

  return gammas;
}

/**
 * Evaluates the dataflow graph rooted at \p output, taking every \ref IntegerConstantOperation at
 * face value and every \ref GraphImport from \p environment. Returns std::nullopt as soon as a node
 * is reached that the evaluator does not understand.
 *
 * The guard of an unrolled loop is pure dataflow over the loop's initial value, bound and step, so
 * folding it is enough to ask whether the guard is taken for a concrete set of inputs. That is
 * what the wrap tests need: the defect they cover is only observable in the *value* of the guard,
 * not in the shape of the graph.
 */
static std::optional<jlm::llvm::IntegerValueRepresentation>
evaluateConstantDataflow(
    jlm::rvsdg::Output * output,
    const std::unordered_map<std::string, uint64_t> & environment,
    const size_t numBits)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  if (const auto [_, constant] = TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*output);
      constant != nullptr)
    return constant->Representation();

  if (auto graphImport = dynamic_cast<GraphImport *>(output))
  {
    const auto it = environment.find(graphImport->Name());
    if (it == environment.end())
      return std::nullopt;

    return IntegerValueRepresentation(numBits, it->second);
  }

  auto node = TryGetOwnerNode<SimpleNode>(*output);
  if (node == nullptr || node->noutputs() != 1 || node->ninputs() != 2)
    return std::nullopt;

  const auto lhs = evaluateConstantDataflow(node->input(0)->origin(), environment, numBits);
  const auto rhs = evaluateConstantDataflow(node->input(1)->origin(), environment, numBits);
  if (!lhs || !rhs)
    return std::nullopt;

  if (is<IntegerAddOperation>(node))
    return lhs->add(*rhs);
  if (is<IntegerSubOperation>(node))
    return lhs->sub(*rhs);
  if (is<IntegerMulOperation>(node))
    return lhs->mul(*rhs);
  if (is<IntegerUDivOperation>(node))
    return lhs->udiv(*rhs);
  if (is<IntegerURemOperation>(node))
    return lhs->umod(*rhs);
  if (is<IntegerAndOperation>(node))
    return lhs->land(*rhs);
  if (is<IntegerOrOperation>(node))
    return lhs->lor(*rhs);
  if (is<IntegerXorOperation>(node))
    return lhs->lxor(*rhs);

  /*
    The guard's root is the loop's comparison. Its result feeds a MatchOperation, so evaluate it to
    the 1-bit value 1 (taken) or 0 (not taken).
  */
  /*
    The comparison helpers return a char '0'/'1', not a bool: '0' is non-zero, so it must be
    compared against '1' explicitly or every false comparison would read as true.
  */
  auto comparisonHolds = std::optional<bool>();
  if (is<IntegerUltOperation>(node))
    comparisonHolds = lhs->ult(*rhs) == '1';
  else if (is<IntegerUleOperation>(node))
    comparisonHolds = lhs->ule(*rhs) == '1';
  else if (is<IntegerUgtOperation>(node))
    comparisonHolds = lhs->ugt(*rhs) == '1';
  else if (is<IntegerUgeOperation>(node))
    comparisonHolds = lhs->uge(*rhs) == '1';
  else if (is<IntegerSltOperation>(node))
    comparisonHolds = lhs->slt(*rhs) == '1';
  else if (is<IntegerSleOperation>(node))
    comparisonHolds = lhs->sle(*rhs) == '1';
  else if (is<IntegerSgtOperation>(node))
    comparisonHolds = lhs->sgt(*rhs) == '1';
  else if (is<IntegerSgeOperation>(node))
    comparisonHolds = lhs->sge(*rhs) == '1';
  else if (is<IntegerEqOperation>(node))
    comparisonHolds = lhs->eq(*rhs) == '1';
  else if (is<IntegerNeOperation>(node))
    comparisonHolds = lhs->ne(*rhs) == '1';

  if (comparisonHolds)
    return IntegerValueRepresentation(1, *comparisonHolds ? 1 : 0);

  return std::nullopt;
}

/**
 * Collects every loop comparison that the batch guard of \p gamma is built from.
 *
 * The batch guard is a conjunction of the loop's own comparison evaluated at each intermediate
 * induction-variable value, so it is a tree of `and` nodes whose leaves are comparisons. This walks
 * that tree and appends the comparisons in the order they are reached.
 */
static void
collectGuardComparisons(
    jlm::rvsdg::Output * output,
    std::vector<jlm::rvsdg::SimpleNode *> & comparisons)
{
  using namespace jlm::rvsdg;

  auto node = TryGetOwnerNode<SimpleNode>(*output);
  if (node == nullptr)
    return;

  if (jlm::rvsdg::is<jlm::llvm::IntegerAndOperation>(node))
  {
    collectGuardComparisons(node->input(0)->origin(), comparisons);
    collectGuardComparisons(node->input(1)->origin(), comparisons);
    return;
  }

  comparisons.push_back(node);
}

/**
 * Returns every comparison the batch guard of \p gamma is built from.
 */
static std::vector<jlm::rvsdg::SimpleNode *>
collectGuardComparisons(jlm::rvsdg::GammaNode * gamma)
{
  using namespace jlm::rvsdg;

  std::vector<SimpleNode *> comparisons;

  auto matchNode = TryGetOwnerNode<SimpleNode>(*gamma->predicate()->origin());
  if (matchNode == nullptr || !is<MatchOperation>(matchNode))
    return comparisons;

  collectGuardComparisons(matchNode->input(0)->origin(), comparisons);

  return comparisons;
}

/**
 * Returns every comparison the guard with the given \p predicate is built from.
 */
static std::vector<jlm::rvsdg::SimpleNode *>
collectGuardComparisons(jlm::rvsdg::Output * predicate)
{
  using namespace jlm::rvsdg;

  std::vector<SimpleNode *> comparisons;

  auto matchNode = TryGetOwnerNode<SimpleNode>(*predicate);
  if (matchNode == nullptr || !is<MatchOperation>(matchNode))
    return comparisons;

  collectGuardComparisons(matchNode->input(0)->origin(), comparisons);

  return comparisons;
}

/**
 * Returns a copy of the comparison operation that \p theta uses for its exit condition.
 */
static std::unique_ptr<jlm::rvsdg::Operation>
loopComparisonOperation(jlm::rvsdg::ThetaNode & theta)
{
  using namespace jlm::rvsdg;

  auto comparison = findPredicateComparison(theta.predicate()->origin());
  if (comparison == nullptr)
    return nullptr;

  return comparison->GetOperation().copy();
}

/**
 * Returns the known constant value behind \p output, if it is a constant.
 */
static std::optional<uint64_t>
knownConstantValue(jlm::rvsdg::Output & output)
{
  const auto [_, constant] =
      jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<jlm::llvm::IntegerConstantOperation>(output);
  if (constant == nullptr || !constant->Representation().is_known())
    return std::nullopt;

  return constant->Representation().to_uint();
}

/**
 * Describes the loop's original bound, captured before the transformation runs.
 *
 * A pointer cannot be kept because the original bound node is destroyed together with the theta, so
 * the shape of the bound is recorded instead.
 */
struct BoundExpectation
{
  /** The bound's value, when the bound is a known constant. */
  std::optional<uint64_t> constantValue;
  /** True when the bound is a region argument rather than a constant. */
  bool isArgument = false;
};

/**
 * Captures the shape of \p bound before the transformation removes it.
 */
static BoundExpectation
captureBoundExpectation(jlm::rvsdg::Output & bound)
{
  BoundExpectation expectation;
  expectation.constantValue = knownConstantValue(bound);
  expectation.isArgument = dynamic_cast<jlm::rvsdg::RegionArgument *>(&bound) != nullptr;
  return expectation;
}

/**
 * Returns true when \p comparison still uses the loop's own bound.
 *
 * The same bound legitimately appears as a constant in the enclosing region and as a region
 * argument inside the loop, so both forms are accepted. What must not happen is a *different*
 * constant, i.e. the folded `end - factor*step` value that the transformation used to synthesise.
 */
static bool
comparesAgainstLoopBound(jlm::rvsdg::SimpleNode & comparison, const BoundExpectation & expected)
{
  using namespace jlm::rvsdg;

  for (const auto & input : comparison.Inputs())
  {
    if (dynamic_cast<RegionArgument *>(input.origin()) != nullptr)
      return true;

    const auto value = knownConstantValue(*input.origin());
    if (!value)
      continue;

    /*
      A constant operand is the loop's bound only when it matches the original value. If the
      original bound was not a known constant, any constant here is a synthesised one.
    */
    if (expected.constantValue && *value == *expected.constantValue)
      return true;
  }

  return false;
}

/**
 * Recursively walks the dataflow rooted at \p output and reports whether any multiplication by a
 * constant equal to \p factor is reachable.
 *
 * The guard must advance the induction variable one step at a time; a `factor * step` term is the
 * closed form that overflows for descending loops.
 */
static bool
multipliesByConstant(
    jlm::rvsdg::Output & output,
    const size_t factor,
    jlm::util::HashSet<const jlm::rvsdg::Output *> & visited)
{
  using namespace jlm::rvsdg;

  if (!visited.insert(&output))
    return false;

  auto node = TryGetOwnerNode<SimpleNode>(output);
  if (node == nullptr)
    return false;

  if (jlm::rvsdg::is<jlm::llvm::IntegerMulOperation>(node))
  {
    for (const auto & input : node->Inputs())
    {
      const auto [_, constant] =
          TryGetSimpleNodeAndOptionalOp<jlm::llvm::IntegerConstantOperation>(*input.origin());
      if (constant != nullptr && constant->Representation().is_known()
          && constant->Representation().to_uint() == factor)
        return true;
    }
  }

  for (const auto & input : node->Inputs())
  {
    if (multipliesByConstant(*input.origin(), factor, visited))
      return true;
  }

  return false;
}

/**
 * Returns true when the batch guard of \p gamma computes a multiplication by \p factor anywhere in
 * its dataflow.
 */
static bool
guardMultipliesByFactor(jlm::rvsdg::GammaNode * gamma, const size_t factor)
{
  using namespace jlm::rvsdg;

  auto matchNode = TryGetOwnerNode<SimpleNode>(*gamma->predicate()->origin());
  if (matchNode == nullptr || !is<MatchOperation>(matchNode))
    return false;

  jlm::util::HashSet<const Output *> visited;
  return multipliesByConstant(*matchNode->input(0)->origin(), factor, visited);
}

TEST(LoopUnrollingTests, test_integer_loop_unrollinfo)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto bitType = BitType::Create(32);

  {
    Graph graph;
    auto x = &GraphImport::Create(graph, bitType, "x");
    auto theta = createIntegerTheta(32, x, x, x);
    auto ui = LoopUnrollInfo::create(theta);

    ASSERT_NE(ui, nullptr);
    EXPECT_TRUE(ui->is_additive());
    EXPECT_FALSE(ui->is_subtractive());
    EXPECT_FALSE(ui->is_known());
    EXPECT_FALSE(ui->niterations());
    EXPECT_EQ(ui->nbits(), 32u);
    EXPECT_EQ(ui->theta(), theta);
  }

  {
    Graph graph;
    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);

    auto theta = createIntegerTheta(32, init, step, end);
    auto ui = LoopUnrollInfo::create(theta);

    ASSERT_NE(ui, nullptr);
    EXPECT_TRUE(ui->is_known());
    ASSERT_NE(ui->niterations(), nullptr);
    EXPECT_EQ(*ui->niterations(), 100u);
  }

  {
    Graph graph;
    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);

    auto theta = createIntegerTheta(32, init, step, end, true);
    auto ui = LoopUnrollInfo::create(theta);

    ASSERT_NE(ui, nullptr);
    EXPECT_TRUE(ui->is_subtractive());
    EXPECT_FALSE(ui->is_additive());
  }
}

TEST(LoopUnrollingTests, test_integer_loop_known_boundaries)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  {
    Graph graph;
    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 4).output(0);

    auto theta = createIntegerTheta(32, init, step, end);

    /*
      The unroll factor is greater than or equal to the number of iterations, so the loop should be
      fully unrolled and the theta removed.
    */
    unroll(theta, 4);
    EXPECT_EQ(nthetas(&graph.GetRootRegion()), 0u);
  }

  {
    Graph graph;
    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 100).output(0);

    auto theta = createIntegerTheta(32, init, step, end);

    unroll(theta, 2);
    EXPECT_EQ(nthetas(&graph.GetRootRegion()), 1u);
  }

  {
    Graph graph;
    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    /*
      101 iterations with an unroll factor of 2 leaves a remainder, which exercises the residual
      iteration handling of the transformation.
    */
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 101).output(0);

    auto theta = createIntegerTheta(32, init, step, end);

    unroll(theta, 2);
    EXPECT_LE(nthetas(&graph.GetRootRegion()), 2u);
  }
}

TEST(LoopUnrollingTests, test_integer_loop_unknown_boundary)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  LlvmRvsdgModule module(jlm::util::FilePath(""), "", "");
  auto & graph = module.Rvsdg();

  auto bitType = BitType::Create(32);
  auto n = &GraphImport::Create(graph, bitType, "n");

  auto theta = createDecrementingTheta(32, n);
  auto & loopResult = GraphExport::Create(*theta->GetLoopVars()[0].output, "result");

  /*
    The transformation removes the original theta, so record its comparison operation and bound
    beforehand. The bound is whichever comparison operand is not produced by the arm operation.
  */
  auto loopComparison = loopComparisonOperation(*theta);
  ASSERT_NE(loopComparison, nullptr);

  auto originalComparison = findPredicateComparison(theta->predicate()->origin());
  ASSERT_NE(originalComparison, nullptr);

  Output * loopBound = nullptr;
  for (const auto & input : originalComparison->Inputs())
  {
    auto arm = TryGetOwnerNode<SimpleNode>(*input.origin());
    if (arm == nullptr || (!is<IntegerAddOperation>(arm) && !is<IntegerSubOperation>(arm)))
      loopBound = input.origin();
  }
  ASSERT_NE(loopBound, nullptr);

  const auto boundExpectation = captureBoundExpectation(*loopBound);

  const size_t factor = 4;
  LoopUnrolling loopUnrolling(factor);
  loopUnrolling.Run(module, statisticsCollector);

  /*
    The loop has no statically known trip count, so it is unrolled behind a gamma that guards the
    unrolled iterations, with the original theta kept for the residual iterations.
  */
  auto outerGamma = TryGetOwnerNode<GammaNode>(*loopResult.origin());
  ASSERT_NE(outerGamma, nullptr);
  EXPECT_EQ(nthetas(outerGamma->subregion(1)), 1u);

  /*
    The export points at the residual gamma, whose theta correctly keeps the loop's original
    predicate. The unrolled gamma is the other gamma in the root region.
  */
  GammaNode * unrolledGamma = nullptr;
  for (auto gamma : findGammas(&graph.GetRootRegion()))
  {
    if (gamma != outerGamma)
      unrolledGamma = gamma;
  }
  ASSERT_NE(unrolledGamma, nullptr);

  auto thetas = find_thetas(unrolledGamma->subregion(1));
  ASSERT_EQ(thetas.size(), 1u);

  /*
    The theta inside the unrolled branch repeats only after a full batch of `factor` iterations, so
    its predicate must check every copy of the next batch. It is therefore a MatchOperation over a
    conjunction of the loop's own comparison, not a single comparison.
  */
  auto innerComparisons = collectGuardComparisons(thetas[0]->predicate()->origin());
  EXPECT_EQ(innerComparisons.size(), factor)
      << "The unrolled theta must check every copy of the next batch";

  auto gammaComparisons = collectGuardComparisons(unrolledGamma);
  EXPECT_EQ(gammaComparisons.size(), factor - 1)
      << "The enclosing gamma must check every copy after the unconditional first one";

  /*
    Every comparison in both guards must be the loop's original comparison operation, applied to the
    loop's bound. Using a different operation would silently change the loop's exit condition.
  */
  const auto & loopComparisonOperation = *loopComparison;

  for (auto comparison : innerComparisons)
  {
    EXPECT_EQ(comparison->GetOperation(), loopComparisonOperation)
        << "The unrolled theta must reuse the loop's comparison operation";
    EXPECT_TRUE(comparesAgainstLoopBound(*comparison, boundExpectation))
        << "The unrolled theta must compare against the loop's bound, not a folded constant";
  }

  for (auto comparison : gammaComparisons)
  {
    EXPECT_EQ(comparison->GetOperation(), loopComparisonOperation)
        << "The enclosing gamma must reuse the loop's comparison operation";
    EXPECT_TRUE(comparesAgainstLoopBound(*comparison, boundExpectation))
        << "The enclosing gamma must compare against the loop's bound, not a folded constant";
  }

  /*
    The guard must not compute `factor * step`: that closed form is what overflows for descending
    loops. It advances the induction variable one step at a time instead.
  */
  EXPECT_FALSE(guardMultipliesByFactor(unrolledGamma, factor))
      << "The guard must not compute a factor-scaled offset, which can overflow";
}

/**
 * An RVSDG theta is a do-while construct: the body always runs at least once and the predicate is
 * evaluated afterwards. A theta whose bound equals its initial value therefore executes its body
 * exactly once, not zero times.
 *
 * `LoopUnrollInfo::niterations()` reports `(end - start) / step`, which is that execution count.
 * For `start == end` it reports zero. Zero executions cannot be represented by a theta, so the
 * transformation leaves the theta in place.
 */
TEST(LoopUnrollingTests, test_integer_loop_zero_iterations_is_rejected)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  Graph graph;
  auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
  auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
  auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);

  auto theta = createIntegerTheta(32, init, step, end);

  auto ui = LoopUnrollInfo::create(theta);
  ASSERT_NE(ui, nullptr);
  EXPECT_TRUE(ui->is_known());
  ASSERT_NE(ui->niterations(), nullptr);
  EXPECT_EQ(ui->niterations()->to_uint(), 0u);

  const auto nodesBefore = nnodes(&graph.GetRootRegion());

  /*
    The theta and the node count must both be unchanged, i.e. the loop body is not copied.
  */
  unroll(theta, 4);

  EXPECT_EQ(nthetas(&graph.GetRootRegion()), 1u)
      << "A theta with zero reported iterations must be left alone";
  EXPECT_EQ(nnodes(&graph.GetRootRegion()), nodesBefore)
      << "Unrolling a zero-iteration theta must not copy the loop body";
}

/**
 * The same zero-iteration shape must be rejected for several unroll factors, since the
 * transformation is reachable with any factor.
 */
TEST(LoopUnrollingTests, test_integer_loop_zero_iterations_is_rejected_any_factor)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  for (const size_t factor : { 2u, 3u, 4u, 8u })
  {
    Graph graph;
    auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 7).output(0);
    auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
    auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 7).output(0);

    auto theta = createIntegerTheta(32, init, step, end);

    const auto nodesBefore = nnodes(&graph.GetRootRegion());

    unroll(theta, factor);

    EXPECT_EQ(nthetas(&graph.GetRootRegion()), 1u) << "factor=" << factor;
    EXPECT_EQ(nnodes(&graph.GetRootRegion()), nodesBefore) << "factor=" << factor;
  }
}

/**
 * A one-iteration loop is the smallest loop a theta can represent. It is fully unrolled: the body
 * is copied exactly once and the theta disappears.
 */
TEST(LoopUnrollingTests, test_integer_loop_single_iteration)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  Graph graph;
  auto init = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3).output(0);
  auto step = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1).output(0);
  /* `for (i = 3; i < 4; i += 1)`: the body runs once, producing i = 4. */
  auto end = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 4).output(0);

  auto theta = createIntegerTheta(32, init, step, end);
  auto & result = GraphExport::Create(*theta->GetLoopVars()[0].output, "result");

  auto ui = LoopUnrollInfo::create(theta);
  ASSERT_NE(ui, nullptr);
  ASSERT_NE(ui->niterations(), nullptr);
  EXPECT_EQ(ui->niterations()->to_uint(), 1u);

  unroll(theta, 4);

  EXPECT_EQ(nthetas(&graph.GetRootRegion()), 0u);

  /*
    The single iteration adds the step to the initial value, so the exported value is that sum
    rather than the initial value.
  */
  auto arm = TryGetOwnerNode<SimpleNode>(*result.origin());
  ASSERT_NE(arm, nullptr);
  EXPECT_TRUE(is<IntegerAddOperation>(arm));
  EXPECT_EQ(arm->input(0)->origin(), init);
}

/**
 * Builds the loop shape `while (idv < end) { idv += step; }` where `init`, `end` and `step` are
 * *not* constants, i.e. the shape that makes the unrolling pass adjust the bound at runtime.
 *
 * The other guard tests all use constant bounds and steps, which lets the transformation fold the
 * adjusted bound into a constant. That fold is not where the pass goes wrong; the runtime
 * subtraction is. This helper exists so a test can reach that path.
 */
static jlm::rvsdg::ThetaNode *
createRuntimeBoundTheta(jlm::rvsdg::Graph & graph, const size_t numBits)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto bitType = BitType::Create(numBits);
  auto init = &GraphImport::Create(graph, bitType, "init");
  auto end = &GraphImport::Create(graph, bitType, "end");
  auto step = &GraphImport::Create(graph, bitType, "step");

  auto theta = ThetaNode::create(&graph.GetRootRegion());
  auto idv = theta->AddLoopVar(init);
  auto lvs = theta->AddLoopVar(step);
  auto lve = theta->AddLoopVar(end);

  auto & arm = IntegerAddOperation::createNode(numBits, *idv.pre, *lvs.pre);
  auto & cmp = IntegerUltOperation::createNode(numBits, *arm.output(0), *lve.pre);
  auto & matchNode = MatchOperation::CreateNode(*cmp.output(0), { { 1, 1 } }, 0, 2);

  idv.post->divert_to(arm.output(0));
  theta->set_predicate(matchNode.output(0));

  return theta;
}

/**
 * Returns the value of the unrolled-loop guard when the graph's imports are bound by \p
 * environment.
 *
 * The guard is pure dataflow over the loop's pre-loop values, so binding those values is enough to
 * ask whether the unrolled body is entered for a concrete set of inputs.
 */
static bool
evaluateUnrolledGuard(
    jlm::rvsdg::Graph & graph,
    const std::unordered_map<std::string, uint64_t> & environment,
    const size_t numBits)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  auto gammas = findGammas(&graph.GetRootRegion());

  /*
    Unrolling leaves two gammas in the root region: one that guards the unrolled batch and one that
    guards the residual loop. Both hold a theta, so they are told apart by their outputs: the
    residual loop is the one the graph exports, and the batch gamma is the other one.
  */
  jlm::rvsdg::GammaNode * residualGamma = nullptr;
  for (auto result : graph.GetRootRegion().Results())
    residualGamma = TryGetOwnerNode<jlm::rvsdg::GammaNode>(*result->origin());

  EXPECT_NE(residualGamma, nullptr) << "Expected the graph export to come from a gamma";
  if (residualGamma == nullptr)
    return false;

  jlm::rvsdg::GammaNode * batchGamma = nullptr;
  for (auto gamma : gammas)
  {
    if (gamma != residualGamma)
      batchGamma = gamma;
  }

  EXPECT_NE(batchGamma, nullptr) << "Expected a gamma whose then-branch holds the unrolled theta";
  if (batchGamma == nullptr)
    return false;

  /*
    The batch guard is a conjunction of the loop's own comparison, so fold the whole expression
    feeding the MatchOperation instead of looking for one comparison node.
  */
  auto matchNode = TryGetOwnerNode<SimpleNode>(*batchGamma->predicate()->origin());
  EXPECT_NE(matchNode, nullptr) << "Expected the gamma predicate to be a MatchOperation";
  if (matchNode == nullptr || !is<MatchOperation>(matchNode))
    return false;

  const auto predicateValue =
      evaluateConstantDataflow(matchNode->input(0)->origin(), environment, numBits);
  EXPECT_TRUE(predicateValue.has_value()) << "Could not fold the guard predicate to a constant";

  /*
    The guard feeds a MatchOperation that maps the value 1 to the unrolled branch, so the guard is
    taken exactly when the folded conjunction yields 1.
  */
  return predicateValue.has_value() && *predicateValue == 1;
}

/**
 * The guard of an unrolled loop must only admit a batch whose copies are all legal for the
 * original loop predicate.
 *
 * The pass builds the guard as `cmp(idv, end - factor*step)`. For an unsigned comparison that
 * subtraction underflows whenever fewer than `factor*step` values remain, which turns the bound
 * into a huge value and the guard into an unconditional "enter the unrolled body". The body then
 * executes iterations the original loop was not supposed to run. That is what made the
 * region-aware llvm-test-suite job fail: `nsieve-bits`, `fannkuch`, `chomp` and `lowercase` all
 * segfault, and bounded loops became effectively unbounded, which is why the job took four hours
 * instead of ten minutes.
 *
 * This test states the property the guard has to satisfy rather than one particular formula:
 * whenever the guard is taken, all `factor` copies must be legal. Being more conservative than
 * that is allowed, so the test does not pin the exact boundary.
 */
TEST(LoopUnrollingTests, test_integer_loop_guard_is_safe)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  const size_t factor = 4;

  for (const uint64_t step : { 1u, 3u, 7u, 30u })
  {
    /*
      The underflow needs `end < factor*step`, and the batch is illegal once
      `init + (factor-1)*step >= end`. Deriving the bound from `step` puts the interesting values
      (just below, at, and just above `factor*step`) in the grid for every step.
    */
    const std::vector<uint64_t> ends = {
      0, 1, step, 2 * step, 3 * step, factor * step - 1, factor * step, factor * step + 1, 8 * step,
    };

    const std::vector<uint64_t> inits = { 0, 1, step };

    for (const uint64_t init : inits)
    {
      for (const uint64_t end : ends)
      {
        LlvmRvsdgModule module(jlm::util::FilePath(""), "", "");
        auto theta = createRuntimeBoundTheta(module.Rvsdg(), 32);
        GraphExport::Create(*theta->GetLoopVars()[0].output, "result");

        LoopUnrolling loopUnrolling(factor);
        loopUnrolling.Run(module, statisticsCollector);

        /*
          The theta evaluates `idv + step < end` after every body execution, so body copy j
          (1-indexed, j <= factor) is legal exactly when `init + j*step < end` holds in the loop's
          own wrapping arithmetic. Copy 1 always runs. The unrolled body executes `factor` copies,
          so the batch is safe only when every copy after the first is legal. Deriving this copy by
          copy keeps the expectation independent of any closed form that could itself wrap.
        */
        auto advanced = IntegerValueRepresentation(32, static_cast<int64_t>(init));
        const auto stepValue = IntegerValueRepresentation(32, static_cast<int64_t>(step));
        const auto endValue = IntegerValueRepresentation(32, static_cast<int64_t>(end));
        bool allCopiesAreLegal = true;
        for (size_t copy = 1; copy < factor; ++copy)
        {
          advanced = advanced.add(stepValue);
          allCopiesAreLegal = allCopiesAreLegal && advanced.ult(endValue) == '1';
        }

        if (evaluateUnrolledGuard(
                module.Rvsdg(),
                { { "init", init }, { "end", end }, { "step", step } },
                32))
        {
          EXPECT_TRUE(allCopiesAreLegal)
              << "init=" << init << " end=" << end << " step=" << step << " factor=" << factor
              << ": the guard admitted a batch of " << factor
              << " copies, but not all of them are legal for the original loop";
        }
      }
    }
  }
}

/**
 * The guard must still admit batches that clearly fit, otherwise the pass would never unroll
 * anything and the safety test above would be satisfied trivially.
 */
TEST(LoopUnrollingTests, test_integer_loop_guard_admits_batch_that_fits)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  const size_t factor = 4;

  for (const uint64_t step : { 1u, 3u, 7u, 30u })
  {
    for (const uint64_t end : { 1000u, 4096u })
    {
      LlvmRvsdgModule module(jlm::util::FilePath(""), "", "");
      auto theta = createRuntimeBoundTheta(module.Rvsdg(), 32);
      GraphExport::Create(*theta->GetLoopVars()[0].output, "result");

      LoopUnrolling loopUnrolling(factor);
      loopUnrolling.Run(module, statisticsCollector);

      EXPECT_TRUE(evaluateUnrolledGuard(
          module.Rvsdg(),
          { { "init", 0 }, { "end", end }, { "step", step } },
          32))
          << "end=" << end << " step=" << step << " factor=" << factor << ": a batch of " << factor
          << " iterations fits in the " << end << " remaining values, so the guard must be taken";
    }
  }
}

/**
 * The guard of a loop with a runtime bound must admit exactly the batches that the original loop
 * would have executed, for every value of the induction variable.
 *
 * `while (n > 1) n -= 1` with an unroll factor of four: the batch is legal when all four decrements
 * stay inside the loop. Copy \c k executes when the predicate holds at the value the induction
 * variable has after \c k-1 copies, so the batch is legal exactly when the predicate holds at
 * `n-1`, `n-2` and `n-3` (copy 1 is unconditional). The expectation is derived copy by copy rather
 * than from a closed form, because every closed form in terms of `n - factor` wraps for small `n`
 * and that wrapping is precisely the defect this test guards against.
 */
TEST(LoopUnrollingTests, test_integer_loop_guard_matches_loop_semantics)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  LlvmRvsdgModule module(jlm::util::FilePath(""), "", "");
  auto & graph = module.Rvsdg();

  auto bitType = BitType::Create(32);
  auto n = &GraphImport::Create(graph, bitType, "n");

  auto theta = createDecrementingTheta(32, n);
  GraphExport::Create(*theta->GetLoopVars()[0].output, "result");

  const size_t factor = 4;
  LoopUnrolling loopUnrolling(factor);
  loopUnrolling.Run(module, statisticsCollector);

  const auto one = IntegerValueRepresentation(32, 1);

  for (const uint64_t value : { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 100u, 0xFFFFFFFFu })
  {
    /*
      Walk the copies the way the loop does: start at `n`, apply one decrement, and require the
      predicate `idv > 1` to hold for every copy after the first.
    */
    auto advanced = IntegerValueRepresentation(32, static_cast<int64_t>(value));
    bool expected = true;
    for (size_t copy = 1; copy < factor; ++copy)
    {
      advanced = advanced.sub(one);
      expected = expected && advanced.ugt(one) == '1';
    }

    EXPECT_EQ(evaluateUnrolledGuard(graph, { { "n", value } }, 32), expected)
        << "n=" << value << ": the guard must agree with the loop's own decision";
  }
}

/**
 * A loop with an equality predicate cannot be unrolled behind a batch guard.
 *
 * `for (i = 0; i != n; ++i)` advances by a step that need not divide the remaining range, so the
 * loop may pass through `n` in the *middle* of a batch and stop there. A single guard on the value
 * after `factor` copies cannot express that: it would admit batches that run past the exit value.
 *
 * The transformation must therefore leave such loops alone instead of emitting a guard.
 */
TEST(LoopUnrollingTests, test_integer_loop_equality_bound_is_not_unrolled)
{
  using namespace jlm::rvsdg;
  using namespace jlm::llvm;

  LlvmRvsdgModule module(jlm::util::FilePath(""), "", "");
  auto & graph = module.Rvsdg();

  auto bitType = BitType::Create(32);
  auto n = &GraphImport::Create(graph, bitType, "n");

  auto theta = ThetaNode::create(&graph.GetRootRegion());
  auto subregion = theta->subregion();
  auto zero = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0).output(0);
  auto idv = theta->AddLoopVar(zero);
  auto bound = theta->AddLoopVar(n);
  auto & one = IntegerConstantOperation::Create(*subregion, 32, 1);
  auto & arm = IntegerAddOperation::createNode(32, *idv.pre, *one.output(0));
  auto & cmp = IntegerNeOperation::createNode(32, *arm.output(0), *bound.pre);
  auto & matchNode = MatchOperation::CreateNode(*cmp.output(0), { { 1, 1 } }, 0, 2);
  idv.post->divert_to(arm.output(0));
  theta->set_predicate(matchNode.output(0));
  GraphExport::Create(*theta->GetLoopVars()[0].output, "result");

  LoopUnrolling loopUnrolling(4);
  loopUnrolling.Run(module, statisticsCollector);

  EXPECT_TRUE(findGammas(&graph.GetRootRegion()).empty())
      << "An equality-bounded loop must not be unrolled: a batch guard cannot express stopping "
         "in the middle of a batch";
  EXPECT_EQ(find_thetas(&graph.GetRootRegion()).size(), 1u)
      << "The original loop must be left in place";
}
