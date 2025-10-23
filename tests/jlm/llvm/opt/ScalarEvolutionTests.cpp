/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <test-registry.hpp>

#include <cassert>

static std::unordered_map<const jlm::rvsdg::Output *, std::unique_ptr<jlm::llvm::SCEV>>
RunScalarEvolution(const jlm::rvsdg::ThetaNode & thetaNode)
{
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::ScalarEvolution scalarEvolution;
  const auto inductionVariables = scalarEvolution.FindInductionVariables(thetaNode);
  auto chrecMap = scalarEvolution.CreateChainRecurrences(inductionVariables, thetaNode);
  return chrecMap;
}

static void
ConstantInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);

  lv1.post->divert_to(c1.output(0));

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 0); // Not a valid induction variable. No SCEV should have been created
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-ConstantInductionVariable",
    ConstantInductionVariable)

static void
SimpleInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c2 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c2.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 2);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());
  assert(chrecMap.find(lv2.pre) != chrecMap.end());

  // lv1 is a simple induction variable: {0,+,1}
  SCEVAddExpr scevAddExpr = SCEVAddExpr();
  scevAddExpr.SetLeftOperand(std::make_unique<SCEVConstant>(0));
  scevAddExpr.SetRightOperand(std::make_unique<SCEVConstant>(1));
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr));

  // lv2 is a constant 2
  SCEVConstant scevConstant = SCEVConstant(2);
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv2.pre], scevConstant));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-SimpleInductionVariable",
    SimpleInductionVariable)

static void
InductionVariableWithMultiplication()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c3 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c3.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode.output(0);

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv2.pre, c2.output(0) }, 32);
  const auto res2 = mulNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 1);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());
  assert(chrecMap.find(lv2.pre) == chrecMap.end()); // Ensure that lv2 is not counted as an IV

  // lv1 is a simple induction variable: {0,+,1}
  SCEVAddExpr scevAddExpr = SCEVAddExpr();
  scevAddExpr.SetLeftOperand(std::make_unique<SCEVConstant>(0));
  scevAddExpr.SetRightOperand(std::make_unique<SCEVConstant>(1));
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-InductionVariableWithMultiplication",
    InductionVariableWithMultiplication)

static void
RecursiveInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c4 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 4);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c4.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, c2.output(0) }, 32);
  const auto res2 = addNode2.output(0);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & addNode3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ res2, c3.output(0) }, 32);
  const auto res3 = addNode3.output(0);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);
  lv2.post->divert_to(res3);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 2);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());
  assert(chrecMap.find(lv2.pre) != chrecMap.end());

  // lv1 is a simple induction variable: {0,+,1}
  SCEVAddExpr scevAddExpr1 = SCEVAddExpr();
  scevAddExpr1.SetLeftOperand(std::make_unique<SCEVConstant>(0));
  scevAddExpr1.SetRightOperand(std::make_unique<SCEVConstant>(1));
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr1));

  // lv2 is a second-degree polynomial: {{4,+,2},+,3}
  // Inner: {4,+,2}
  SCEVAddExpr inner = SCEVAddExpr();
  inner.SetLeftOperand(std::make_unique<SCEVConstant>(4));
  inner.SetRightOperand(std::make_unique<SCEVConstant>(2));

  // Outer: {{4,+,2},+,3}
  SCEVAddExpr outer = SCEVAddExpr();
  outer.SetLeftOperand(inner.Clone());
  outer.SetRightOperand(std::make_unique<SCEVConstant>(3));

  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv2.pre], outer));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-RecursiveInductionVariable",
    RecursiveInductionVariable)

static void
PolynomialInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c2 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c2.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, res1 }, 32);
  const auto res2 = addNode2.output(0);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 2);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());
  assert(chrecMap.find(lv2.pre) != chrecMap.end());

  // lv1 is a simple induction variable: {0,+,1}
  SCEVAddExpr scevAddExpr1 = SCEVAddExpr();
  scevAddExpr1.SetLeftOperand(std::make_unique<SCEVConstant>(0));
  scevAddExpr1.SetRightOperand(std::make_unique<SCEVConstant>(1));
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr1));

  // lv2 is a second-degree polynomial: {2,+,{{0,+,1},+,1}}
  // Innermost: {0,+,1}
  SCEVAddExpr innermost = SCEVAddExpr();
  innermost.SetLeftOperand(std::make_unique<SCEVConstant>(0));
  innermost.SetRightOperand(std::make_unique<SCEVConstant>(1));

  // Middle: {{0,+,1},+,1}
  SCEVAddExpr middle = SCEVAddExpr();
  middle.SetLeftOperand(innermost.Clone());
  middle.SetRightOperand(std::make_unique<SCEVConstant>(1));

  // Outermost: {2,+,{{0,+,1},+,1}}
  SCEVAddExpr outermost = SCEVAddExpr();
  outermost.SetLeftOperand(std::make_unique<SCEVConstant>(2));
  outermost.SetRightOperand(middle.Clone());

  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv2.pre], outermost));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-PolynomialInductionVariable",
    PolynomialInductionVariable)

static void
ThirdDegreePolynomialInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c2 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c2.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);

  auto & addNode_1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, c1.output(0) }, 32);
  const auto result_1 = addNode_1.output(0);

  auto & addNode_2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, result_1 }, 32);
  const auto result_2 = addNode_2.output(0);

  auto & addNode_3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, result_2 }, 32);
  const auto result_3 = addNode_3.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result_1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result_3);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 2);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());
  assert(chrecMap.find(lv2.pre) != chrecMap.end());

  // lv2 is constant: 2
  SCEVConstant scevConstant = SCEVConstant(2);
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv2.pre], scevConstant));

  // lv1 is a third-degree polynomial: {0,+,{2,+,{2,+,1}}}
  // Innermost: {2,+,1}
  SCEVAddExpr innermost = SCEVAddExpr();
  innermost.SetLeftOperand(std::make_unique<SCEVConstant>(2));
  innermost.SetRightOperand(std::make_unique<SCEVConstant>(1));

  // Middle: {2,+,{2,+,1}}
  SCEVAddExpr middle = SCEVAddExpr();
  middle.SetLeftOperand(std::make_unique<SCEVConstant>(2));
  middle.SetRightOperand(innermost.Clone());

  // Outermost: {0,+,{2,+,{2,+,1}}}
  SCEVAddExpr outermost = SCEVAddExpr();
  outermost.SetLeftOperand(std::make_unique<SCEVConstant>(0));
  outermost.SetRightOperand(middle.Clone());

  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], outermost));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-ThirdDegreePolynomialInductionVariable",
    ThirdDegreePolynomialInductionVariable)

static void
SelfRecursiveInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));

  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, lv1.pre }, 32);
  const auto result = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 1);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());

  // Since it is self-recursive we will get {Unknown,+,Unknown}
  SCEVAddExpr scevAddExpr = SCEVAddExpr();
  scevAddExpr.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-SelfRecursiveInductionVariable",
    SelfRecursiveInductionVariable)

static void
DependentOnInvalidInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c1.output(0));

  auto & addNode_1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, lv1.pre }, 32);
  const auto result_1 = addNode_1.output(0);

  auto & addNode_2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, lv1.pre }, 32);
  const auto result_2 = addNode_2.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result_1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result_1);
  lv2.post->divert_to(result_2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 2);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());

  // lv1 is self-recursive: {Unknown,+,Unknown}
  SCEVAddExpr scevAddExpr1 = SCEVAddExpr();
  scevAddExpr1.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr1.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr1));

  // lv2 depends on invalid lv1: {0,+,Unknown}
  SCEVAddExpr scevAddExpr2 = SCEVAddExpr();
  scevAddExpr2.SetLeftOperand(std::make_unique<SCEVConstant>(0));
  scevAddExpr2.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv2.pre], scevAddExpr2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-DependentOnInvalidInductionVariable",
    DependentOnInvalidInductionVariable)

static void
MutuallyDependentInductionVariables()
{
  // Testing mutually dependent variables (A->B->A)
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0)); // A
  const auto lv2 = theta->AddLoopVar(c1.output(0)); // B

  auto & addNode_1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, lv2.pre }, 32);
  const auto result_1 = addNode_1.output(0);

  auto & addNode_2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, lv2.pre }, 32);
  const auto result_2 = addNode_2.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result_1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result_1);
  lv2.post->divert_to(result_2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 2);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());
  assert(chrecMap.find(lv2.pre) != chrecMap.end());

  // Both lv1 and lv2 have SCEV {Unknown,+,Unknown} since they are mutually dependent
  SCEVAddExpr scevAddExpr1 = SCEVAddExpr();
  scevAddExpr1.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr1.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr1));

  SCEVAddExpr scevAddExpr2 = SCEVAddExpr();
  scevAddExpr2.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr2.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv2.pre], scevAddExpr2));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-MutuallyDependentInductionVariables",
    MutuallyDependentInductionVariables)

static void
MultiLayeredMutuallyDependentInductionVariables()
{
  // Testing chains of mutually dependent IVs (A->B->C->D->A)
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1);
  const auto & c2 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);
  const auto & c3 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0)); // A
  const auto lv2 = theta->AddLoopVar(c1.output(0)); // B
  const auto lv3 = theta->AddLoopVar(c2.output(0)); // C
  const auto lv4 = theta->AddLoopVar(c3.output(0)); // D

  auto & addNode_1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, lv2.pre }, 32);
  const auto result_1 = addNode_1.output(0);
  auto & addNode_2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, lv3.pre }, 32);
  const auto result_2 = addNode_2.output(0);
  auto & addNode_3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3.pre, lv4.pre }, 32);
  const auto result_3 = addNode_3.output(0);
  auto & addNode_4 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv4.pre, lv1.pre }, 32);
  const auto result_4 = addNode_4.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result_1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result_1);
  lv2.post->divert_to(result_2);
  lv3.post->divert_to(result_3);
  lv4.post->divert_to(result_4);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  assert(chrecMap.size() == 4);
  assert(chrecMap.find(lv1.pre) != chrecMap.end());
  assert(chrecMap.find(lv2.pre) != chrecMap.end());
  assert(chrecMap.find(lv3.pre) != chrecMap.end());
  assert(chrecMap.find(lv4.pre) != chrecMap.end());

  // All four IVs form a circular dependency chain: {Unknown,+,Unknown}
  SCEVAddExpr scevAddExpr1 = SCEVAddExpr();
  scevAddExpr1.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr1.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv1.pre], scevAddExpr1));

  SCEVAddExpr scevAddExpr2 = SCEVAddExpr();
  scevAddExpr2.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr2.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv2.pre], scevAddExpr2));

  SCEVAddExpr scevAddExpr3 = SCEVAddExpr();
  scevAddExpr3.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr3.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv3.pre], scevAddExpr3));

  SCEVAddExpr scevAddExpr4 = SCEVAddExpr();
  scevAddExpr4.SetLeftOperand(std::make_unique<SCEVUnknown>());
  scevAddExpr4.SetRightOperand(std::make_unique<SCEVUnknown>());
  assert(ScalarEvolution::StructurallyEqual(*chrecMap[lv4.pre], scevAddExpr4));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-MultiLayeredMutuallyDependentInductionVariables",
    MultiLayeredMutuallyDependentInductionVariables)
