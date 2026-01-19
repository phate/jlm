/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <gtest/gtest.h>

static std::
    unordered_map<const jlm::rvsdg::Output *, std::unique_ptr<jlm::llvm::SCEVChainRecurrence>>
    RunScalarEvolution(const jlm::rvsdg::ThetaNode & thetaNode)
{
  jlm::llvm::ScalarEvolution scalarEvolution;
  auto chrecMap = scalarEvolution.PerformSCEVAnalysis(thetaNode);
  return chrecMap;
}

TEST(ScalarEvolutionTests, ConstantInductionVariable)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);

  lv1.post->divert_to(c1.output(0));

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  EXPECT_EQ(chrecMap.size(), 1u);

  // Since lv1 is not a valid induction variable, it's chain recurrence should be
  // {Unknown}<THETA>
  auto testChrec = SCEVChainRecurrence(*theta);
  testChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv1.pre)));
}

TEST(ScalarEvolutionTests, SimpleInductionVariable)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a constant with the recurrence {2}<THETA>
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(2));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, RecursiveInductionVariable)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a recursive induction variable, which should be folded to {4,+,5}<THETA>
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(4));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(5));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, PolynomialInductionVariable)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a (second degree) polynomial induction variable with three operands,
  // Recurrence: {2,+,1,+,1}<THETA>
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(2));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, ThirdDegreePolynomialInductionVariable)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c2 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);
  const auto & c3 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3);
  const auto & c4 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 4);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c2.output(0));
  const auto lv2 = theta->AddLoopVar(c3.output(0));
  const auto lv3 = theta->AddLoopVar(c4.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);

  auto & addNode_1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result_1 = addNode_1.output(0);

  auto & addNode_2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, lv2.pre }, 32);
  const auto result_2 = addNode_2.output(0);

  auto & addNode_3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3.pre, lv2.pre }, 32);
  const auto result_3 = addNode_3.output(0);

  lv1.post->divert_to(result_1);
  lv2.post->divert_to(result_2);
  lv3.post->divert_to(result_3);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  EXPECT_EQ(chrecMap.size(), 3u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {2,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(2));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a second degree polynomial induction variable with three operands, {3,+,2,+,1}<THETA>
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(3));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(2));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // lv2 is a third degree polynomial induction variable with three operands,
  // Recurrence: {4,+,3,+,2,+,1}<THETA>
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(std::make_unique<SCEVConstant>(4));
  lv3TestChrec.AddOperand(std::make_unique<SCEVConstant>(3));
  lv3TestChrec.AddOperand(std::make_unique<SCEVConstant>(2));
  lv3TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));
}

TEST(ScalarEvolutionTests, InductionVariableWithMultiplication)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c2 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c2.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  auto res1 = addNode1.output(0);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ res1, c3.output(0) }, 32);

  auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, mulNode.output(0) }, 32);
  auto res2 = addNode2.output(0);

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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is incremented with 3 * {1,+,1} (lv1 + 1) each iteration. With the starting value of 2,
  // this should give us the recurrence {2,+,3,+,3}<THETA>
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(2));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(3));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(3));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, InvalidInductionVariableWithMultiplication)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c1.output(0));

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, c2.output(0) }, 32);
  const auto res1 = mulNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  EXPECT_EQ(chrecMap.size(), 1u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  // lv1 is not an induction variable because of illegal mult operation
  // (results in quadratic recurrence). It should therefore be modeled as {Unknown}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));
}

TEST(ScalarEvolutionTests, PolynomialInductionVariableWithMultiplication)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c3 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c3.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  auto res1 = addNode1.output(0);

  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, lv1.pre }, 32);

  auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, mulNode.output(0) }, 32);
  auto res2 = addNode2.output(0);

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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // {0,+,1} * {0,+,1} folded together should be {0,+,1,+,2}, so adding this to lv2, should give us
  // {3,+,0,+,1,+,2}<THETA> as the recurrence for lv2
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(3));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(2));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, InvalidPolynomialInductionVariableWithMultiplication)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c3 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c3.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  auto res1 = addNode1.output(0);

  auto & mulNode = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv2.pre, lv1.pre }, 32);
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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv1 is not an induction variable because of illegal mult operation
  // (results in quadratic recurrence). It should therefore be modeled as {Unknown}<THETA>
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, InductionVariableWithSubtraction)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c10 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 10);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c10.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);

  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto & res1 = subNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSgtOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  EXPECT_EQ(chrecMap.size(), 1u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  // lv1 is a simple negative induction variable with the recurrence {10,+,-1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(10));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(-1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));
}

TEST(ScalarEvolutionTests, PolynomialInductionVariableWithSubtraction)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c3 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c3.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode.output(0);

  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ lv2.pre, res1 }, 32);
  const auto res2 = subNode.output(0);

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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv1TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is decremented with {1,+,1} every iteration (lv1 + 1}, and has a start value of 3, this
  // results in the recurrence {3,+,-1,+,-1}<THETA>
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(3));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(-1));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(-1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, InductionVariablesWithNonConstantInitialValues)
{
  // This test checks the functionality of the folding rules for variables that have start values
  // that are not constants. These will get a SCEVInit node instead, which cannot be folded like a
  // constant.
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  jlm::rvsdg::Graph & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, intType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, intType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, intType, "z");
  auto w = &jlm::rvsdg::GraphImport::Create(graph, intType, "w");
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ intType }, { intType }),
          "f",
          Linkage::externalLinkage));

  // We wrap the theta in a lambda node to get init nodes in the SCEV tree

  auto cv1 = lambda->AddContextVar(*x).inner;
  auto cv2 = lambda->AddContextVar(*y).inner;
  auto cv3 = lambda->AddContextVar(*z).inner;
  auto cv4 = lambda->AddContextVar(*w).inner;

  const auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto lv1 = theta->AddLoopVar(cv1);
  auto lv2 = theta->AddLoopVar(cv2);
  auto lv3 = theta->AddLoopVar(cv3);
  auto lv4 = theta->AddLoopVar(cv4);

  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, lv1.pre }, 32);
  const auto res1 = addNode1.output(0);

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3.pre, res1 }, 32);
  const auto res2 = addNode2.output(0);

  auto & addNode3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ res2, res1 }, 32);
  const auto res3 = addNode3.output(0);

  auto & addNode4 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv4.pre, res3 }, 32);
  const auto res4 = addNode4.output(0);

  lv2.post->divert_to(res1);
  lv3.post->divert_to(res2);
  lv4.post->divert_to(res4);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  EXPECT_EQ(chrecMap.size(), 4u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv4.pre), chrecMap.end());

  // lv1 is a trivial (constant) induction variable.
  // Recurrence: {Init(a0)}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a general induction variable which is incremented by the value of lv1 for each
  // iteration.
  // Recurrence: {Init(a1),+,Init(a0)}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv2.pre));
  lv2TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // Tests that two init nodes folded together creates an NAryAdd expression
  // Recurrence: {Init(a2),+,(Init(a1) + Init(a0)),+,Init(a0)}
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv3.pre));
  lv3TestChrec.AddOperand(std::make_unique<SCEVNAryAddExpr>(
      std::make_unique<SCEVInit>(*lv2.pre),
      std::make_unique<SCEVInit>(*lv1.pre)));
  lv3TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));

  // Tests that when two NAryAdd expressions are folded together, the operands of the RHS add is
  // added to the LHS add
  // Recurrence: {Init(a3),+,(Init(a1) + Init(a0) + Init(a2) + Init(a1) + Init(a0)),+,(Init(a1) +
  // Init(a0) + Init(a0) + Init(a0)),+,Init(a0)}
  auto lv4TestChrec = SCEVChainRecurrence(*theta);
  lv4TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv4.pre));
  lv4TestChrec.AddOperand(std::make_unique<SCEVNAryAddExpr>(
      std::make_unique<SCEVInit>(*lv2.pre),
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVInit>(*lv3.pre),
      std::make_unique<SCEVInit>(*lv2.pre),
      std::make_unique<SCEVInit>(*lv1.pre)));
  lv4TestChrec.AddOperand(std::make_unique<SCEVNAryAddExpr>(
      std::make_unique<SCEVInit>(*lv2.pre),
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVInit>(*lv1.pre)));
  lv4TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv4TestChrec, *chrecMap.at(lv4.pre)));
}

TEST(ScalarEvolutionTests, InductionVariablesWithNonConstantInitialValuesAndMultiplication)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  jlm::rvsdg::Graph & graph = rvsdgModule.Rvsdg();

  auto x = &jlm::rvsdg::GraphImport::Create(graph, intType, "x");
  auto y = &jlm::rvsdg::GraphImport::Create(graph, intType, "y");
  auto z = &jlm::rvsdg::GraphImport::Create(graph, intType, "z");
  auto w = &jlm::rvsdg::GraphImport::Create(graph, intType, "w");
  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create({ intType }, { intType }),
          "f",
          Linkage::externalLinkage));

  auto cv1 = lambda->AddContextVar(*x).inner;
  auto cv2 = lambda->AddContextVar(*y).inner;
  auto cv3 = lambda->AddContextVar(*z).inner;
  auto cv4 = lambda->AddContextVar(*w).inner;

  const auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto lv1 = theta->AddLoopVar(cv1);
  auto lv2 = theta->AddLoopVar(cv2);
  auto lv3 = theta->AddLoopVar(cv3);
  auto lv4 = theta->AddLoopVar(cv4);

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, c1.output(0) }, 32);
  const auto res1 = addNode1.output(0);

  auto & mulNode1 = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1.pre, lv2.pre }, 32);
  const auto res2 = mulNode1.output(0);

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3.pre, res2 }, 32);
  const auto res3 = addNode2.output(0);

  auto & mulNode2 = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ res2, res2 }, 32);

  auto & addNode3 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv4.pre, mulNode2.output(0) }, 32);
  const auto res4 = addNode3.output(0);

  lv2.post->divert_to(res1);
  lv3.post->divert_to(res3);
  lv4.post->divert_to(res4);

  jlm::rvsdg::view(graph, stdout);

  // Act
  auto chrecMap = RunScalarEvolution(*theta);

  // Assert
  EXPECT_EQ(chrecMap.size(), 4u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv4.pre), chrecMap.end());

  // lv1 is a trivial (constant) induction variable.
  // Recurrence: {Init(a0)}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a general induction variable which is incremented by 1 each iteration.
  // Recurrence: {Init(a1),+,1}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv2.pre));
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // Tests multiplying two init nodes together creates an n-ary mult expression.
  // Recurrence: {Init(a2),+,(Init(a0) * Init(a1)),+,Init(a0)}
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv3.pre));
  lv3TestChrec.AddOperand(std::make_unique<SCEVNAryMulExpr>(
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVInit>(*lv2.pre)));
  lv3TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv1.pre));

  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));

  // Tests that when two n-ary mult expressions are folded together, the operands of the RHS mult is
  // added to the LHS mult
  // Recurrence: {Init(a3),+,(Init(a0) * Init(a1) * Init(a0) * Init(a1)),+,((Init(a0) * Init(a1) *
  // Init(a0)) + (Init(a0) * Init(a1) * Init(a0)) + (Init(a0) * Init(a0))),+,(Init(a0) * Init(a0) *
  // 2)}<THETA>
  auto lv4TestChrec = SCEVChainRecurrence(*theta);
  lv4TestChrec.AddOperand(std::make_unique<SCEVInit>(*lv4.pre));
  lv4TestChrec.AddOperand(std::make_unique<SCEVNAryMulExpr>(
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVInit>(*lv2.pre),
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVInit>(*lv2.pre)));
  lv4TestChrec.AddOperand(std::make_unique<SCEVNAryAddExpr>(
      std::make_unique<SCEVNAryMulExpr>(
          std::make_unique<SCEVInit>(*lv1.pre),
          std::make_unique<SCEVInit>(*lv2.pre),
          std::make_unique<SCEVInit>(*lv1.pre)),
      std::make_unique<SCEVNAryMulExpr>(
          std::make_unique<SCEVInit>(*lv1.pre),
          std::make_unique<SCEVInit>(*lv2.pre),
          std::make_unique<SCEVInit>(*lv1.pre)),
      std::make_unique<SCEVNAryMulExpr>(
          std::make_unique<SCEVInit>(*lv1.pre),
          std::make_unique<SCEVInit>(*lv1.pre))));
  lv4TestChrec.AddOperand(std::make_unique<SCEVNAryMulExpr>(
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVInit>(*lv1.pre),
      std::make_unique<SCEVConstant>(2)));

  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv4TestChrec, *chrecMap.at(lv4.pre)));
}

TEST(ScalarEvolutionTests, SelfRecursiveInductionVariable)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(chrecMap.size(), 1u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  // lv1 is not an induction variable because of self dependency. Should be {Unknown}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));
}

TEST(ScalarEvolutionTests, DependentOnInvalidInductionVariable)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is not an induction variable because of self dependency. Should be {Unknown}<THETA>
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 has the recurrence {0,+,Unknown}<THETA> due to the dependency of lv1
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(std::make_unique<SCEVConstant>(0));
  lv2TestChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, MutuallyDependentInductionVariables)
{
  // Testing mutually dependent variables (A->B->A)
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(chrecMap.size(), 2u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // Both lv1 and lv2 should be {Unknown}<THETA> due to mutual dependency (A->B->A)
  auto testChrec = SCEVChainRecurrence(*theta);
  testChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv1.pre)));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, MultiLayeredMutuallyDependentInductionVariables)
{
  // Testing chains of mutually dependent IVs (A->B->C->D->A)
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
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
  EXPECT_EQ(chrecMap.size(), 4u);
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv4.pre), chrecMap.end());

  // All variables should be {Unknown}<THETA> due to mutual dependency chain (A->B->C->D->A)
  auto testChrec = SCEVChainRecurrence(*theta);
  testChrec.AddOperand(std::make_unique<SCEVUnknown>());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv1.pre)));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv2.pre)));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv1.pre)));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv2.pre)));
}
