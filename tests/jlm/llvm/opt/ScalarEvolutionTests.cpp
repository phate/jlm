/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/view.hpp>

#include <gtest/gtest.h>

static std::pair<
    std::unordered_map<const jlm::rvsdg::Output *, std::unique_ptr<jlm::llvm::SCEVChainRecurrence>>,
    std::unordered_map<const jlm::rvsdg::ThetaNode *, size_t>>
RunScalarEvolution(jlm::rvsdg::RvsdgModule & rvsdgModule)
{
  jlm::llvm::ScalarEvolution scalarEvolution;
  jlm::util::StatisticsCollector statisticsCollector;
  scalarEvolution.Run(rvsdgModule, statisticsCollector);
  return { scalarEvolution.GetChrecMap(), scalarEvolution.GetTripCountMap() };
}

TEST(ScalarEvolutionTests, NonEvolvingVariable)
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

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ lv1.pre, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert

  // Since lv1 is a variable which does not depend on the previous value (no evolution). There
  // should be no computed chrec for it
  EXPECT_EQ(chrecMap.find(lv1.pre), chrecMap.end());
}

TEST(ScalarEvolutionTests, InductionVariable)
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a constant with the recurrence {2}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a recursive induction variable, which should be folded to {4,+,5}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(4));
  lv2TestChrec.AddOperand(SCEVConstant::Create(5));
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a (second degree) polynomial induction variable with three operands,
  // Recurrence: {2,+,1,+,1}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
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

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result_1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result_1);
  lv2.post->divert_to(result_2);
  lv3.post->divert_to(result_3);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {2,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(2));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a second degree polynomial induction variable with three operands, {3,+,2,+,1}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(3));
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // lv2 is a third degree polynomial induction variable with three operands,
  // Recurrence: {4,+,3,+,2,+,1}
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(SCEVConstant::Create(4));
  lv3TestChrec.AddOperand(SCEVConstant::Create(3));
  lv3TestChrec.AddOperand(SCEVConstant::Create(2));
  lv3TestChrec.AddOperand(SCEVConstant::Create(1));
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is incremented with 3 * {1,+,1} (lv1 + 1) each iteration. With the starting value of 2,
  // this should give us the recurrence {2,+,3,+,3}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
  lv2TestChrec.AddOperand(SCEVConstant::Create(3));
  lv2TestChrec.AddOperand(SCEVConstant::Create(3));
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  // lv1 is not an induction variable because of illegal mult operation
  // (results in quadratic recurrence). It should therefore be modeled as {Unknown}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVUnknown::Create());
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // {0,+,1} * {0,+,1} folded together should be {0,+,1,+,2}, so adding this to lv2, should give us
  // {3,+,0,+,1,+,2} as the recurrence for lv2
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(3));
  lv2TestChrec.AddOperand(SCEVConstant::Create(0));
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
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

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);
  auto & mulNode =
      jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv2.pre, addNode2.output(0) }, 32);
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv1 is not an induction variable because of illegal mult operation
  // (results in quadratic recurrence). It should therefore be modeled as {Unknown}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVUnknown::Create());
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  // lv1 is a simple negative induction variable with the recurrence {10,+,-1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(10));
  lv1TestChrec.AddOperand(SCEVConstant::Create(-1));
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is decremented with {1,+,1} every iteration (lv1 + 1}, and has a start value of 3, this
  // results in the recurrence {3,+,-1,+,-1}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(3));
  lv2TestChrec.AddOperand(SCEVConstant::Create(-1));
  lv2TestChrec.AddOperand(SCEVConstant::Create(-1));
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

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv2.post->divert_to(res1);
  lv3.post->divert_to(res2);
  lv4.post->divert_to(res4);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv4.pre), chrecMap.end());

  // lv1 is a trivial (constant) induction variable.
  // Recurrence: {Init(a0)}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVInit::Create(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a general induction variable which is incremented by the value of lv1 for each
  // iteration.
  // Recurrence: {Init(a1),+,Init(a0)}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVInit::Create(*lv2.pre));
  lv2TestChrec.AddOperand(SCEVInit::Create(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // Tests that two init nodes folded together creates an NAryAdd expression
  // Recurrence: {Init(a2),+,(Init(a1) + Init(a0)),+,Init(a0)}
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(SCEVInit::Create(*lv3.pre));
  lv3TestChrec.AddOperand(
      SCEVNAryAddExpr::Create(SCEVInit::Create(*lv2.pre), SCEVInit::Create(*lv1.pre)));
  lv3TestChrec.AddOperand(SCEVInit::Create(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));

  // Tests that when two NAryAdd expressions are folded together, the operands of the RHS add is
  // added to the LHS add
  // Recurrence: {Init(a3),+,(Init(a1) + Init(a0) + Init(a2) + Init(a1) + Init(a0)),+,(Init(a1) +
  // Init(a0) + Init(a0) + Init(a0)),+,Init(a0)}
  auto lv4TestChrec = SCEVChainRecurrence(*theta);
  lv4TestChrec.AddOperand(SCEVInit::Create(*lv4.pre));
  lv4TestChrec.AddOperand(SCEVNAryAddExpr::Create(
      SCEVInit::Create(*lv2.pre),
      SCEVInit::Create(*lv1.pre),
      SCEVInit::Create(*lv3.pre),
      SCEVInit::Create(*lv2.pre),
      SCEVInit::Create(*lv1.pre)));
  lv4TestChrec.AddOperand(SCEVNAryAddExpr::Create(
      SCEVInit::Create(*lv2.pre),
      SCEVInit::Create(*lv1.pre),
      SCEVInit::Create(*lv1.pre),
      SCEVInit::Create(*lv1.pre)));
  lv4TestChrec.AddOperand(SCEVInit::Create(*lv1.pre));
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

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv2.post->divert_to(res1);
  lv3.post->divert_to(res3);
  lv4.post->divert_to(res4);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv4.pre), chrecMap.end());

  // lv1 is a trivial (constant) induction variable.
  // Recurrence: {Init(a0)}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVInit::Create(*lv1.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a general induction variable which is incremented by 1 each iteration.
  // Recurrence: {Init(a1),+,1}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVInit::Create(*lv2.pre));
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // Tests multiplying two init nodes together creates an n-ary mult expression.
  // Recurrence: {Init(a2),+,(Init(a0) * Init(a1)),+,Init(a0)}
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(SCEVInit::Create(*lv3.pre));
  lv3TestChrec.AddOperand(
      SCEVNAryMulExpr::Create(SCEVInit::Create(*lv1.pre), SCEVInit::Create(*lv2.pre)));
  lv3TestChrec.AddOperand(SCEVInit::Create(*lv1.pre));

  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));

  // Tests that when two n-ary mult expressions are folded together, the operands of the RHS mult is
  // added to the LHS mult
  // Recurrence: {Init(a3),+,(Init(a0) * Init(a1) * Init(a0) * Init(a1)),+,((Init(a0) * Init(a1) *
  // Init(a0)) + (Init(a0) * Init(a1) * Init(a0)) + (Init(a0) * Init(a0))),+,(Init(a0) * Init(a0) *
  // 2)}
  auto lv4TestChrec = SCEVChainRecurrence(*theta);
  lv4TestChrec.AddOperand(SCEVInit::Create(*lv4.pre));
  lv4TestChrec.AddOperand(SCEVNAryMulExpr::Create(
      SCEVInit::Create(*lv1.pre),
      SCEVInit::Create(*lv2.pre),
      SCEVInit::Create(*lv1.pre),
      SCEVInit::Create(*lv2.pre)));
  lv4TestChrec.AddOperand(SCEVNAryAddExpr::Create(
      SCEVNAryMulExpr::Create(
          SCEVInit::Create(*lv1.pre),
          SCEVInit::Create(*lv2.pre),
          SCEVInit::Create(*lv1.pre)),
      SCEVNAryMulExpr::Create(
          SCEVInit::Create(*lv1.pre),
          SCEVInit::Create(*lv2.pre),
          SCEVInit::Create(*lv1.pre)),
      SCEVNAryMulExpr::Create(SCEVInit::Create(*lv1.pre), SCEVInit::Create(*lv1.pre))));
  lv4TestChrec.AddOperand(SCEVNAryMulExpr::Create(
      SCEVInit::Create(*lv1.pre),
      SCEVInit::Create(*lv1.pre),
      SCEVConstant::Create(2)));

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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  // lv1 is not an induction variable because of self dependency. Should be {Unknown}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVUnknown::Create());
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // lv1 is not an induction variable because of self dependency. Should be {Unknown}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVUnknown::Create());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 has the recurrence {0,+,Unknown} due to the dependency of lv1
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(0));
  lv2TestChrec.AddOperand(SCEVUnknown::Create());
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());

  // Both lv1 and lv2 should be {Unknown} due to mutual dependency (A->B->A)
  auto testChrec = SCEVChainRecurrence(*theta);
  testChrec.AddOperand(SCEVUnknown::Create());
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
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv4.pre), chrecMap.end());

  // All variables should be {Unknown} due to mutual dependency chain (A->B->C->D->A)
  auto testChrec = SCEVChainRecurrence(*theta);
  testChrec.AddOperand(SCEVUnknown::Create());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv1.pre)));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv2.pre)));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv1.pre)));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(testChrec, *chrecMap.at(lv2.pre)));
}

TEST(ScalarEvolutionTests, InductionVariablesInNestedLoops)
{
  // Tests "stitching" of induction variables in outer loops
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto theta1 = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta1->AddLoopVar(c0.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode1.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 5);
  auto & sltNode1 = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult1 =
      jlm::rvsdg::MatchOperation::Create(*sltNode1.output(0), { { 1, 1 } }, 0, 2);

  const auto theta2 = jlm::rvsdg::ThetaNode::create(theta1->subregion());
  const auto lv2 = theta2->AddLoopVar(c1.output(0));
  const auto lv3 = theta2->AddLoopVar(lv1.pre);

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, lv3.pre }, 32);
  const auto & res2 = addNode2.output(0);

  const auto & c10 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 10);
  auto & sltNode2 = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res2, c10.output(0) }, 32);

  const auto matchResult2 =
      jlm::rvsdg::MatchOperation::Create(*sltNode2.output(0), { { 1, 1 } }, 0, 2);

  theta1->set_predicate(matchResult1);
  lv1.post->divert_to(res1);

  theta2->set_predicate(matchResult2);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<1>
  auto lv1TestChrec = SCEVChainRecurrence(*theta1);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a nested induction variable, which is incremented by the value of lv1 for each iteration
  auto lv2TestChrec = SCEVChainRecurrence(*theta2);
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
  lv2TestChrec.AddOperand(lv1TestChrec.Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // lv3 has the same value as lv1, but for the inner loop
  auto lv3TestChrec = SCEVChainRecurrence(*theta2);
  lv3TestChrec.AddOperand(lv1TestChrec.Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));
}

TEST(ScalarEvolutionTests, InductionVariablesInNestedLoopsWithFolding)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0_1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c2_1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);
  const auto & c3_1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 3);

  const auto theta1 = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1_1 = theta1->AddLoopVar(c0_1.output(0));
  const auto lv2_1 = theta1->AddLoopVar(c2_1.output(0));
  const auto lv3_1 = theta1->AddLoopVar(c3_1.output(0));

  const auto & c1_2 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 1);
  const auto & c2_2 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 2);
  const auto & c3_2 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 3);

  auto & addNode1 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1_1.pre, c1_2.output(0) }, 32);
  auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2_1.pre, c2_2.output(0) }, 32);
  auto & addNode3 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3_1.pre, c3_2.output(0) }, 32);
  const auto res1 = addNode1.output(0);
  const auto res2 = addNode2.output(0);
  const auto res3 = addNode3.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 5);
  auto & sltNode1 = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult1 =
      jlm::rvsdg::MatchOperation::Create(*sltNode1.output(0), { { 1, 1 } }, 0, 2);

  const auto theta2 = jlm::rvsdg::ThetaNode::create(theta1->subregion());
  const auto lv1_2 = theta2->AddLoopVar(res1);
  const auto lv2_2 = theta2->AddLoopVar(res2);
  const auto lv3_2 = theta2->AddLoopVar(res3);
  const auto lv4 = theta2->AddLoopVar(c1_2.output(0));
  const auto lv5 = theta2->AddLoopVar(c1_2.output(0));

  auto & addNode4 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1_2.pre, lv2_2.pre }, 32);
  auto & addNode5 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ addNode4.output(0), lv3_2.pre }, 32);
  auto & addNode6 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv4.pre, addNode5.output(0) }, 32);
  const auto & res4 = addNode6.output(0);

  auto & mulNode1 = jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv1_2.pre, lv2_2.pre }, 32);
  auto & addNode7 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv5.pre, mulNode1.output(0) }, 32);
  const auto & res5 = addNode7.output(0);

  const auto & c10 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 10);
  auto & sltNode2 = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res4, c10.output(0) }, 32);

  const auto matchResult2 =
      jlm::rvsdg::MatchOperation::Create(*sltNode2.output(0), { { 1, 1 } }, 0, 2);

  // Outer loop
  theta1->set_predicate(matchResult1);
  lv1_1.post->divert_to(res1);
  lv2_1.post->divert_to(res2);
  lv3_1.post->divert_to(res3);

  // Inner loop
  theta2->set_predicate(matchResult2);
  lv4.post->divert_to(res4);
  lv5.post->divert_to(res5);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1_1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2_1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3_1.pre), chrecMap.end());

  EXPECT_NE(chrecMap.find(lv1_2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2_2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3_2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv4.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv5.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<1>
  auto lv1TestChrec = SCEVChainRecurrence(*theta1);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1_1.pre)));

  // lv2 is a simple induction variable with the recurrence {2,+,2}<1>
  auto lv2TestChrec = SCEVChainRecurrence(*theta1);
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2_1.pre)));

  // lv3 is a simple induction variable with the recurrence {3,+,3}<1>
  auto lv3TestChrec = SCEVChainRecurrence(*theta1);
  lv3TestChrec.AddOperand(SCEVConstant::Create(3));
  lv3TestChrec.AddOperand(SCEVConstant::Create(3));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3_1.pre)));

  // lv1_2 is lv1_1 incremented by 1 but in the inner loop
  auto lv1_2TestChrec = SCEVChainRecurrence(*theta2);
  const auto lv1_2InnerChrec = SCEVChainRecurrence::Create(*theta1);
  lv1_2InnerChrec->AddOperand(SCEVConstant::Create(1));
  lv1_2InnerChrec->AddOperand(SCEVConstant::Create(1));
  lv1_2TestChrec.AddOperand(lv1_2InnerChrec->Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1_2TestChrec, *chrecMap.at(lv1_2.pre)));

  // lv2_2 is lv2_1 incremented by 2 but in the inner loop
  auto lv2_2TestChrec = SCEVChainRecurrence(*theta2);
  const auto lv2_2InnerChrec = SCEVChainRecurrence::Create(*theta1);
  lv2_2InnerChrec->AddOperand(SCEVConstant::Create(4));
  lv2_2InnerChrec->AddOperand(SCEVConstant::Create(2));
  lv2_2TestChrec.AddOperand(lv2_2InnerChrec->Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2_2TestChrec, *chrecMap.at(lv2_2.pre)));

  // lv3_2 is lv3_1 incremented by 3 but in the inner loop
  auto lv3_2TestChrec = SCEVChainRecurrence(*theta2);
  const auto lv3_2InnerChrec = SCEVChainRecurrence::Create(*theta1);
  lv3_2InnerChrec->AddOperand(SCEVConstant::Create(6));
  lv3_2InnerChrec->AddOperand(SCEVConstant::Create(3));
  lv3_2TestChrec.AddOperand(lv3_2InnerChrec->Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3_2TestChrec, *chrecMap.at(lv3_2.pre)));

  // lv4 is in the inner loop and has a start value of 1 which is incremented by the result of
  // folding ({1,+,1}<1> + {4,+,2}<1> + {6,+,3}<1>) = {11,+,6}<1>. Recurrence: {1,+,{11,+,6}<1>}<2>
  auto lv4TestChrec = SCEVChainRecurrence(*theta2);
  lv4TestChrec.AddOperand(SCEVConstant::Create(1));
  const auto lv4InnerChrec = SCEVChainRecurrence::Create(*theta1);
  lv4InnerChrec->AddOperand(SCEVConstant::Create(11));
  lv4InnerChrec->AddOperand(SCEVConstant::Create(6));
  lv4TestChrec.AddOperand(lv4InnerChrec->Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv4TestChrec, *chrecMap.at(lv4.pre)));

  // lv5 is in the inner loop and has a start value of 1 which is incremented by the result of
  // folding ({1,+,1}<1> * {4,+,2}<1>) = {4,+,8,+,4}<1>. Recurrence: {1,+,{4,+,8,+,4}<1>}<2>
  auto lv5TestChrec = SCEVChainRecurrence(*theta2);
  lv5TestChrec.AddOperand(SCEVConstant::Create(1));
  const auto lv5InnerChrec = SCEVChainRecurrence::Create(*theta1);
  lv5InnerChrec->AddOperand(SCEVConstant::Create(4));
  lv5InnerChrec->AddOperand(SCEVConstant::Create(8));
  lv5InnerChrec->AddOperand(SCEVConstant::Create(4));
  lv5TestChrec.AddOperand(lv5InnerChrec->Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv5TestChrec, *chrecMap.at(lv5.pre)));
}

TEST(ScalarEvolutionTests, InductionVariablesInSisterLoops)
{
  // Tests "stitching" of induction variables in other loops that are on the same level, AKA "sister
  // loops"
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto theta1 = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta1->AddLoopVar(c0.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 1);
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode1.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta1->subregion(), 32, 5);
  auto & sltNode1 = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult1 =
      jlm::rvsdg::MatchOperation::Create(*sltNode1.output(0), { { 1, 1 } }, 0, 2);

  const auto & c2 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 2);
  const auto theta2 = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv2 = theta2->AddLoopVar(c2.output(0));
  const auto lv3 = theta2->AddLoopVar(theta1->output(0));

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, lv3.pre }, 32);
  const auto & res2 = addNode2.output(0);

  const auto & c10 = IntegerConstantOperation::Create(*theta2->subregion(), 32, 10);
  auto & sltNode2 = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res2, c10.output(0) }, 32);

  const auto matchResult2 =
      jlm::rvsdg::MatchOperation::Create(*sltNode2.output(0), { { 1, 1 } }, 0, 2);

  theta1->set_predicate(matchResult1);
  lv1.post->divert_to(res1);

  theta2->set_predicate(matchResult2);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());

  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}<1>
  auto lv1TestChrec = SCEVChainRecurrence(*theta1);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is a nested induction variable, which is incremented by the post iteration value of lv1
  // ({1,+,1}<1>) for each iteration
  auto lv2TestChrec = SCEVChainRecurrence(*theta2);
  lv2TestChrec.AddOperand(SCEVConstant::Create(2));
  auto lv2InnerChrec = SCEVChainRecurrence(*theta1);
  lv2InnerChrec.AddOperand(SCEVConstant::Create(1));
  lv2InnerChrec.AddOperand(SCEVConstant::Create(1));
  lv2TestChrec.AddOperand(lv2InnerChrec.Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // lv3 has the same value as the post iteration value of lv1, but wrapped in the inner loop
  auto lv3TestChrec = SCEVChainRecurrence(*theta2);
  lv3TestChrec.AddOperand(lv2InnerChrec.Clone());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));
}

TEST(ScalarEvolutionTests, ComputeRecurrenceForArrayGEP)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto intArrayType = ArrayType::Create(intType, 5);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { pointerType, memoryStateType },
              { intType, memoryStateType }),
          "f",
          Linkage::externalLinkage));

  const auto & delta = jlm::rvsdg::DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(intArrayType, "", Linkage::externalLinkage, "", false));

  const auto & arrayC1 = IntegerConstantOperation::Create(*delta->subregion(), 32, 1);
  const auto & arrayC2 = IntegerConstantOperation::Create(*delta->subregion(), 32, 2);
  const auto & arrayC3 = IntegerConstantOperation::Create(*delta->subregion(), 32, 3);
  const auto & arrayC4 = IntegerConstantOperation::Create(*delta->subregion(), 32, 4);
  const auto & arrayC5 = IntegerConstantOperation::Create(*delta->subregion(), 32, 5);

  const auto & constantArray = ConstantDataArray::Create({ arrayC1.output(0),
                                                           arrayC2.output(0),
                                                           arrayC3.output(0),
                                                           arrayC4.output(0),
                                                           arrayC5.output(0) });
  delta->finalize(constantArray);

  const auto cv1 = lambda->AddContextVar(delta->output());

  const auto & c0_1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 0);
  const auto & c1_1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 1);
  const auto & theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  const auto lv1 = theta->AddLoopVar(c0_1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(c1_1.output(0)); // sum
  const auto lv3 = theta->AddLoopVar(cv1.inner);      // arr ptr

  const auto & c1_2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);

  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1_2.output(0) }, 32);
  const auto res1 = addNode1.output(0);

  const auto & c0_2 = IntegerConstantOperation::Create(*theta->subregion(), 64, 0);
  const auto & lv1SExt = SExtOperation::create(64, lv1.pre);

  const auto gep = GetElementPtrOperation::Create(
      lv3.pre,
      { c0_2.output(0), lv1SExt },
      intArrayType,
      pointerType);

  auto loadedValue = LoadNonVolatileOperation::Create(gep, { memoryState.pre }, intType, 32)[0];

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, loadedValue }, 32);
  const auto res2 = addNode2.output(0);

  const auto & c4 = IntegerConstantOperation::Create(*theta->subregion(), 32, 4);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c4.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);

  lv1.post->divert_to(res1);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(gep), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is reliant on the result from a LOAD operation, and should therefore be {1,+,Unknown}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
  lv2TestChrec.AddOperand(SCEVUnknown::Create());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // lv3 (base pointer in GEP) is unchanged, it's recurrence should be {Init(a3)}
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(SCEVInit::Create(*lv3.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));

  // The GEP should have the following SCEV: (PH(a3) + ((0 * 20) + (PH(a1) * 4)))
  // Replacing the placeholders gives us ({Init(a3)} + (0 * 20) + ({0,+,1} * 4)})
  // Folding constants together gives us ({Init(a3)} + {0,+,4}) = {Init(a3),+,4} which is the
  // resulting recurrence
  auto gepTestChrec = SCEVChainRecurrence(*theta);
  gepTestChrec.AddOperand(SCEVInit::Create(*lv3.pre));
  gepTestChrec.AddOperand(SCEVConstant::Create(4));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(gepTestChrec, *chrecMap.at(gep)));
}

TEST(ScalarEvolutionTests, ComputeRecurrenceForStructGEP)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);
  const auto intStructType =
      StructType::CreateLiteral({ intType, intType, intType, intType, intType }, false);
  const auto pointerType = PointerType::Create();
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      LlvmLambdaOperation::Create(
          jlm::rvsdg::FunctionType::Create(
              { pointerType, memoryStateType },
              { intType, memoryStateType }),
          "f",
          Linkage::externalLinkage));

  const auto & delta = jlm::rvsdg::DeltaNode::Create(
      &graph.GetRootRegion(),
      DeltaOperation::Create(intStructType, "", Linkage::externalLinkage, "", false));

  const auto & structC1 = IntegerConstantOperation::Create(*delta->subregion(), 32, 1);
  const auto & structC2 = IntegerConstantOperation::Create(*delta->subregion(), 32, 2);
  const auto & structC3 = IntegerConstantOperation::Create(*delta->subregion(), 32, 3);
  const auto & structC4 = IntegerConstantOperation::Create(*delta->subregion(), 32, 4);
  const auto & structC5 = IntegerConstantOperation::Create(*delta->subregion(), 32, 5);

  auto & constantStruct = ConstantStruct::Create(
      *delta->subregion(),
      { structC1.output(0),
        structC2.output(0),
        structC3.output(0),
        structC4.output(0),
        structC5.output(0) },
      intStructType);
  delta->finalize(&constantStruct);

  const auto cv1 = lambda->AddContextVar(delta->output());

  const auto & c0_1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 0);
  const auto & c1_1 = IntegerConstantOperation::Create(*lambda->subregion(), 32, 1);
  const auto & theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());

  const auto memoryState = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  const auto lv1 = theta->AddLoopVar(c0_1.output(0)); // i
  const auto lv2 = theta->AddLoopVar(c1_1.output(0)); // sum
  const auto lv3 = theta->AddLoopVar(cv1.inner);      // struct ptr

  const auto & c1_2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);

  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1_2.output(0) }, 32);
  const auto res1 = addNode1.output(0);

  const auto & lv1SExt = SExtOperation::create(64, lv1.pre);
  const auto gep = GetElementPtrOperation::Create(lv3.pre, { lv1SExt }, intType, pointerType);

  auto loadedValue = LoadNonVolatileOperation::Create(gep, { memoryState.pre }, intType, 32)[0];

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, loadedValue }, 32);
  const auto res2 = addNode2.output(0);

  const auto & c4 = IntegerConstantOperation::Create(*theta->subregion(), 32, 4);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c4.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);

  lv1.post->divert_to(res1);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & chrecMap = RunScalarEvolution(rvsdgModule).first;

  // Assert
  EXPECT_NE(chrecMap.find(lv1.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv2.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(lv3.pre), chrecMap.end());
  EXPECT_NE(chrecMap.find(gep), chrecMap.end());

  // lv1 is a simple induction variable with the recurrence {0,+,1}
  auto lv1TestChrec = SCEVChainRecurrence(*theta);
  lv1TestChrec.AddOperand(SCEVConstant::Create(0));
  lv1TestChrec.AddOperand(SCEVConstant::Create(1));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv1TestChrec, *chrecMap.at(lv1.pre)));

  // lv2 is reliant on the result from a LOAD operation, and should therefore be {1,+,Unknown}
  auto lv2TestChrec = SCEVChainRecurrence(*theta);
  lv2TestChrec.AddOperand(SCEVConstant::Create(1));
  lv2TestChrec.AddOperand(SCEVUnknown::Create());
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv2TestChrec, *chrecMap.at(lv2.pre)));

  // lv3 (base pointer in GEP) is unchanged, it's recurrence should be {Init(a3)}
  auto lv3TestChrec = SCEVChainRecurrence(*theta);
  lv3TestChrec.AddOperand(SCEVInit::Create(*lv3.pre));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(lv3TestChrec, *chrecMap.at(lv3.pre)));

  // The GEP should have the following SCEV: (PH(a3) + (PH(a1) * 4))
  // Replacing the placeholders gives us ({Init(a3)} + ({0,+,1} * 4))
  // Folding constants together gives us ({Init(a3)} + {0,+,4}) = {Init(a3),+,4} which is the
  // resulting recurrence
  auto gepTestChrec = SCEVChainRecurrence(*theta);
  gepTestChrec.AddOperand(SCEVInit::Create(*lv3.pre));
  gepTestChrec.AddOperand(SCEVConstant::Create(4));
  EXPECT_TRUE(ScalarEvolution::StructurallyEqual(gepTestChrec, *chrecMap.at(gep)));
}

TEST(ScalarEvolutionTests, ComputeTripCountForSLTComparisonWithAffineRecurrence)
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
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  EXPECT_NE(tripCountMap.find(theta), tripCountMap.end());
  const auto tripCount = tripCountMap.at(theta);
  EXPECT_TRUE(tripCount == 5);
}

TEST(ScalarEvolutionTests, ComputeTripCountForSLEComparisonWithAffineRecurrence)
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
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sleNode = jlm::rvsdg::CreateOpNode<IntegerSleOperation>({ result, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sleNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  EXPECT_NE(tripCountMap.find(theta), tripCountMap.end());
  // For equals, we expect an extra iteration compared to strict lesser than
  const auto tripCount = tripCountMap.at(theta);

  EXPECT_TRUE(tripCount == 6);
}

TEST(ScalarEvolutionTests, ComputeTripCountForSGTComparisonWithAffineRecurrence)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c5 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 5);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c5.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = subNode.output(0);

  const auto & c0 = IntegerConstantOperation::Create(*theta->subregion(), 32, 0);
  auto & sgtNode = jlm::rvsdg::CreateOpNode<IntegerSgtOperation>({ result, c0.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sgtNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  EXPECT_NE(tripCountMap.find(theta), tripCountMap.end());
  const auto tripCount = tripCountMap.at(theta);
  EXPECT_TRUE(tripCount == 5);
}

TEST(ScalarEvolutionTests, ComputeTripCountForSGEComparisonWithAffineRecurrence)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c5 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 5);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c5.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = subNode.output(0);

  const auto & c0 = IntegerConstantOperation::Create(*theta->subregion(), 32, 0);
  auto & sgeNode = jlm::rvsdg::CreateOpNode<IntegerSgeOperation>({ result, c0.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sgeNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  EXPECT_NE(tripCountMap.find(theta), tripCountMap.end());
  // For equals, we expect an extra iteration compared to strict greater than
  const auto tripCount = tripCountMap.at(theta);
  EXPECT_TRUE(tripCount == 6);
}

TEST(ScalarEvolutionTests, ComputeTripCountForNEComparisonWithAffineRecurrence)
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
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & neNode = jlm::rvsdg::CreateOpNode<IntegerNeOperation>({ result, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*neNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  EXPECT_NE(tripCountMap.find(theta), tripCountMap.end());
  const auto tripCount = tripCountMap.at(theta);
  EXPECT_TRUE(tripCount == 5);
}

TEST(ScalarEvolutionTests, ComputeTripCountForEQComparisonWithAffineRecurrence)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));

  const auto & c1_1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1_1.output(0) }, 32);
  const auto result = addNode.output(0);

  const auto & c1_2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & eqNode = jlm::rvsdg::CreateOpNode<IntegerEqOperation>({ result, c1_2.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*eqNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  EXPECT_NE(tripCountMap.find(theta), tripCountMap.end());
  const auto tripCount = tripCountMap.at(theta);
  EXPECT_TRUE(tripCount == 2);
}

TEST(ScalarEvolutionTests, ComputeTripCountForInfiniteSLTComparisonWithAffineRecurrence)
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
  auto & subNode = jlm::rvsdg::CreateOpNode<IntegerSubOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = subNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  // Negative increment. Since the start value of lv1 (0) is already less than the comparison value
  // (5), we get an infinite loop.
  EXPECT_EQ(tripCountMap.find(theta), tripCountMap.end());
}

TEST(ScalarEvolutionTests, ComputeTripCountForInfiniteSGTComparisonWithAffineRecurrence)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c5 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 5);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c5.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = addNode.output(0);

  const auto & c0 = IntegerConstantOperation::Create(*theta->subregion(), 32, 0);
  auto & sgeNode = jlm::rvsdg::CreateOpNode<IntegerSgeOperation>({ result, c0.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sgeNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  // Positive increment. Since the start value of lv1 (5) is already greater than the comparison
  // value (0), we get an infinite loop.
  EXPECT_EQ(tripCountMap.find(theta), tripCountMap.end());
}

TEST(ScalarEvolutionTests, ComputeTripCountForInfiniteNEComparisonWithAffineRecurrence)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c2.output(0) }, 32);
  const auto result = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  auto & neNode = jlm::rvsdg::CreateOpNode<IntegerNeOperation>({ result, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*neNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  // Start value of 0, increment by 2. This means that lv1 will never have the value of 5, which
  // is the value in the "not equals" comparison. Therefore, we get an infinite loop.
  EXPECT_EQ(tripCountMap.find(theta), tripCountMap.end());
}

TEST(ScalarEvolutionTests, ComputeTripCountForSimpleQuadraticRecurrence)
{
  using namespace jlm::llvm;

  // Arrange
  const auto intType = jlm::rvsdg::BitType::Create(32);

  LlvmRvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  const auto & graph = rvsdgModule.Rvsdg();

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c1_1 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 1);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  const auto lv1 = theta->AddLoopVar(c0.output(0));
  const auto lv2 = theta->AddLoopVar(c1_1.output(0));

  const auto & c1_2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  auto & addNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1_2.output(0) }, 32);
  const auto res1 = addNode.output(0);

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, res1 }, 32);
  const auto res2 = addNode2.output(0);

  const auto & c10 = IntegerConstantOperation::Create(*theta->subregion(), 32, 10);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res2, c10.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  EXPECT_NE(tripCountMap.find(theta), tripCountMap.end());
  const auto tripCount = tripCountMap.at(theta);
  EXPECT_TRUE(tripCount == 4);
}

TEST(ScalarEvolutionTests, TestTripCountCouldNotCompute)
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
  auto & addNode1 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode1.output(0);

  auto & addNode2 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, res1 }, 32);
  const auto res2 = addNode2.output(0);

  auto & addNode3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv3.pre, res2 }, 32);
  const auto res3 = addNode3.output(0);

  const auto & c50 = IntegerConstantOperation::Create(*theta->subregion(), 32, 50);
  auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res3, c50.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);
  lv2.post->divert_to(res2);
  lv3.post->divert_to(res3);

  jlm::rvsdg::view(graph, stdout);

  // Act
  const auto & tripCountMap = RunScalarEvolution(rvsdgModule).second;

  // Assert
  // The recurrence is a third degree polynomial ({a,+,b,+,c,+,d}). For these cases, we return could
  // not compute, as there is no implementation for solving recurrences with an order greater
  // than 2.
  EXPECT_EQ(tripCountMap.find(theta), tripCountMap.end());
}
