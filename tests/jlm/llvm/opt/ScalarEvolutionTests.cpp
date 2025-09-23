/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/llvm/ir/operators/IntegerOperations.hpp"
#include "jlm/rvsdg/bitstring/arithmetic.hpp"
#include "jlm/rvsdg/bitstring/constant.hpp"
#include "jlm/util/Statistics.hpp"
#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>

#include <cassert>

static jlm::llvm::ScalarEvolution::InductionVariableSet
RunScalarEvolution(const jlm::rvsdg::ThetaNode * thetaNode)
{
  jlm::util::StatisticsCollector statisticsCollector;
  jlm::llvm::ScalarEvolution scalarEvolution;
  return scalarEvolution.FindInductionVariables(thetaNode);
}

static void
SimpleInductionVariable()
{
  using namespace jlm::llvm;

  // Arrange
  const auto valueType = jlm::tests::ValueType::Create();
  const auto intType = jlm::rvsdg::BitType::Create(32);

  RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();
  const auto x = &jlm::rvsdg::GraphImport::Create(graph, intType, "x");
  const auto y = &jlm::rvsdg::GraphImport::Create(graph, valueType, "y");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto lv1 = theta->AddLoopVar(x);
  lv1.input->divert_to(c0.output(0));
  theta->AddLoopVar(y);

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  const auto & addNode =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto result = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  const auto & sltNode =
      jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ result, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(result);

  jlm::rvsdg::view(graph, stdout);

  // Act
  ScalarEvolution::InductionVariableSet inductionVariables = RunScalarEvolution(theta);

  // Assert
  assert(inductionVariables.Size() == 2);
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
  auto & graph = rvsdgModule.Rvsdg();
  const auto x = &jlm::rvsdg::GraphImport::Create(graph, intType, "x");
  const auto y = &jlm::rvsdg::GraphImport::Create(graph, intType, "y");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto lv1 = theta->AddLoopVar(x);
  lv1.input->divert_to(c0.output(0));
  auto lv2 = theta->AddLoopVar(y);

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  const auto & addNode =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode.output(0);

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  const auto & mulNode =
      jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ lv2.pre, c2.output(0) }, 32);
  const auto res2 = mulNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  const auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);
  lv2.post->divert_to(res2);

  jlm::rvsdg::view(graph, stdout);

  // Act
  ScalarEvolution::InductionVariableSet inductionVariables = RunScalarEvolution(theta);

  // Assert
  assert(inductionVariables.Size() == 1);
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
  auto & graph = rvsdgModule.Rvsdg();
  const auto x = &jlm::rvsdg::GraphImport::Create(graph, intType, "x");
  const auto y = &jlm::rvsdg::GraphImport::Create(graph, intType, "y");

  const auto & c0 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 0);
  const auto & c4 = IntegerConstantOperation::Create(graph.GetRootRegion(), 32, 4);

  const auto theta = jlm::rvsdg::ThetaNode::create(&graph.GetRootRegion());
  auto lv1 = theta->AddLoopVar(x);
  lv1.input->divert_to(c0.output(0));
  auto lv2 = theta->AddLoopVar(y);
  lv2.input->divert_to(c4.output(0));

  const auto & c1 = IntegerConstantOperation::Create(*theta->subregion(), 32, 1);
  const auto & addNode =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv1.pre, c1.output(0) }, 32);
  const auto res1 = addNode.output(0);

  const auto & c5 = IntegerConstantOperation::Create(*theta->subregion(), 32, 5);
  const auto & sltNode = jlm::rvsdg::CreateOpNode<IntegerSltOperation>({ res1, c5.output(0) }, 32);
  const auto matchResult =
      jlm::rvsdg::MatchOperation::Create(*sltNode.output(0), { { 1, 1 } }, 0, 2);

  const auto & c2 = IntegerConstantOperation::Create(*theta->subregion(), 32, 2);
  const auto & addNode2 =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ lv2.pre, c2.output(0) }, 32);
  const auto res2 = addNode2.output(0);

  const auto & c3 = IntegerConstantOperation::Create(*theta->subregion(), 32, 3);
  const auto & addNode3 = jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ res2, c3.output(0) }, 32);
  const auto res3 = addNode3.output(0);

  theta->set_predicate(matchResult);
  lv1.post->divert_to(res1);
  lv2.post->divert_to(res3);

  jlm::rvsdg::view(graph, stdout);

  // Act
  ScalarEvolution::InductionVariableSet inductionVariables = RunScalarEvolution(theta);

  // Assert
  assert(inductionVariables.Size() == 2);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/ScalarEvolutionTests-RecursiveInductionVariable",
    RecursiveInductionVariable)
