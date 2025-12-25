/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/hls/backend/rvsdg2rhls/DeadNodeElimination.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/TestType.hpp>

static void
TestDeadLoopNode()
{
  using namespace jlm::hls;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::ControlType::Create(2), valueType },
      { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::Linkage::externalLinkage));

  LoopNode::create(lambdaNode->subregion());

  lambdaNode->finalize({ lambdaNode->GetFunctionArguments()[1] });

  // Act
  EliminateDeadNodes(rvsdgModule);

  // Assert
  assert(lambdaNode->subregion()->numNodes() == 0);
}

static void
TestDeadLoopNodeOutput()
{
  using namespace jlm::hls;

  // Arrange
  auto valueType = jlm::rvsdg::TestType::createValueType();
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::ControlType::Create(2), valueType },
      { jlm::rvsdg::ControlType::Create(2) });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::Linkage::externalLinkage));

  auto p = lambdaNode->GetFunctionArguments()[0];
  auto x = lambdaNode->GetFunctionArguments()[1];

  auto loopNode = LoopNode::create(lambdaNode->subregion());

  jlm::rvsdg::Output * buffer = nullptr;
  auto output0 = loopNode->AddLoopVar(p, &buffer);
  loopNode->AddLoopVar(x);
  loopNode->set_predicate(buffer);

  auto lambdaOutput = lambdaNode->finalize({ output0 });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  // Act
  EliminateDeadNodes(rvsdgModule);

  // Assert
  assert(loopNode->noutputs() == 1);
  assert(loopNode->ninputs() == 2); // I believe that it actually should only have one input.
  // FIXME: The DNE seems to already be broken for a simple dead edge through it. It removes the
  // output from the loop node, but then seems to fail to remove the corresponding input, arguments,
  // and results.
}

static void
TestDeadNodeElimination()
{
  TestDeadLoopNode();
  TestDeadLoopNodeOutput();
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/DeadNodeEliminationTests",
    TestDeadNodeElimination)
