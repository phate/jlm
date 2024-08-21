/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/hls/backend/rvsdg2rhls/DeadNodeElimination.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>

static void
TestDeadLoopNode()
{
  using namespace jlm::hls;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType =
      jlm::llvm::FunctionType::Create({ jlm::rvsdg::ctltype::Create(2), valueType }, { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = jlm::llvm::lambda::node::create(
      rvsdg.root(),
      functionType,
      "f",
      jlm::llvm::linkage::external_linkage);

  loop_node::create(lambdaNode->subregion());

  lambdaNode->finalize({ lambdaNode->fctargument(1) });

  // Act
  EliminateDeadNodes(rvsdgModule);

  // Assert
  assert(lambdaNode->subregion()->nnodes() == 0);
}

static void
TestDeadLoopNodeOutput()
{
  using namespace jlm::hls;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  auto functionType = jlm::llvm::FunctionType::Create(
      { jlm::rvsdg::ctltype::Create(2), valueType },
      { jlm::rvsdg::ctltype::Create(2) });

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::filepath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = jlm::llvm::lambda::node::create(
      rvsdg.root(),
      functionType,
      "f",
      jlm::llvm::linkage::external_linkage);

  auto p = lambdaNode->fctargument(0);
  auto x = lambdaNode->fctargument(1);

  auto loopNode = loop_node::create(lambdaNode->subregion());

  jlm::rvsdg::output * buffer;
  auto output0 = loopNode->add_loopvar(p, &buffer);
  loopNode->add_loopvar(x);
  loopNode->set_predicate(buffer);

  auto lambdaOutput = lambdaNode->finalize({ output0 });

  jlm::llvm::GraphExport::Create(*lambdaOutput, "f");

  // Act
  EliminateDeadNodes(rvsdgModule);

  // Assert
  assert(loopNode->noutputs() == 1);
  assert(loopNode->ninputs() == 2); // I believe that it actually should only have one input.
  // FIXME: The DNE seems to already be broken for a simple dead edge through it. It removes the
  // output from the loop node, but then seems to fail to remove the corresponding input, arguments,
  // and results.
}

static int
TestDeadNodeElimination()
{
  TestDeadLoopNode();
  TestDeadLoopNodeOutput();

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/DeadNodeEliminationTests",
    TestDeadNodeElimination)
