/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>

#include <cassert>

static void
testTracingIOBarrier()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Creates a graph that looks like
  // GraphImport("x")  GraphImport("io")
  //       |        /---------/
  //       v       v          |
  //      IOBarrier           |
  //          |     /---------/
  //          v    v
  //      IOBarrier
  //
  // And checks that both the IOBarrier outputs are traced back up to the "x" graph import

  // Arrange
  rvsdg::Graph graph;

  const auto int32Type = rvsdg::BitType::Create(32);
  const auto ioStateType = IOStateType::Create();

  const auto myInt = &rvsdg::GraphImport::Create(graph, int32Type, "x");
  const auto myIo = &rvsdg::GraphImport::Create(graph, ioStateType, "io");

  const auto ioBarrier1 = &rvsdg::CreateOpNode<IOBarrierOperation>({ myInt, myIo }, int32Type);
  const auto ioBarrier1Output = ioBarrier1->output(0);

  const auto ioBarrier2 =
      &rvsdg::CreateOpNode<IOBarrierOperation>({ ioBarrier1Output, myIo }, int32Type);
  const auto ioBarrier2Output = ioBarrier2->output(0);

  // Assert
  assert(&jlm::llvm::traceOutput(*ioBarrier1Output) == myInt);
  assert(&jlm::llvm::traceOutput(*ioBarrier2Output) == myInt);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TraceTests-testTracingIOBarrier", testTracingIOBarrier)

static void
testGetConstantSignedInteger()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Creates a graph that looks like
  //
  //     BITS64(-37)
  //         |
  //         v
  //   +-------------------------------------------+
  //   | LAMBDA f()                                |
  //   +-------------------------------------------+
  //   |     |                                     |
  //   |     |                                     |
  //   |     |                                     |
  //   |     |     IntegerConstantOperation(20)    |
  //   |     v        |                            |
  //   |   MATCH      |                            |
  //   |     v        v                            |
  //   |   +-----------------------+               |
  //   |   |  gamma                |               |
  //   |   | +-------+   +-------+ |               |
  //   |   | |   |   |   |   |   | |               |
  //   |   | |   v   |   |   v   | |               |
  //   |   | +-------+   +-------+ |               |
  //   |   +-----------------------+               |
  //   |              |                            |
  //   |              v                            |
  //   +-------------------------------------------+
  // And checks that outputs with constant integer values lead to the correct value.
  //

  // Arrange
  rvsdg::Graph graph;

  const auto int64Type = rvsdg::BitType::Create(64);
  const auto int32Type = rvsdg::BitType::Create(32);

  const auto bits64Output = &rvsdg::BitConstantOperation::create(
      graph.GetRootRegion(),
      rvsdg::BitValueRepresentation(64, -37));

  const auto functionType = rvsdg::FunctionType::Create({}, { int32Type });
  const auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      graph.GetRootRegion(),
      jlm::llvm::LlvmLambdaOperation::Create(
          functionType,
          "f",
          jlm::llvm::Linkage::externalLinkage));

  const auto bits64CtxVar = lambdaNode->AddContextVar(*bits64Output).inner;
  const auto matchOutput = rvsdg::MatchOperation::Create(*bits64CtxVar, { { 0, 0 } }, 1, 2);
  const auto & integerConstantNode =
      IntegerConstantOperation::Create(*lambdaNode->subregion(), 32, 20);

  auto & gammaNode = *rvsdg::GammaNode::create(matchOutput, 2);
  const auto entryVar = gammaNode.AddEntryVar(integerConstantNode.output(0));

  const auto exitVarOutput =
      gammaNode.AddExitVar({ entryVar.branchArgument[0], entryVar.branchArgument[1] }).output;
  const auto lambdaOutput = lambdaNode->finalize({ exitVarOutput });

  // Assert

  // The -37 can be found both inside and outside the lambda
  assert(tryGetConstantSignedInteger(*bits64Output) == -37);
  assert(tryGetConstantSignedInteger(*bits64CtxVar) == -37);

  // The 20 can be found both before, inside and after the gamma
  assert(tryGetConstantSignedInteger(*integerConstantNode.output(0)) == 20);
  assert(tryGetConstantSignedInteger(*entryVar.branchArgument[0]) == 20);
  assert(tryGetConstantSignedInteger(*entryVar.branchArgument[1]) == 20);
  assert(tryGetConstantSignedInteger(*exitVarOutput) == 20);

  // A match output is not a constant integer, neither is the lambda output
  assert(tryGetConstantSignedInteger(*matchOutput) == std::nullopt);
  assert(tryGetConstantSignedInteger(*lambdaOutput) == std::nullopt);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TraceTests-testGetConstantSignedInteger",
    testGetConstantSignedInteger)
