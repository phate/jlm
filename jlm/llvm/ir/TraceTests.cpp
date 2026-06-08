/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/Bitcast.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>

#include <cassert>

TEST(TraceTests, testTracingIOBarrier)
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
  EXPECT_EQ(&jlm::llvm::traceOutput(*ioBarrier1Output), myInt);
  EXPECT_EQ(&jlm::llvm::traceOutput(*ioBarrier2Output), myInt);
}

TEST(TraceTests, testGetConstantSignedInteger)
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
  EXPECT_EQ(tryGetConstantSignedInteger(*bits64Output), -37);
  EXPECT_EQ(tryGetConstantSignedInteger(*bits64CtxVar), -37);

  // The 20 can be found both before, inside and after the gamma
  EXPECT_EQ(tryGetConstantSignedInteger(*integerConstantNode.output(0)), 20);
  EXPECT_EQ(tryGetConstantSignedInteger(*entryVar.branchArgument[0]), 20);
  EXPECT_EQ(tryGetConstantSignedInteger(*entryVar.branchArgument[1]), 20);
  EXPECT_EQ(tryGetConstantSignedInteger(*exitVarOutput), 20);

  // A match output is not a constant integer, neither is the lambda output
  EXPECT_EQ(tryGetConstantSignedInteger(*matchOutput), std::nullopt);
  EXPECT_EQ(tryGetConstantSignedInteger(*lambdaOutput), std::nullopt);
}

TEST(TraceTests, testGetConstantSignedIntegerExtAndTrunc)
{
  using namespace jlm;
  using namespace jlm::llvm;

  /**
   * Creates an RVSDG graph that look like:
   *
   * c = BITS32(5)
   * sext = SExt(32 -> 64) c  // Should be 5
   * zext = ZExt(32 -> 64) c  // Should be 5
   *
   * c2 = BITS8(-20)          // Should be -20  (0b1110 1100)
   * sext2 = SExt(8 -> 32) c2 // Should be -20  (0b1111 .... 1110 1100)
   * zext2 = ZExt(8 -> 32) c2 // Should be 236  (0b0000 .... 1110 1100)
   *
   * c3 = BITS32(1023)
   * trunc3 = Trunc(32 -> 8)       // Should be -1   (0b1111 1111)
   * sext3 = SExt(8 -> 32) trunc3  // Should be -1   (0b1111 .... 1111 1111)
   * zext3 = ZExt(8 -> 32) trunc3  // Should be  255 (0b0000 .... 1111 1111)
   *
   * and uses tryGetConstantSignedInteger to get the integer values of the different constants.
   */

  // Arrange
  rvsdg::Graph graph;

  const auto int64Type = rvsdg::BitType::Create(64);
  const auto int32Type = rvsdg::BitType::Create(32);
  const auto int8Type = rvsdg::BitType::Create(8);

  auto & bits32Output5 = rvsdg::BitConstantOperation::create(
      graph.GetRootRegion(),
      rvsdg::BitValueRepresentation(32, 5));
  auto & sextOutput = SExtOperation::create(64, bits32Output5);
  auto & zextOutput = ZExtOperation::create(64, bits32Output5);

  auto & bits8OutputMinus20 = rvsdg::BitConstantOperation::create(
      graph.GetRootRegion(),
      rvsdg::BitValueRepresentation(8, -20));
  auto & sext2Output = SExtOperation::create(32, bits8OutputMinus20);
  auto & zext2Output = ZExtOperation::create(32, bits8OutputMinus20);

  auto & bits32Output1023 = rvsdg::BitConstantOperation::create(
      graph.GetRootRegion(),
      rvsdg::BitValueRepresentation(32, 1023));
  auto & truncOutput = TruncOperation::create(8, bits32Output1023);
  auto & sext3Output = SExtOperation::create(32, truncOutput);
  auto & zext3Output = ZExtOperation::create(32, truncOutput);

  // Assert
  // c = BITS32(5), sext = SExt(32 -> 64), zext = ZExt(32 -> 64)
  EXPECT_EQ(tryGetConstantSignedInteger(bits32Output5), 5);
  EXPECT_EQ(tryGetConstantSignedInteger(sextOutput), 5);
  EXPECT_EQ(tryGetConstantSignedInteger(zextOutput), 5);

  // c2 = BITS8(-20), sext2 = SExt(8 -> 32), zext2 = ZExt(8 -> 32)
  EXPECT_EQ(tryGetConstantSignedInteger(bits8OutputMinus20), -20);
  EXPECT_EQ(tryGetConstantSignedInteger(sext2Output), -20);
  EXPECT_EQ(tryGetConstantSignedInteger(zext2Output), 236);

  // c3 = BITS32(1023), trunc3 = Trunc(32 -> 8), sext3 = SExt(8 -> 32), zext3 = ZExt(8 -> 32)
  EXPECT_EQ(tryGetConstantSignedInteger(bits32Output1023), 1023);
  EXPECT_EQ(tryGetConstantSignedInteger(truncOutput), -1);
  EXPECT_EQ(tryGetConstantSignedInteger(sext3Output), -1);
  EXPECT_EQ(tryGetConstantSignedInteger(zext3Output), 255);
}
