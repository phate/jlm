/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <TestRvsdgs.hpp>

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/trace.hpp>
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
  assert(&jlm::llvm::TraceOutput(*ioBarrier1Output) == myInt);
  assert(&jlm::llvm::TraceOutput(*ioBarrier2Output) == myInt);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TraceTests-testTracingIOBarrier", testTracingIOBarrier)


static void
testGetConstantSignedInteger()
{
  using namespace jlm;
  using namespace jlm::llvm;

  // Creates a graph that looks like
  //
  //    IntegerConstantOperation(20)  BITS32(-34)  GraphImport("io")
  //                 |                    |               |
  //      undef      |                    v               v
  //        v        v
  //      +-----------------------+
  //      |  gamma                |
  //      | +-------+   +-------+ |
  //      | |   |   |   |   |   | |
  //      | |   v   |   |   v   | |
  //      | +-------+   +-------+ |
  //      +-----------------------+
  //                 |
  //                 v
  //
  // And checks that outputs with constant integer values lead to the correct value.
  //

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
  assert(&jlm::llvm::TraceOutput(*ioBarrier1Output) == myInt);
  assert(&jlm::llvm::TraceOutput(*ioBarrier2Output) == myInt);
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TraceTests-testGetConstantSignedInteger", testGetConstantSignedInteger)