/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/NodeNormalization.hpp>
#include <jlm/rvsdg/TestOperations.hpp>

namespace jlm::llvm
{

TEST(MemoryHoistBarrierTests, normalizeNestedMemoryHoistBarriers)
{
  using namespace jlm::rvsdg;

  // Arrange
  auto ptrType = PointerType::Create();
  auto ioStateType = IOStateType::Create();

  Graph graph;
  auto & a1 = GraphImport::Create(graph, ptrType, "a0");
  auto & a2 = GraphImport::Create(graph, ptrType, "a1");
  auto & ioState = GraphImport::Create(graph, ioStateType, "ioState");

  auto & mhbNode1 = MemoryHoistBarrierOperation::createNode(a1, ioState, 1);
  auto & mhbNode2 = MemoryHoistBarrierOperation::createNode(*mhbNode1.output(0), ioState, 4);

  auto & mhbNode3 = MemoryHoistBarrierOperation::createNode(a2, ioState, 2);

  auto testNode = TestOperation::createNode(&graph.GetRootRegion(), { &ioState }, { ioStateType });
  auto & mhbNode4 =
      MemoryHoistBarrierOperation::createNode(*mhbNode1.output(0), *testNode->output(0), 3);

  auto & ctlFalse = ControlConstantOperation::createFalse(graph.GetRootRegion());
  auto gammaNode = GammaNode::create(&ctlFalse, 2);
  auto ptrEntryVar = gammaNode->AddEntryVar(mhbNode1.output(0));
  auto ioStateEntryVar = gammaNode->AddEntryVar(&ioState);

  // gammaNode - subregion 0
  auto & mhbNode5 = MemoryHoistBarrierOperation::createNode(
      *ptrEntryVar.branchArgument[0],
      *ioStateEntryVar.branchArgument[0],
      5);

  // gammaNode - subregion 1
  // Nothing needs to be done

  // gammaNode - finalize
  auto ptrExitVar = gammaNode->AddExitVar({ mhbNode5.output(0), ptrEntryVar.branchArgument[1] });

  auto & x1 = GraphExport::Create(*mhbNode1.output(0), "x1");
  auto & x2 = GraphExport::Create(*mhbNode2.output(0), "x2");
  auto & x3 = GraphExport::Create(*mhbNode3.output(0), "x3");
  auto & x4 = GraphExport::Create(*mhbNode4.output(0), "x4");
  auto & x5 = GraphExport::Create(*ptrExitVar.output, "x5");

  // Act
  ReduceNode<MemoryHoistBarrierOperation>(
      MemoryHoistBarrierOperation::normalizeNestedMemoryHoistBarriers,
      mhbNode1);
  ReduceNode<MemoryHoistBarrierOperation>(
      MemoryHoistBarrierOperation::normalizeNestedMemoryHoistBarriers,
      mhbNode2);
  ReduceNode<MemoryHoistBarrierOperation>(
      MemoryHoistBarrierOperation::normalizeNestedMemoryHoistBarriers,
      mhbNode3);
  ReduceNode<MemoryHoistBarrierOperation>(
      MemoryHoistBarrierOperation::normalizeNestedMemoryHoistBarriers,
      mhbNode4);
  ReduceNode<MemoryHoistBarrierOperation>(
      MemoryHoistBarrierOperation::normalizeNestedMemoryHoistBarriers,
      mhbNode5);

  // Assert
  {
    // We expect that nothing happened with mhbNode1
    EXPECT_EQ(x1.origin(), mhbNode1.output(0));
  }

  {
    // We expect that mhbNode2 was replaced
    auto [mhbNode, mhbOp] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryHoistBarrierOperation>(*x2.origin());
    EXPECT_NE(mhbNode, nullptr);
    EXPECT_EQ(mhbOp->getDereferenceableSize(), 1);
    EXPECT_EQ(mhbNode->input(0)->origin(), &a1);
    EXPECT_EQ(mhbNode->input(1)->origin(), &ioState);
  }

  {
    // We expect that nothing happened with mhbNode3
    EXPECT_EQ(x3.origin(), mhbNode3.output(0));
  }

  {
    // We expect that nothing happened with mhbNode4
    EXPECT_EQ(x4.origin(), mhbNode4.output(0));
  }

  {
    // We expect that nothing happened with mhbNode5
    EXPECT_EQ(x5.origin(), mhbNode5.output(0));
  }
}

}
