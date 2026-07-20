/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/NodeReduction.hpp>
#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/bitstring/value-representation.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/TestNodes.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
TEST(NodeReductionTests, MultipleReductionsPerRegion)
{
  using namespace jlm::rvsdg;

  // Arrange
  const auto bitType = BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  LlvmRvsdgModule rvsdgModule(util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto & sizeArgument = GraphImport::Create(graph, bitType, "size");

  auto testStructuralNode = TestStructuralNode::create(&graph.GetRootRegion(), 1);
  auto & subregion = *testStructuralNode->subregion(0);
  auto inputVar = testStructuralNode->addInputWithArguments(sizeArgument);

  auto allocaResults = AllocaOperation::create(bitType, inputVar.argument[0], 4);

  const auto c3 =
      &BitConstantOperation::create(subregion, BitValueRepresentation(32, 3));
  auto storeResults =
      StoreNonVolatileOperation::Create(allocaResults[0], c3, { allocaResults[1] }, 4);
  auto loadResults =
      LoadNonVolatileOperation::Create(allocaResults[0], { storeResults[0] }, bitType, 4);

  const auto c5 =
      &BitConstantOperation::create(subregion, BitValueRepresentation(32, 5));
  auto sum = bitadd_op::create(32, loadResults[0], c5);

  auto outputVar = testStructuralNode->addOutputWithResults({ sum });

  GraphExport::Create(*outputVar.output, "sum");

  view(graph, stdout);

  // Act
  NodeReduction nodeReduction;
  util::StatisticsCollector statisticsCollector(
      util::StatisticsCollectorSettings({ util::Statistics::Id::ReduceNodes }));
  nodeReduction.Run(rvsdgModule, statisticsCollector);

  view(graph, stdout);

  // Assert
  // We expect that two reductions are applied:
  // 1. NormalizeLoadStore - This ensures that the stored constant value is directly forwarded to
  // the add operation
  // 2. Constant folding on the add operation
  // The result is that a single constant node with value 8 is left in the graph.
  EXPECT_EQ(graph.GetRootRegion().numNodes(), 1u);

  auto constantNode = TryGetOwnerNode<SimpleNode>(*outputVar.result[0]->origin());
  auto constantOperation =
      dynamic_cast<const BitConstantOperation *>(&constantNode->GetOperation());
  EXPECT_EQ(constantOperation->value().to_uint(), 8u);

  auto & statistics = *statisticsCollector.CollectedStatistics().begin();
  auto & nodeReductionStatistics = dynamic_cast<const NodeReduction::Statistics &>(statistics);

  EXPECT_EQ(nodeReductionStatistics.GetNumIterations(graph.GetRootRegion()).value(), 1u);
  EXPECT_EQ(nodeReductionStatistics.GetNumIterations(subregion), 2u);
  EXPECT_EQ(nodeReductionStatistics.getNumRegions(), 2u);
  EXPECT_EQ(nodeReductionStatistics.getTotalIterations(), 3u);
  EXPECT_EQ(nodeReductionStatistics.getMaxIterationsPerRegion(), 2u);
}

}
