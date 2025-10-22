/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
MultipleReductionsPerRegion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  const auto bitType = BitType::Create(32);
  const auto memoryStateType = MemoryStateType::Create();

  jlm::llvm::RvsdgModule rvsdgModule(jlm::util::FilePath(""), "", "");
  auto & graph = rvsdgModule.Rvsdg();

  auto & sizeArgument = jlm::rvsdg::GraphImport::Create(graph, bitType, "size");
  auto allocaResults = AllocaOperation::create(bitType, &sizeArgument, 4);

  const auto c3 = bitconstant_op::create(&graph.GetRootRegion(), BitValueRepresentation(32, 3));
  auto storeResults =
      StoreNonVolatileOperation::Create(allocaResults[0], c3, { allocaResults[1] }, 4);
  auto loadResults =
      LoadNonVolatileOperation::Create(allocaResults[0], { storeResults[0] }, bitType, 4);

  const auto c5 = bitconstant_op::create(&graph.GetRootRegion(), BitValueRepresentation(32, 5));
  auto sum = bitadd_op::create(32, loadResults[0], c5);

  auto & sumExport = jlm::rvsdg::GraphExport::Create(*sum, "sum");

  view(graph, stdout);

  // Act
  NodeReduction nodeReduction;
  jlm::util::StatisticsCollector statisticsCollector(
      jlm::util::StatisticsCollectorSettings({ jlm::util::Statistics::Id::ReduceNodes }));
  nodeReduction.Run(rvsdgModule, statisticsCollector);

  view(graph, stdout);

  // Assert
  // We expect that two reductions are applied:
  // 1. NormalizeLoadStore - This ensures that the stored constant value is directly forwarded to
  // the add operation
  // 2. Constant folding on the add operation
  // The result is that a single constant node with value 8 is left in the graph.
  assert(graph.GetRootRegion().numNodes() == 1);

  auto constantNode = TryGetOwnerNode<SimpleNode>(*sumExport.origin());
  auto constantOperation = dynamic_cast<const bitconstant_op *>(&constantNode->GetOperation());
  assert(constantOperation->value().to_uint() == 8);

  // We expect that the node reductions transformation iterated over the root region 2 times.
  auto & statistics = *statisticsCollector.CollectedStatistics().begin();
  auto & nodeReductionStatistics = dynamic_cast<const NodeReduction::Statistics &>(statistics);
  auto numIterations = nodeReductionStatistics.GetNumIterations(graph.GetRootRegion()).value();
  assert(numIterations == 2);
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/opt/NodeReductionTests-MultipleReductionsPerRegion",
    MultipleReductionsPerRegion)
