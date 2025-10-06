/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/hls/backend/rvsdg2rhls/add-sinks.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/view.hpp>

static void
SinkInsertion()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::util;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  auto functionType = FunctionType::Create({ valueType }, { valueType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = jlm::rvsdg::LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::external_linkage));
  auto argument = lambdaNode->GetFunctionArguments()[0];

  auto structuralNode = jlm::tests::TestStructuralNode::create(lambdaNode->subregion(), 1);
  const auto inputVar0 = structuralNode->AddInputWithArguments(*argument);
  const auto inputVar1 = structuralNode->AddInputWithArguments(*argument);

  const auto outputVar0 = structuralNode->AddOutputWithResults({ inputVar1.argument[0] });
  const auto outputVar1 = structuralNode->AddOutputWithResults({ inputVar1.argument[0] });

  auto lambdaOutput = lambdaNode->finalize({ outputVar1.output });

  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  SinkInsertion::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
  assert(structuralNode->subregion(0)->nnodes() == 1);
  assert(lambdaNode->subregion()->nnodes() == 2);

  // The sink insertion pass should have inserted a SinkOperation node at output o0
  {
    assert(outputVar0.output->nusers() == 1);
    assert(IsOwnerNodeOperation<SinkOperation>(*outputVar0.output->Users().begin()));
  }

  // The sink insertion pass should have inserted a SinkOperation node at the argument of i0
  {
    auto & i0Argument = *inputVar0.argument[0];
    assert(i0Argument.nusers() == 1);
    assert(IsOwnerNodeOperation<SinkOperation>(*i0Argument.Users().begin()));
  }
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/SinkInsertionTests-SinkInsertion", SinkInsertion)
