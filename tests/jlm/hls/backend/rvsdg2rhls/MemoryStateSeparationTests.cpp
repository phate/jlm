/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static void
IndependentStores()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = ValueType::Create();
  auto functionType = FunctionType::Create(
      { pointerType, pointerType, valueType, memoryStateType },
      { memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto pointerArgument1 = lambdaNode->GetFunctionArguments()[0];
  auto pointerArgument2 = lambdaNode->GetFunctionArguments()[1];
  auto valueArgument = lambdaNode->GetFunctionArguments()[2];
  auto memoryStateArgument = lambdaNode->GetFunctionArguments()[3];

  auto & storeNode1 = StoreNonVolatileOperation::CreateNode(
      *pointerArgument1,
      *valueArgument,
      { memoryStateArgument },
      4);

  auto & storeNode2 = StoreNonVolatileOperation::CreateNode(
      *pointerArgument2,
      *valueArgument,
      { storeNode1.output(0) },
      4);

  auto lambdaOutput = lambdaNode->finalize({ storeNode2.output(0) });

  GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  MemoryStateSeparation::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/MemoryStateSeparationTests-IndependentStores",
    IndependentStores)

static void
DependentStores()
{
  using namespace jlm::hls;
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;
  using namespace jlm::tests;
  using namespace jlm::util;

  // Arrange
  auto controlType = ControlType::Create(2);
  auto pointerType = PointerType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto valueType = ValueType::Create();
  auto functionType = FunctionType::Create(
      { controlType, pointerType, pointerType, valueType, memoryStateType },
      { memoryStateType });

  jlm::llvm::RvsdgModule rvsdgModule(FilePath(""), "", "");
  auto & rvsdg = rvsdgModule.Rvsdg();

  auto lambdaNode = LambdaNode::Create(
      rvsdg.GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto controlArgument = lambdaNode->GetFunctionArguments()[0];
  auto pointerArgument1 = lambdaNode->GetFunctionArguments()[1];
  auto pointerArgument2 = lambdaNode->GetFunctionArguments()[2];
  auto valueArgument = lambdaNode->GetFunctionArguments()[3];
  auto memoryStateArgument = lambdaNode->GetFunctionArguments()[4];

  auto gammaNode = GammaNode::create(controlArgument, 2);
  auto pointer1EntryVar = gammaNode->AddEntryVar(pointerArgument1);
  auto pointer2EntryVar = gammaNode->AddEntryVar(pointerArgument2);

  auto pointerExitVar = gammaNode->AddExitVar(
      { pointer1EntryVar.branchArgument[0], pointer2EntryVar.branchArgument[1] });

  auto & storeNode1 = StoreNonVolatileOperation::CreateNode(
      *pointerArgument1,
      *valueArgument,
      { memoryStateArgument },
      4);

  auto & storeNode2 = StoreNonVolatileOperation::CreateNode(
      *pointerArgument2,
      *valueArgument,
      { storeNode1.output(0) },
      4);

  auto & storeNode3 = StoreNonVolatileOperation::CreateNode(
      *pointerExitVar.output,
      *valueArgument,
      { storeNode2.output(0) },
      4);

  auto lambdaOutput = lambdaNode->finalize({ storeNode3.output(0) });

  GraphExport::Create(*lambdaOutput, "");

  view(rvsdg, stdout);

  // Act
  StatisticsCollector statisticsCollector;
  MemoryStateSeparation::CreateAndRun(rvsdgModule, statisticsCollector);
  view(rvsdg, stdout);

  // Assert
}

JLM_UNIT_TEST_REGISTER(
    "jlm/hls/backend/rvsdg2rhls/MemoryStateSeparationTests-DependentStores",
    DependentStores)
