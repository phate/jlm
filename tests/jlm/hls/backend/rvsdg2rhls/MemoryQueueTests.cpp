/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rvsdg2rhls/mem-queue.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

TEST(MemoryQueueTests, TestSingleLoad)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() },
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto constant = &jlm::rvsdg::BitConstantOperation::create(*theta->subregion(), { 1, 1 });
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, constant);
  theta->set_predicate(match);

  // Load node
  auto functionArguments = lambda->GetFunctionArguments();
  auto loadAddress = theta->AddLoopVar(functionArguments[0]);
  auto memoryStateArgument = theta->AddLoopVar(functionArguments[1]);
  auto loadOutput = LoadNonVolatileOperation::Create(
      loadAddress.pre,
      { memoryStateArgument.pre },
      PointerType::Create(),
      32);
  loadAddress.post->divert_to(loadOutput[0]);
  memoryStateArgument.post->divert_to(loadOutput[1]);

  auto lambdaOutput = lambda->finalize({ theta->output(0), theta->output(1) });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryStateSeparation::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto & entryMemoryStateSplitInput = lambdaRegion->argument(1)->SingleUser();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(entryMemoryStateSplitInput);
  jlm::util::assertedCast<const LambdaEntryMemoryStateSplitOperation>(
      &entryMemoryStateSplitNode->GetOperation());
  auto exitMemoryStateMergeNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*lambdaRegion->result(1)->origin());
  jlm::util::assertedCast<const LambdaExitMemoryStateMergeOperation>(
      &exitMemoryStateMergeNode->GetOperation());

  // Act
  ThetaNodeConversion::CreateAndRun(*rvsdgModule, statisticsCollector);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsNodeType<LoopNode>(*lambdaRegion, true));

  // Act
  AddressQueueInsertion::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<StateGateOperation>(*lambdaRegion, true));
  EXPECT_FALSE(jlm::rvsdg::Region::ContainsOperation<AddressQueueOperation>(*lambdaRegion, true));
}

TEST(MemoryQueueTests, TestLoadStore)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(),
        jlm::llvm::PointerType::Create(),
        MemoryStateType::Create() },
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto constant = &jlm::rvsdg::BitConstantOperation::create(*theta->subregion(), { 1, 1 });
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, constant);
  theta->set_predicate(match);

  // Load node
  auto functionArguments = lambda->GetFunctionArguments();
  auto loadAddress = theta->AddLoopVar(functionArguments[0]);
  auto storeAddress = theta->AddLoopVar(functionArguments[1]);
  auto memoryStateArgument = theta->AddLoopVar(functionArguments[2]);
  auto loadOutput = LoadNonVolatileOperation::Create(
      loadAddress.pre,
      { memoryStateArgument.pre },
      PointerType::Create(),
      32);
  auto storeOutput = StoreNonVolatileOperation::Create(
      storeAddress.pre,
      &jlm::rvsdg::BitConstantOperation::create(*theta->subregion(), { 32, 1 }),
      { loadOutput[1] },
      32);

  loadAddress.post->divert_to(loadOutput[0]);
  memoryStateArgument.post->divert_to(storeOutput[0]);

  auto lambdaOutput = lambda->finalize({ theta->output(0), theta->output(2) });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryStateSeparation::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto & entryMemoryStateSplitInput = lambdaRegion->argument(2)->SingleUser();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(entryMemoryStateSplitInput);
  jlm::util::assertedCast<const LambdaEntryMemoryStateSplitOperation>(
      &entryMemoryStateSplitNode->GetOperation());
  auto exitMemoryStateMergeNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*lambdaRegion->result(1)->origin());
  jlm::util::assertedCast<const LambdaExitMemoryStateMergeOperation>(
      &exitMemoryStateMergeNode->GetOperation());

  // Act
  ThetaNodeConversion::CreateAndRun(*rvsdgModule, statisticsCollector);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsNodeType<LoopNode>(*lambdaRegion, true));

  // Act
  AddressQueueInsertion::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<StateGateOperation>(*lambdaRegion, true));
  EXPECT_FALSE(jlm::rvsdg::Region::ContainsOperation<AddressQueueOperation>(*lambdaRegion, true));
}

TEST(MemoryQueueTests, TestAddrQueue)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = RvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() },
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto constant = &jlm::rvsdg::BitConstantOperation::create(*theta->subregion(), { 1, 1 });
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, constant);
  theta->set_predicate(match);

  // Load node
  auto functionArguments = lambda->GetFunctionArguments();
  auto address = theta->AddLoopVar(functionArguments[0]);
  auto memoryStateArgument = theta->AddLoopVar(functionArguments[1]);
  auto loadOutput = LoadNonVolatileOperation::Create(
      address.pre,
      { memoryStateArgument.pre },
      PointerType::Create(),
      32);
  auto storeOutput =
      StoreNonVolatileOperation::Create(address.pre, loadOutput[0], { loadOutput[1] }, 32);

  address.post->divert_to(loadOutput[0]);
  memoryStateArgument.post->divert_to(storeOutput[0]);

  auto lambdaOutput = lambda->finalize({ theta->output(0), theta->output(1) });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryStateSeparation::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto & entryMemoryStateSplitInput = lambdaRegion->argument(1)->SingleUser();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(entryMemoryStateSplitInput);
  jlm::util::assertedCast<const LambdaEntryMemoryStateSplitOperation>(
      &entryMemoryStateSplitNode->GetOperation());
  auto exitMemoryStateMergeNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*lambdaRegion->result(1)->origin());
  jlm::util::assertedCast<const LambdaExitMemoryStateMergeOperation>(
      &exitMemoryStateMergeNode->GetOperation());

  // Act
  ThetaNodeConversion::CreateAndRun(*rvsdgModule, statisticsCollector);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsNodeType<LoopNode>(*lambdaRegion, true));

  // Act
  AddressQueueInsertion::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<StateGateOperation>(*lambdaRegion, true));
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<AddressQueueOperation>(*lambdaRegion, true));

  for (auto & node : jlm::rvsdg::TopDownTraverser(lambdaRegion))
  {
    if (auto loopNode = dynamic_cast<jlm::hls::LoopNode *>(node))
    {
      for (auto & node : jlm::rvsdg::TopDownTraverser(loopNode->subregion()))
      {
        if (is<StoreNonVolatileOperation>(node))
        {
          auto loadNode =
              jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*node->input(1)->origin());
          jlm::util::assertedCast<const jlm::llvm::LoadOperation>(&loadNode->GetOperation());
          auto stateGate =
              jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*loadNode->input(0)->origin());
          jlm::util::assertedCast<const StateGateOperation>(&stateGate->GetOperation());
          auto addrQueue =
              jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*stateGate->input(0)->origin());
          jlm::util::assertedCast<const AddressQueueOperation>(&addrQueue->GetOperation());
        }
      }
    }
  }
}
