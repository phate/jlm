/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-queue.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

TEST(MemoryConverterTests, TestTraceArgument)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(),
        jlm::llvm::PointerType::Create(),
        jlm::rvsdg::BitType::Create(32),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Load followed by store
  auto loadAddress = lambda->GetFunctionArguments()[0];
  auto memoryStateArgument = lambda->GetFunctionArguments()[3];
  auto loadOutput = LoadNonVolatileOperation::Create(
      loadAddress,
      { memoryStateArgument },
      jlm::llvm::PointerType::Create(),
      32);

  auto storeAddress = lambda->GetFunctionArguments()[1];
  auto storeData = lambda->GetFunctionArguments()[2];
  auto storeOutput =
      StoreNonVolatileOperation::Create(storeAddress, storeData, { loadOutput[1] }, 32);

  auto lambdaOutput = lambda->finalize({ storeOutput[0] });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  // Act
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  const auto tracedPointerNodesVector = TracePointerArguments(lambda);
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Assert
  EXPECT_EQ(tracedPointerNodesVector.size(), 2);               // 2 pointer arguments
  EXPECT_EQ(tracedPointerNodesVector[0].loadNodes.size(), 1);  // 1 load for the first pointer
  EXPECT_EQ(tracedPointerNodesVector[0].storeNodes.size(), 0); // 0 store for the first pointer
  EXPECT_EQ(
      tracedPointerNodesVector[0].decoupleNodes.size(),
      0);                                                      // 0 decouple for the first pointer
  EXPECT_EQ(tracedPointerNodesVector[1].loadNodes.size(), 0);  // 0 load for the first pointer
  EXPECT_EQ(tracedPointerNodesVector[1].storeNodes.size(), 1); // 1 store for the second pointer
  EXPECT_EQ(tracedPointerNodesVector[1].decoupleNodes.size(), 0); // 0 load for the first pointer
}

TEST(MemoryConverterTests, TestLoad)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() },
      { jlm::rvsdg::BitType::Create(32), MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Single load
  auto loadAddress = lambda->GetFunctionArguments()[0];
  auto memoryStateArgument = lambda->GetFunctionArguments()[1];
  auto loadOutput = LoadNonVolatileOperation::Create(
      loadAddress,
      { memoryStateArgument },
      jlm::rvsdg::BitType::Create(32),
      32);

  auto lambdaOutput = lambda->finalize({ loadOutput[0], loadOutput[1] });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryConverter::CreateAndRun(*rvsdgModule, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Memory Converter replaces the lambda so we start from the root of the graph
  auto region = &rvsdgModule->Rvsdg().GetRootRegion();
  EXPECT_EQ(region->numNodes(), 1);
  lambda = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());

  // Assert
  auto lambdaRegion = lambda->subregion();
  EXPECT_EQ(lambdaRegion->numNodes(), 3);
  EXPECT_EQ(lambdaRegion->narguments(), 3);
  EXPECT_EQ(lambdaRegion->nresults(), 3);

  // Memory state
  EXPECT_TRUE(is<MemoryStateType>(lambdaRegion->result(1)->origin()->Type()));

  // Load Address
  auto loadNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(0)->origin())->node();
  EXPECT_TRUE(is<jlm::hls::LoadOperation>(loadNode));

  // Load Data
  loadNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(1)->origin())->node();
  EXPECT_TRUE(is<jlm::hls::LoadOperation>(loadNode));

  // Request Node
  auto requestNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(2)->origin())->node();
  EXPECT_TRUE(is<MemoryRequestOperation>(requestNode));

  // Response Node
  auto responseNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(loadNode->input(2)->origin())->node();
  EXPECT_TRUE(is<MemoryResponseOperation>(responseNode));

  // Response source
  auto responseSource = responseNode->input(0)->origin();
  auto regionArgument = jlm::util::assertedCast<jlm::rvsdg::RegionArgument>(responseSource);
  EXPECT_EQ(regionArgument->index(), 2);
}

TEST(MemoryConverterTests, TestStore)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(),
        jlm::rvsdg::BitType::Create(32),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Single load
  auto storeAddress = lambda->GetFunctionArguments()[0];
  auto storeData = lambda->GetFunctionArguments()[1];
  auto memoryStateArgument = lambda->GetFunctionArguments()[2];
  auto storeOutput =
      StoreNonVolatileOperation::Create(storeAddress, storeData, { memoryStateArgument }, 32);

  auto lambdaOutput = lambda->finalize({ storeOutput[0] });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryConverter::CreateAndRun(*rvsdgModule, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Memory Converter replaces the lambda so we start from the root of the graph
  auto region = &rvsdgModule->Rvsdg().GetRootRegion();
  EXPECT_EQ(region->numNodes(), 1);
  lambda = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());

  // Assert
  auto lambdaRegion = lambda->subregion();
  EXPECT_EQ(lambdaRegion->numNodes(), 4);
  EXPECT_EQ(lambdaRegion->narguments(), 4);
  EXPECT_EQ(lambdaRegion->nresults(), 2);

  EXPECT_TRUE(is<MemoryStateType>(lambdaRegion->result(0)->origin()->Type()));
  auto bufferNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(0)->origin())->node();
  auto storeNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(bufferNode->input(0)->origin())->node();
  EXPECT_TRUE(is<jlm::hls::StoreOperation>(storeNode));
  auto requestNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(1)->origin())->node();
  EXPECT_TRUE(is<MemoryRequestOperation>(requestNode));

  // Request source
  auto requestSource = requestNode->input(0)->origin();
  storeNode = jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(requestSource)->node();
  EXPECT_TRUE(is<jlm::hls::StoreOperation>(storeNode));
}

TEST(MemoryConverterTests, TestLoadStore)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::llvm::PointerType::Create(),
        jlm::rvsdg::BitType::Create(32),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Load followed by store
  auto loadAddress = lambda->GetFunctionArguments()[0];
  auto storeData = lambda->GetFunctionArguments()[1];
  auto memoryStateArgument = lambda->GetFunctionArguments()[2];
  auto loadOutput = LoadNonVolatileOperation::Create(
      loadAddress,
      { memoryStateArgument },
      jlm::llvm::PointerType::Create(),
      32);
  auto storeOutput =
      StoreNonVolatileOperation::Create(loadOutput[0], storeData, { loadOutput[1] }, 32);

  auto lambdaOutput = lambda->finalize({ storeOutput[0] });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryConverter::CreateAndRun(*rvsdgModule, statisticsCollector);

  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Memory Converter replaces the lambda so we start from the root of the graph
  auto region = &rvsdgModule->Rvsdg().GetRootRegion();
  EXPECT_EQ(region->numNodes(), 1);
  lambda = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());

  // Assert
  auto lambdaRegion = lambda->subregion();
  EXPECT_EQ(lambdaRegion->numNodes(), 7);
  EXPECT_EQ(lambdaRegion->narguments(), 5);
  EXPECT_EQ(lambdaRegion->nresults(), 3);

  std::cout << lambdaRegion->result(0)->origin()->Type()->debug_string() << std::endl;
  EXPECT_TRUE(is<MemoryStateType>(lambdaRegion->result(0)->origin()->Type()));
  auto bufferNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(0)->origin())->node();
  auto storeNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(bufferNode->input(0)->origin())->node();
  EXPECT_TRUE(is<jlm::hls::StoreOperation>(storeNode));
  auto firstRequestNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(1)->origin())->node();
  EXPECT_TRUE(is<MemoryRequestOperation>(firstRequestNode));
  auto secondRequestNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(2)->origin())->node();
  EXPECT_TRUE(is<MemoryRequestOperation>(secondRequestNode));
  auto loadNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(storeNode->input(0)->origin())->node();
  EXPECT_TRUE(is<jlm::hls::LoadOperation>(loadNode));
  auto responseNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(loadNode->input(2)->origin())->node();
  EXPECT_TRUE(is<MemoryResponseOperation>(responseNode));
}

TEST(MemoryConverterTests, TestThetaLoad)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32),
        jlm::rvsdg::BitType::Create(32),
        jlm::rvsdg::BitType::Create(32),
        jlm::llvm::PointerType::Create(),
        MemoryStateType::Create() },
      { jlm::llvm::PointerType::Create(), MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  // Predicate
  auto idv = theta->AddLoopVar(lambda->GetFunctionArguments()[0]);
  auto lvs = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  auto lve = theta->AddLoopVar(lambda->GetFunctionArguments()[2]);
  auto arm = jlm::rvsdg::CreateOpNode<jlm::rvsdg::bitadd_op>({ idv.pre, lvs.pre }, 32).output(0);
  auto cmp = jlm::rvsdg::CreateOpNode<jlm::rvsdg::bitult_op>({ arm, lve.pre }, 32).output(0);
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);
  idv.post->divert_to(arm);
  theta->set_predicate(match);

  // Load node
  auto loadAddress = theta->AddLoopVar(lambda->GetFunctionArguments()[3]);
  auto memoryStateArgument = theta->AddLoopVar(lambda->GetFunctionArguments()[4]);
  auto loadOutput = LoadNonVolatileOperation::Create(
      loadAddress.pre,
      { memoryStateArgument.pre },
      PointerType::Create(),
      32);
  loadAddress.post->divert_to(loadOutput[0]);
  memoryStateArgument.post->divert_to(loadOutput[1]);

  auto lambdaOutput = lambda->finalize({ theta->output(3), theta->output(4) });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryStateSeparation::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto & entryMemoryStateSplitInput = lambdaRegion->argument(4)->SingleUser();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(entryMemoryStateSplitInput);
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(entryMemoryStateSplitNode));
  auto exitMemoryStateMergeNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(1)->origin())->node();
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(exitMemoryStateMergeNode));

  // Act
  ThetaNodeConversion::CreateAndRun(*rvsdgModule, statisticsCollector);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsNodeType<LoopNode>(*lambdaRegion, true));

  // Act
  AddressQueueInsertion::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Simple assert as mem_queue() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<StateGateOperation>(*lambdaRegion, true));
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<MemoryStateSplitOperation>(*lambdaRegion, true));
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<MemoryStateMergeOperation>(*lambdaRegion, true));

  // Act
  MemoryConverter::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Memory Converter replaces the lambda so we start from the root of the graph
  auto region = &rvsdgModule->Rvsdg().GetRootRegion();
  EXPECT_EQ(region->numNodes(), 1);
  lambda = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
  lambdaRegion = lambda->subregion();

  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<MemoryResponseOperation>(*lambdaRegion, true));
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<MemoryRequestOperation>(*lambdaRegion, true));

  // Request Node
  auto requestNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(2)->origin())->node();
  EXPECT_TRUE(is<MemoryRequestOperation>(requestNode));

  // HLS_LOOP Node
  auto loopOutput =
      jlm::util::assertedCast<const jlm::rvsdg::StructuralOutput>(requestNode->input(0)->origin());
  auto loopNode = jlm::util::assertedCast<const jlm::rvsdg::StructuralNode>(loopOutput->node());
  EXPECT_NE(dynamic_cast<const LoopNode *>(loopNode), nullptr);
  // Loop Result
  auto & thetaResult = loopOutput->results;
  EXPECT_EQ(thetaResult.size(), 1);
  // Load Node
  auto loadNode =
      jlm::util::assertedCast<const jlm::rvsdg::NodeOutput>(thetaResult.first()->origin())->node();
  EXPECT_TRUE(is<DecoupledLoadOperation>(loadNode));
  // Loop Argument
  auto thetaArgument =
      jlm::util::assertedCast<const jlm::rvsdg::RegionArgument>(loadNode->input(1)->origin());
  auto thetaInput = thetaArgument->input();

  // Response Node
  auto responseNode =
      jlm::util::assertedCast<const jlm::rvsdg::NodeOutput>(thetaInput->origin())->node();
  EXPECT_TRUE(is<MemoryResponseOperation>(responseNode));

  // Lambda argument
  EXPECT_TRUE(is<jlm::rvsdg::RegionArgument>(responseNode->input(0)->origin()));
}

TEST(MemoryConverterTests, TestThetaStore)
{
  using namespace jlm::llvm;
  using namespace jlm::hls;

  auto rvsdgModule = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");

  // Setup the function
  std::cout << "Function Setup" << std::endl;
  auto functionType = jlm::rvsdg::FunctionType::Create(
      { jlm::rvsdg::BitType::Create(32),
        jlm::rvsdg::BitType::Create(32),
        jlm::rvsdg::BitType::Create(32),
        jlm::llvm::PointerType::Create(),
        jlm::rvsdg::BitType::Create(32),
        MemoryStateType::Create() },
      { MemoryStateType::Create() });

  auto lambda = jlm::rvsdg::LambdaNode::Create(
      rvsdgModule->Rvsdg().GetRootRegion(),
      LlvmLambdaOperation::Create(functionType, "test", Linkage::externalLinkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  // Predicate
  auto idv = theta->AddLoopVar(lambda->GetFunctionArguments()[0]);
  auto lvs = theta->AddLoopVar(lambda->GetFunctionArguments()[1]);
  auto lve = theta->AddLoopVar(lambda->GetFunctionArguments()[2]);
  auto arm = jlm::rvsdg::CreateOpNode<jlm::rvsdg::bitadd_op>({ idv.pre, lvs.pre }, 32).output(0);
  auto cmp = jlm::rvsdg::CreateOpNode<jlm::rvsdg::bitult_op>({ arm, lve.pre }, 32).output(0);
  auto match = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);
  idv.post->divert_to(arm);
  theta->set_predicate(match);

  // Store node
  auto storeAddress = theta->AddLoopVar(lambda->GetFunctionArguments()[3]);
  auto storeData = theta->AddLoopVar(lambda->GetFunctionArguments()[4]);
  auto memoryStateArgument = theta->AddLoopVar(lambda->GetFunctionArguments()[5]);
  auto storeOutput = StoreNonVolatileOperation::Create(
      storeAddress.pre,
      storeData.pre,
      { memoryStateArgument.pre },
      32);
  memoryStateArgument.post->divert_to(storeOutput[0]);

  auto lambdaOutput = lambda->finalize({ theta->output(5) });
  jlm::rvsdg::GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  MemoryStateSeparation::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto & entryMemoryStateSplitInput = lambdaRegion->argument(5)->SingleUser();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(entryMemoryStateSplitInput);
  EXPECT_TRUE(is<LambdaEntryMemoryStateSplitOperation>(entryMemoryStateSplitNode));
  auto exitMemoryStateMergeNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(0)->origin())->node();
  EXPECT_TRUE(is<LambdaExitMemoryStateMergeOperation>(exitMemoryStateMergeNode));

  // Act
  ThetaNodeConversion::CreateAndRun(*rvsdgModule, statisticsCollector);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(jlm::rvsdg::Region::ContainsNodeType<LoopNode>(*lambdaRegion, true));

  // Act
  AddressQueueInsertion::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Simple assert as mem_queue() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<MemoryStateSplitOperation>(*lambdaRegion, true));
  EXPECT_TRUE(
      jlm::rvsdg::Region::ContainsOperation<MemoryStateMergeOperation>(*lambdaRegion, true));

  // Act
  MemoryConverter::CreateAndRun(*rvsdgModule, statisticsCollector);

  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Memory Converter replaces the lambda so we start from the root of the graph
  auto region = &rvsdgModule->Rvsdg().GetRootRegion();
  EXPECT_EQ(region->numNodes(), 1);
  lambda = jlm::util::assertedCast<jlm::rvsdg::LambdaNode>(region->Nodes().begin().ptr());
  lambdaRegion = lambda->subregion();

  EXPECT_TRUE(jlm::rvsdg::Region::ContainsOperation<MemoryRequestOperation>(*lambdaRegion, true));

  // Request Node
  auto requestNode =
      jlm::util::assertedCast<jlm::rvsdg::NodeOutput>(lambdaRegion->result(1)->origin())->node();
  EXPECT_TRUE(is<MemoryRequestOperation>(requestNode));

  // HLS_LOOP Node
  auto loopOutput =
      jlm::util::assertedCast<const jlm::rvsdg::StructuralOutput>(requestNode->input(0)->origin());
  auto loopNode = jlm::util::assertedCast<const jlm::rvsdg::StructuralNode>(loopOutput->node());
  EXPECT_NE(dynamic_cast<const LoopNode *>(loopNode), nullptr);
  // Loop Result
  auto & thetaResult = loopOutput->results;
  EXPECT_EQ(thetaResult.size(), 1);
  // Load Node
  auto storeNode =
      jlm::util::assertedCast<const jlm::rvsdg::NodeOutput>(thetaResult.first()->origin())->node();
  EXPECT_TRUE(is<jlm::hls::StoreOperation>(storeNode));
  // NDMux Node
  auto ndMuxNode =
      jlm::util::assertedCast<const jlm::rvsdg::NodeOutput>(storeNode->input(2)->origin())->node();
  EXPECT_TRUE(is<MuxOperation>(ndMuxNode));
  // Loop Argument
  EXPECT_TRUE(is<jlm::rvsdg::RegionArgument>(ndMuxNode->input(2)->origin()));
}
