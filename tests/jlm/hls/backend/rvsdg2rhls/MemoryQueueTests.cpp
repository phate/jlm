/*
 * Copyright 2024 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/hls/backend/rvsdg2rhls/mem-queue.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>

static int
TestSingleLoad()
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
      LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto constant = jlm::rvsdg::create_bitconstant(theta->subregion(), 1, 1);
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
  GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  mem_sep_argument(*rvsdgModule);
  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto * const entryMemoryStateSplitInput = *lambdaRegion->argument(1)->begin();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*entryMemoryStateSplitInput);
  jlm::util::AssertedCast<const LambdaEntryMemoryStateSplitOperation>(
      &entryMemoryStateSplitNode->GetOperation());
  auto exitMemoryStateMergeNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*lambdaRegion->result(1)->origin());
  jlm::util::AssertedCast<const LambdaExitMemoryStateMergeOperation>(
      &exitMemoryStateMergeNode->GetOperation());

  // Act
  ConvertThetaNodes(*rvsdgModule);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(jlm::rvsdg::Region::ContainsNodeType<loop_node>(*lambdaRegion, true));

  // Act
  mem_queue(*rvsdgModule);
  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(jlm::rvsdg::Region::ContainsOperation<StateGateOperation>(*lambdaRegion, true));
  assert(!jlm::rvsdg::Region::ContainsOperation<addr_queue_op>(*lambdaRegion, true));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/MemoryQueueTests-SingleLoad", TestSingleLoad)

static int
TestLoadStore()
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
      LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto constant = jlm::rvsdg::create_bitconstant(theta->subregion(), 1, 1);
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
      jlm::rvsdg::create_bitconstant(theta->subregion(), 32, 1),
      { loadOutput[1] },
      32);

  loadAddress.post->divert_to(loadOutput[0]);
  memoryStateArgument.post->divert_to(storeOutput[0]);

  auto lambdaOutput = lambda->finalize({ theta->output(0), theta->output(2) });
  GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  mem_sep_argument(*rvsdgModule);
  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto * const entryMemoryStateSplitInput = *lambdaRegion->argument(2)->begin();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*entryMemoryStateSplitInput);
  jlm::util::AssertedCast<const LambdaEntryMemoryStateSplitOperation>(
      &entryMemoryStateSplitNode->GetOperation());
  auto exitMemoryStateMergeNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*lambdaRegion->result(1)->origin());
  jlm::util::AssertedCast<const LambdaExitMemoryStateMergeOperation>(
      &exitMemoryStateMergeNode->GetOperation());

  // Act
  ConvertThetaNodes(*rvsdgModule);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(jlm::rvsdg::Region::ContainsNodeType<loop_node>(*lambdaRegion, true));

  // Act
  mem_queue(*rvsdgModule);
  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(jlm::rvsdg::Region::ContainsOperation<StateGateOperation>(*lambdaRegion, true));
  assert(!jlm::rvsdg::Region::ContainsOperation<addr_queue_op>(*lambdaRegion, true));

  return 0;
}
JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/MemoryQueueTests-LoadStore", TestLoadStore)

static int
TestAddrQueue()
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
      LlvmLambdaOperation::Create(functionType, "test", linkage::external_linkage));

  // Theta
  auto theta = jlm::rvsdg::ThetaNode::create(lambda->subregion());
  auto constant = jlm::rvsdg::create_bitconstant(theta->subregion(), 1, 1);
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
  GraphExport::Create(*lambdaOutput, "f");

  auto lambdaRegion = lambda->subregion();
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);

  // Act
  mem_sep_argument(*rvsdgModule);
  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  auto * const entryMemoryStateSplitInput = *lambdaRegion->argument(1)->begin();
  auto * entryMemoryStateSplitNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*entryMemoryStateSplitInput);
  jlm::util::AssertedCast<const LambdaEntryMemoryStateSplitOperation>(
      &entryMemoryStateSplitNode->GetOperation());
  auto exitMemoryStateMergeNode =
      jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*lambdaRegion->result(1)->origin());
  jlm::util::AssertedCast<const LambdaExitMemoryStateMergeOperation>(
      &exitMemoryStateMergeNode->GetOperation());

  // Act
  ConvertThetaNodes(*rvsdgModule);
  // Simple assert as ConvertThetaNodes() is tested in separate unit tests
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(jlm::rvsdg::Region::ContainsNodeType<loop_node>(*lambdaRegion, true));

  // Act
  mem_queue(*rvsdgModule);
  // Assert
  jlm::rvsdg::view(rvsdgModule->Rvsdg(), stdout);
  assert(jlm::rvsdg::Region::ContainsOperation<StateGateOperation>(*lambdaRegion, true));
  assert(jlm::rvsdg::Region::ContainsOperation<addr_queue_op>(*lambdaRegion, true));

  for (auto & node : jlm::rvsdg::TopDownTraverser(lambdaRegion))
  {
    if (auto loopNode = dynamic_cast<jlm::hls::loop_node *>(node))
    {
      for (auto & node : jlm::rvsdg::TopDownTraverser(loopNode->subregion()))
      {
        if (is<StoreNonVolatileOperation>(node))
        {
          auto loadNode =
              jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*node->input(1)->origin());
          jlm::util::AssertedCast<const LoadOperation>(&loadNode->GetOperation());
          auto stateGate =
              jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*loadNode->input(0)->origin());
          jlm::util::AssertedCast<const StateGateOperation>(&stateGate->GetOperation());
          auto addrQueue =
              jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::SimpleNode>(*stateGate->input(0)->origin());
          jlm::util::AssertedCast<const addr_queue_op>(&addrQueue->GetOperation());
          return 0;
        }
      }
    }
  }

  return 1;
}
JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/MemoryQueueTests-AddrQueue", TestAddrQueue)
