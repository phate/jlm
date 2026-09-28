/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/TestRvsdgs.hpp>
#include <jlm/mlir/TestRvsdgs.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/lambda.hpp>

namespace jlm::mlir
{

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
LoadNonVolatileTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto fcttype = rvsdg::FunctionType::Create(
      { MemoryStateType::Create(), PointerType::Create() },
      { jlm::rvsdg::BitType::Create(32), MemoryStateType::Create() });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", Linkage::externalLinkage));
  auto memoryStateArgument = fct->GetFunctionArguments()[0];
  auto pointerArgument = fct->GetFunctionArguments()[1];

  auto load = LoadNonVolatileOperation::Create(
      pointerArgument,
      { memoryStateArgument },
      BitType::Create(32),
      4);

  fct->finalize({ load[0], load[1] });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->load = rvsdg::TryGetOwnerNode<SimpleNode>(*load[0]);

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
LoadVolatileTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto fcttype = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create(), PointerType::Create() },
      { jlm::rvsdg::BitType::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", Linkage::externalLinkage));
  auto iOStateArgument = fct->GetFunctionArguments()[0];
  auto memoryStateArgument = fct->GetFunctionArguments()[1];
  auto pointerArgument = fct->GetFunctionArguments()[2];

  auto & loadNode = LoadVolatileOperation::CreateNode(
      *pointerArgument,
      *iOStateArgument,
      { memoryStateArgument },
      BitType::Create(32),
      4);

  fct->finalize({ loadNode.output(0), loadNode.output(1), loadNode.output(2) });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->load = &loadNode;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
StoreNonVolatileTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto fcttype = rvsdg::FunctionType::Create(
      { MemoryStateType::Create(), PointerType::Create(), jlm::rvsdg::BitType::Create(32) },
      { MemoryStateType::Create() });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", Linkage::externalLinkage));
  auto memoryStateArgument = fct->GetFunctionArguments()[0];
  auto pointerArgument = fct->GetFunctionArguments()[1];
  auto valueArgument = fct->GetFunctionArguments()[2];

  auto store =
      StoreNonVolatileOperation::Create(pointerArgument, valueArgument, { memoryStateArgument }, 4);

  fct->finalize({ store[0] });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->store = rvsdg::TryGetOwnerNode<SimpleNode>(*store[0]);

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
StoreVolatileTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto fcttype = rvsdg::FunctionType::Create(
      { IOStateType::Create(),
        MemoryStateType::Create(),
        PointerType::Create(),
        jlm::rvsdg::BitType::Create(32) },
      { IOStateType::Create(), MemoryStateType::Create() });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", Linkage::externalLinkage));
  auto iOStateArgument = fct->GetFunctionArguments()[0];
  auto memoryStateArgument = fct->GetFunctionArguments()[1];
  auto pointerArgument = fct->GetFunctionArguments()[2];
  auto valueArgument = fct->GetFunctionArguments()[3];

  auto & storeNode = StoreVolatileOperation::CreateNode(
      *pointerArgument,
      *valueArgument,
      *iOStateArgument,
      { memoryStateArgument },
      4);

  fct->finalize({ storeNode.output(0), storeNode.output(1) });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->store = &storeNode;

  return module;
}

}
