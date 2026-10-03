/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
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

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
MemoryHoistBarrierTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto fcttype = rvsdg::FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create(), PointerType::Create() },
      { BitType::Create(32), IOStateType::Create(), MemoryStateType::Create() });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(fcttype, "f", Linkage::externalLinkage));
  auto iOStateArgument = fct->GetFunctionArguments()[0];
  auto memoryStateArgument = fct->GetFunctionArguments()[1];
  auto pointerArgument = fct->GetFunctionArguments()[2];

  // A non-volatile load, but one whose address is sequentialized by a memory hoist barrier.
  auto & memoryHoistBarrierNode =
      MemoryHoistBarrierOperation::createNode(*pointerArgument, *iOStateArgument, 4);

  auto load = LoadNonVolatileOperation::Create(
      memoryHoistBarrierNode.output(0),
      { memoryStateArgument },
      BitType::Create(32),
      4);

  fct->finalize({ load[0], iOStateArgument, load[1] });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->memoryHoistBarrier = &memoryHoistBarrierNode;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
IntegerConversionTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType64 = BitType::Create(64);
  auto bitType32 = BitType::Create(32);
  auto functionType = FunctionType::Create({ bitType32 }, { bitType32 });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto valueArgument = fct->GetFunctionArguments()[0];

  auto & sextNode = SExtOperation::createNode(64, *valueArgument);
  auto & truncNode = TruncOperation::createNode(*sextNode.output(0), bitType32);

  fct->finalize({ truncNode.output(0) });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->sext = &sextNode;
  this->trunc = &truncNode;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
FloatBinaryTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto floatType = FloatingPointType::Create(fpsize::dbl);
  auto functionType =
      FunctionType::Create({ floatType, floatType }, { floatType, floatType, floatType });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto lhsArgument = fct->GetFunctionArguments()[0];
  auto rhsArgument = fct->GetFunctionArguments()[1];

  auto & sum = CreateOpNode<FBinaryOperation>({ lhsArgument, rhsArgument }, fpop::add, floatType);
  auto & difference =
      CreateOpNode<FBinaryOperation>({ lhsArgument, rhsArgument }, fpop::sub, floatType);
  auto & product =
      CreateOpNode<FBinaryOperation>({ lhsArgument, rhsArgument }, fpop::mul, floatType);
  auto & quotient =
      CreateOpNode<FBinaryOperation>({ sum.output(0), difference.output(0) }, fpop::div, floatType);
  auto & remainder =
      CreateOpNode<FBinaryOperation>({ product.output(0), sum.output(0) }, fpop::mod, floatType);

  auto & first =
      CreateOpNode<FBinaryOperation>({ sum.output(0), difference.output(0) }, fpop::add, floatType);
  auto & second = CreateOpNode<FBinaryOperation>(
      { product.output(0), quotient.output(0) },
      fpop::add,
      floatType);
  auto & third = CreateOpNode<FBinaryOperation>(
      { second.output(0), remainder.output(0) },
      fpop::add,
      floatType);

  fct->finalize({ first.output(0), second.output(0), third.output(0) });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
FloatConversionTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto doubleType = FloatingPointType::Create(fpsize::dbl);
  auto floatType = FloatingPointType::Create(fpsize::flt);
  auto bitType64 = BitType::Create(64);
  auto functionType =
      FunctionType::Create({ doubleType, floatType, bitType64 }, { doubleType, doubleType });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto doubleArgument = fct->GetFunctionArguments()[0];
  auto floatArgument = fct->GetFunctionArguments()[1];
  auto integerArgument = fct->GetFunctionArguments()[2];

  auto & constant = CreateOpNode<ConstantFP>(*fct->subregion(), fpsize::dbl, ::llvm::APFloat(2.0));

  auto & extended = CreateOpNode<FPExtOperation>({ floatArgument }, floatType, doubleType);

  auto & sum = CreateOpNode<FBinaryOperation>(
      { extended.output(0), constant.output(0) },
      fpop::add,
      doubleType);

  auto & negated = CreateOpNode<FNegOperation>({ sum.output(0) }, doubleType);

  auto & converted = CreateOpNode<SIToFPOperation>({ integerArgument }, bitType64, doubleType);

  auto & fused = CreateOpNode<FMulAddIntrinsicOperation>(
      { doubleArgument, doubleArgument, converted.output(0) },
      doubleType);

  auto & first = CreateOpNode<FBinaryOperation>(
      { negated.output(0), converted.output(0) },
      fpop::add,
      doubleType);
  auto & second = CreateOpNode<FBinaryOperation>(
      { converted.output(0), fused.output(0) },
      fpop::add,
      doubleType);

  fct->finalize({ first.output(0), second.output(0) });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
ComparisonTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto doubleType = FloatingPointType::Create(fpsize::dbl);
  auto pointerType = PointerType::Create();
  auto bitType32 = BitType::Create(32);
  auto functionType = FunctionType::Create(
      { doubleType, doubleType, pointerType, pointerType },
      { bitType32, bitType32 });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto lhsArgument = fct->GetFunctionArguments()[0];
  auto rhsArgument = fct->GetFunctionArguments()[1];
  auto lhsPointerArgument = fct->GetFunctionArguments()[2];
  auto rhsPointerArgument = fct->GetFunctionArguments()[3];

  auto & floatCompareNode =
      CreateOpNode<FCmpOperation>({ lhsArgument, rhsArgument }, fpcmp::olt, fpsize::dbl);

  auto zero = &IntegerConstantOperation::Create(*fct->subregion(), { 32, 0 });
  auto one = &IntegerConstantOperation::Create(*fct->subregion(), { 32, 1 });

  auto & firstSelect = CreateOpNode<SelectOperation>(
      { floatCompareNode.output(0), one->output(0), zero->output(0) },
      bitType32);

  auto & pointerCompareNode = CreateOpNode<PtrCmpOperation>(
      { lhsPointerArgument, rhsPointerArgument },
      pointerType,
      ICmpPredicate::Slt);

  auto & secondSelect = CreateOpNode<SelectOperation>(
      { pointerCompareNode.output(0), one->output(0), zero->output(0) },
      bitType32);

  fct->finalize({ firstSelect.output(0), secondSelect.output(0) });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->fcmp = &floatCompareNode;
  this->ptrcmp = &pointerCompareNode;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
IOBarrierTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);
  auto functionType = FunctionType::Create(
      { IOStateType::Create(), bitType32 },
      { IOStateType::Create(), bitType32 });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto iOStateArgument = fct->GetFunctionArguments()[0];
  auto valueArgument = fct->GetFunctionArguments()[1];

  auto & barrierNode = IOBarrierOperation::createNode(*valueArgument, *iOStateArgument);

  auto constant = &IntegerConstantOperation::Create(*fct->subregion(), { 32, 1 });
  auto & addNode =
      IntegerAddOperation::createNode(32, *barrierNode.output(0), *constant->output(0));

  fct->finalize({ iOStateArgument, addNode.output(0) });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->barrier = &barrierNode;
  this->add = rvsdg::TryGetOwnerNode<SimpleNode>(*addNode.output(0));

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
ConstantDeltaTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto mutableDelta = DeltaNode::Create(
      &graph->GetRootRegion(),
      LlvmDeltaOperation::Create(
          bitType32,
          "mutableGlobal",
          Linkage::externalLinkage,
          "mysection",
          false,
          4));
  auto mutableConstant =
      IntegerConstantOperation::Create(*mutableDelta->subregion(), 32, 1).output(0);
  auto & mutableOutput = mutableDelta->finalize(mutableConstant);
  GraphExport::Create(mutableOutput, "mutableGlobal");

  auto constantDelta = DeltaNode::Create(
      &graph->GetRootRegion(),
      LlvmDeltaOperation::Create(
          bitType32,
          "constantGlobal",
          Linkage::externalLinkage,
          "mysection",
          true,
          4));
  auto constantConstant =
      IntegerConstantOperation::Create(*constantDelta->subregion(), 32, 1).output(0);
  auto & constantOutput = constantDelta->finalize(constantConstant);
  GraphExport::Create(constantOutput, "constantGlobal");

  this->mutableDelta = mutableDelta;
  this->constantDelta = constantDelta;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
GlobalArrayTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType32 = BitType::Create(32);
  auto arrayType = ArrayType::Create(bitType32, 2);

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto initDelta = DeltaNode::Create(
      &graph->GetRootRegion(),
      LlvmDeltaOperation::Create(arrayType, "initArray", Linkage::externalLinkage, "", false, 4));
  auto one = &IntegerConstantOperation::Create(*initDelta->subregion(), { 32, 1 });
  auto two = &IntegerConstantOperation::Create(*initDelta->subregion(), { 32, 2 });
  auto constantDataArray = ConstantDataArrayOperation::Create({ one->output(0), two->output(0) });
  auto & initOutput = initDelta->finalize(constantDataArray);
  GraphExport::Create(initOutput, "initArray");

  auto zeroDelta = DeltaNode::Create(
      &graph->GetRootRegion(),
      LlvmDeltaOperation::Create(arrayType, "zeroArray", Linkage::externalLinkage, "", false, 4));
  auto constantAggregateZero =
      ConstantAggregateZeroOperation::Create(*zeroDelta->subregion(), arrayType);
  auto & zeroOutput = zeroDelta->finalize(constantAggregateZero);
  GraphExport::Create(zeroOutput, "zeroArray");

  this->initDelta = initDelta;
  this->zeroDelta = zeroDelta;
  this->constantDataArray = rvsdg::TryGetOwnerNode<SimpleNode>(*constantDataArray);
  this->constantAggregateZero = rvsdg::TryGetOwnerNode<SimpleNode>(*constantAggregateZero);

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
WideMemoryNodesTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto bitType64 = BitType::Create(64);
  auto pointerType = PointerType::Create();
  auto functionType = FunctionType::Create(
      { IOStateType::Create(), MemoryStateType::Create() },
      { pointerType, pointerType, IOStateType::Create(), MemoryStateType::Create() });

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto fct = rvsdg::LambdaNode::Create(
      graph->GetRootRegion(),
      llvm::LlvmLambdaOperation::Create(functionType, "f", Linkage::externalLinkage));
  auto iOStateArgument = fct->GetFunctionArguments()[0];
  auto memoryStateArgument = fct->GetFunctionArguments()[1];

  auto size = &IntegerConstantOperation::Create(*fct->subregion(), { 64, 128 });
  auto & mallocNode = MallocOperation::createNode(*size->output(0), *iOStateArgument);

  auto count = &IntegerConstantOperation::Create(*fct->subregion(), { 32, 1 });
  auto & allocaNode = AllocaOperation::createNode(bitType64, *count->output(0), 8);

  auto memoryState = MemoryStateMergeOperation::Create(
      std::vector<jlm::rvsdg::Output *>{ memoryStateArgument,
                                         &MallocOperation::memoryStateOutput(mallocNode),
                                         &AllocaOperation::getMemoryStateOutput(allocaNode) });

  fct->finalize({ &MallocOperation::addressOutput(mallocNode),
                  allocaNode.output(0),
                  &MallocOperation::ioStateOutput(mallocNode),
                  memoryState });

  GraphExport::Create(*fct->output(), "f");

  this->lambda = fct;
  this->malloc = &mallocNode;
  this->alloca = &allocaNode;

  return module;
}

std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
RootRegionNodesTest::SetupRvsdg()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  auto module = LlvmRvsdgModule::Create(jlm::util::FilePath(""), "", "");
  auto graph = &module->Rvsdg();

  auto constant = &IntegerConstantOperation::Create(graph->GetRootRegion(), { 64, 16 });
  GraphExport::Create(*constant->output(0), "constant");

  auto first = &IntegerConstantOperation::Create(graph->GetRootRegion(), { 64, 1 });
  auto second = &IntegerConstantOperation::Create(graph->GetRootRegion(), { 64, 2 });
  auto variadicArgumentList = VariadicArgumentListOperation::Create(
      graph->GetRootRegion(),
      { first->output(0), second->output(0) });
  GraphExport::Create(*variadicArgumentList, "argumentList");

  this->constant = rvsdg::TryGetOwnerNode<SimpleNode>(*constant->output(0));
  this->variadicArgumentList = rvsdg::TryGetOwnerNode<SimpleNode>(*variadicArgumentList);

  return module;
}

}
