/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/CallingConvention.hpp>
#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/ipgraph.hpp>
#include <jlm/llvm/ir/operators/AggregateOperations.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

TEST(LlvmModuleConversionTests, SwitchConversion)
{
  using namespace llvm;

  // Arrange
  LLVMContext context;
  const std::unique_ptr<Module> llvmModule(new Module("module", context));

  auto int64Type = Type::getInt64Ty(context);

  auto functionType = FunctionType::get(int64Type, ArrayRef<Type *>({ int64Type }), false);
  auto function =
      Function::Create(functionType, GlobalValue::ExternalLinkage, "f", llvmModule.get());

  auto bbSplit = BasicBlock::Create(context, "BasicBlockSplit", function);
  auto bb1 = BasicBlock::Create(context, "BasicBlock1", function);
  auto bb2 = BasicBlock::Create(context, "BasicBlock2", function);
  auto bb3 = BasicBlock::Create(context, "BasicBlock4", function);
  auto bb4 = BasicBlock::Create(context, "BasicBlock4", function);
  auto bbJoin = BasicBlock::Create(context, "BasicBlockJoin", function);

  IRBuilder builder(bbSplit);
  auto switchInstruction = builder.CreateSwitch(function->arg_begin(), bb4);
  switchInstruction->addCase(ConstantInt::get(int64Type, 1), bb1);
  switchInstruction->addCase(ConstantInt::get(int64Type, 2), bb2);
  switchInstruction->addCase(ConstantInt::get(int64Type, 3), bb2);
  switchInstruction->addCase(ConstantInt::get(int64Type, 4), bb3);
  switchInstruction->addCase(ConstantInt::get(int64Type, 5), bb3);

  builder.SetInsertPoint(bb1);
  builder.CreateBr(bbJoin);

  builder.SetInsertPoint(bb2);
  builder.CreateBr(bbJoin);

  builder.SetInsertPoint(bb3);
  builder.CreateBr(bbJoin);

  builder.SetInsertPoint(bb4);
  builder.CreateBr(bbJoin);

  builder.SetInsertPoint(bbJoin);
  auto phiInstruction = builder.CreatePHI(int64Type, 4);
  phiInstruction->addIncoming(ConstantInt::get(int64Type, 1), bb1);
  phiInstruction->addIncoming(ConstantInt::get(int64Type, 2), bb2);
  phiInstruction->addIncoming(ConstantInt::get(int64Type, 3), bb3);
  phiInstruction->addIncoming(ConstantInt::get(int64Type, 4), bb4);
  builder.CreateRet(phiInstruction);

  llvmModule->print(errs(), nullptr);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(*llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    using namespace jlm::llvm;

    const auto controlFlowGraph =
        dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("f"))->cfg();

    EXPECT_EQ(controlFlowGraph->nnodes(), 6u);

    // We expect the split node to only have 4 outgoing edges. One for each target basic block of
    // the original LLVM switch statement
    const auto splitNode = controlFlowGraph->entry()->OutEdge(0)->sink();
    EXPECT_EQ(splitNode->NumOutEdges(), 4u);
  }
}

TEST(LlvmModuleConversionTests, FreezeConversion)
{
  /**
   * Tests that freeze instructions in LLVM are converted into FreezeOperations in RVSDG,
   * with the correct operand and type.
   */
  // Arrange
  ::llvm::LLVMContext context;
  ::llvm::Module llvmModule("module", context);

  auto int64Type = ::llvm::Type::getInt64Ty(context);
  auto functionType =
      ::llvm::FunctionType::get(int64Type, ::llvm::ArrayRef<::llvm::Type *>({ int64Type }), false);
  auto function = ::llvm::Function::Create(
      functionType,
      ::llvm::GlobalValue::ExternalLinkage,
      "f",
      &llvmModule);

  auto basicBlock = ::llvm::BasicBlock::Create(context, "BasicBlock", function);

  ::llvm::IRBuilder<> builder(basicBlock);
  auto freezeInstruction = builder.CreateFreeze(function->arg_begin());
  builder.CreateRet(freezeInstruction);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    using namespace jlm::llvm;

    const auto jlmInt64Type = jlm::rvsdg::BitType::Create(64);
    const auto controlFlowGraph =
        dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("f"))->cfg();
    const auto convertedBasicBlock =
        dynamic_cast<const jlm::llvm::BasicBlock *>(controlFlowGraph->entry()->OutEdge(0)->sink());

    size_t numFreezeThreeAddressCodes = 0;
    for (auto tac : *convertedBasicBlock)
    {
      if (!is<FreezeOperation>(tac))
        continue;

      numFreezeThreeAddressCodes++;
      const auto freezeOperation =
          jlm::util::assertedCast<const FreezeOperation>(&tac->operation());

      EXPECT_EQ(tac->noperands(), 1u);
      EXPECT_EQ(tac->nresults(), 1u);
      EXPECT_EQ(tac->operand(0), controlFlowGraph->entry()->argument(0));
      EXPECT_EQ(*freezeOperation->argument(0), *jlmInt64Type);
      EXPECT_EQ(freezeOperation->getType(), *jlmInt64Type);
      EXPECT_EQ(*freezeOperation->result(0), *jlmInt64Type);
    }

    EXPECT_EQ(numFreezeThreeAddressCodes, 1);
  }
}

/**
 * Tests that LLVM's insertvalue instructions are converted to correct RVSDG InsertValueOperations.
 */
TEST(LlvmModuleConversionTests, InsertValueConversion)
{

  // Arrange
  llvm::LLVMContext context;
  llvm::Module llvmModule("module", context);

  auto int64Type = llvm::Type::getInt64Ty(context);
  auto structType = llvm::StructType::get(int64Type, int64Type);
  auto functionType = llvm::FunctionType::get(
      structType,
      ::llvm::ArrayRef<llvm::Type *>({ int64Type, int64Type }),
      false);
  auto function =
      llvm::Function::Create(functionType, llvm::GlobalValue::ExternalLinkage, "f", &llvmModule);

  auto basicBlock = llvm::BasicBlock::Create(context, "BasicBlock", function);

  llvm::IRBuilder builder(basicBlock);
  auto poison = llvm::PoisonValue::get(structType);
  auto insertValue0 = builder.CreateInsertValue(poison, function->arg_begin(), 0);
  auto insertValue1 = builder.CreateInsertValue(insertValue0, function->arg_begin() + 1, 1);
  builder.CreateRet(insertValue1);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    using namespace jlm::llvm;

    const auto jlmInt64Type = jlm::rvsdg::BitType::Create(64);
    const auto controlFlowGraph =
        dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("f"))->cfg();
    const auto convertedBasicBlock =
        dynamic_cast<const BasicBlock *>(controlFlowGraph->entry()->OutEdge(0)->sink());

    size_t numInsertValueAddressCodes = 0;
    for (auto tac : *convertedBasicBlock)
    {
      if (auto insertValueOperation = dynamic_cast<const InsertValueOperation *>(&tac->operation()))
      {
        EXPECT_EQ(tac->noperands(), 2u);
        EXPECT_EQ(tac->nresults(), 1u);
        EXPECT_EQ(insertValueOperation->getIndices().size(), 1u);
        EXPECT_EQ(*insertValueOperation->getValueType(), *jlmInt64Type);

        numInsertValueAddressCodes++;
        if (numInsertValueAddressCodes == 1)
        {
          EXPECT_EQ(tac->operand(1), controlFlowGraph->entry()->argument(0));
          EXPECT_EQ(insertValueOperation->getIndices()[0], 0u);
        }
        else if (numInsertValueAddressCodes == 2)
        {
          EXPECT_EQ(tac->operand(1), controlFlowGraph->entry()->argument(1));
          EXPECT_EQ(insertValueOperation->getIndices()[0], 1u);
        }
      }
    }

    EXPECT_EQ(numInsertValueAddressCodes, 2u);
  }
}

TEST(LlvmModuleConversionTests, CallingConvConversion)
{
  /**
   * Tests that function declarations, function definitions and function calls in LLVM
   * are converted into corresponding InterProceduralGraphModule functions and call operations
   * with the same calling convention.
   *
   * The LLVM looks like:
   *
   *     declare i64 fastcc @imported(i64);
   *
   *     define i64 coldcc @callee(i64 %1) {
   *         ret i64 %1
   *     }
   *
   *     define i64 tailcc @caller(i64 %1) {
   *         %2 = call fastcc i64 @imported(i64 %1)
   *         %3 = call coldcc @callee(i64 %2)
   *         ret i64 %3
   *     }
   *
   * After conversion, the calling convention of the functions and calls are checked.
   */

  // Arrange
  ::llvm::LLVMContext context;
  ::llvm::Module llvmModule("module", context);
  auto int64Type = ::llvm::Type::getInt64Ty(context);
  auto unaryFunctionType =
      ::llvm::FunctionType::get(int64Type, ::llvm::ArrayRef<::llvm::Type *>({ int64Type }), false);

  auto importedFunction = ::llvm::Function::Create(
      unaryFunctionType,
      ::llvm::GlobalValue::ExternalLinkage,
      "imported",
      &llvmModule);
  importedFunction->setCallingConv(::llvm::CallingConv::Fast);

  auto callee = ::llvm::Function::Create(
      unaryFunctionType,
      ::llvm::GlobalValue::ExternalLinkage,
      "callee",
      &llvmModule);
  callee->setCallingConv(::llvm::CallingConv::Cold);

  // Create the function body of the callee function
  {
    auto basicBlock = ::llvm::BasicBlock::Create(context, "BasicBlock", callee);
    ::llvm::IRBuilder<> builder(basicBlock);
    builder.CreateRet(callee->arg_begin());
  }

  auto caller = ::llvm::Function::Create(
      unaryFunctionType,
      ::llvm::GlobalValue::ExternalLinkage,
      "caller",
      &llvmModule);
  caller->setCallingConv(::llvm::CallingConv::Tail);

  // Create the function body of the caller function
  {
    auto basicBlock = ::llvm::BasicBlock::Create(context, "BasicBlock", caller);
    ::llvm::IRBuilder<> builder(basicBlock);
    auto importedCall = builder.CreateCall(importedFunction, { caller->arg_begin() });
    importedCall->setCallingConv(::llvm::CallingConv::Fast);
    auto calleeCall = builder.CreateCall(callee, { importedCall });
    calleeCall->setCallingConv(::llvm::CallingConv::Cold);
    builder.CreateRet(calleeCall);
  }

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(llvmModule);

  // Assert
  {
    using namespace jlm::llvm;

    const auto importedNode =
        dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("imported"));
    const auto calleeNode = dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("callee"));
    const auto callerNode = dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("caller"));

    std::cout << ControlFlowGraph::ToAscii(*callerNode->cfg()) << std::endl;

    ASSERT_NE(importedNode, nullptr);
    ASSERT_NE(calleeNode, nullptr);
    ASSERT_NE(callerNode, nullptr);

    EXPECT_FALSE(importedNode->hasBody());
    EXPECT_TRUE(calleeNode->hasBody());
    EXPECT_TRUE(callerNode->hasBody());

    EXPECT_EQ(importedNode->callingConvention(), CallingConvention::Fast);
    EXPECT_EQ(calleeNode->callingConvention(), CallingConvention::Cold);
    EXPECT_EQ(callerNode->callingConvention(), CallingConvention::Tail);

    const auto convertedBasicBlock =
        dynamic_cast<const BasicBlock *>(callerNode->cfg()->entry()->OutEdge(0)->sink());
    ASSERT_NE(convertedBasicBlock, nullptr);
    auto it = convertedBasicBlock->begin();

    // Return the next CallOperation in the basic block
    const auto nextCallTac = [&]()
    {
      while (true)
      {
        EXPECT_NE(it, convertedBasicBlock->end());
        if (is<CallOperation>(*it))
          return *it++;
        it++;
      }
    };

    // Check that the call to imported has been converted correctly
    {
      auto callImportedTac = nextCallTac();
      EXPECT_EQ(callImportedTac->operand(0), ipgModule->variable(importedNode));
      auto op = jlm::util::assertedCast<const CallOperation>(&callImportedTac->operation());
      EXPECT_EQ(op->getCallingConvention(), CallingConvention::Fast);
    }

    // Check that the call to callee has been converted correctly
    {
      auto callCalleeTac = nextCallTac();
      EXPECT_EQ(callCalleeTac->operand(0), ipgModule->variable(calleeNode));
      auto op = jlm::util::assertedCast<const CallOperation>(&callCalleeTac->operation());
      EXPECT_EQ(op->getCallingConvention(), CallingConvention::Cold);
    }
  }
}

TEST(LlvmModuleConversionTests, MemCpyConversion)
{
  using namespace llvm;

  // Arrange
  LLVMContext context;
  std::unique_ptr<Module> llvmModule(new Module("module", context));

  auto int64Type = Type::getInt64Ty(context);
  auto pointerType = PointerType::getUnqual(context);
  auto voidType = Type::getVoidTy(context);

  auto functionType =
      FunctionType::get(voidType, ArrayRef<Type *>({ pointerType, pointerType, int64Type }), false);
  auto function =
      Function::Create(functionType, GlobalValue::ExternalLinkage, "f", llvmModule.get());
  auto destination = function->getArg(0);
  auto source = function->getArg(1);
  auto length = function->getArg(2);

  auto llvmBasicBlock = BasicBlock::Create(context, "BasicBlock", function);

  IRBuilder<> builder(llvmBasicBlock);
  builder.CreateMemCpy(destination, MaybeAlign(), source, MaybeAlign(), length, true);
  builder.CreateMemCpy(destination, MaybeAlign(), source, MaybeAlign(), length, false);
  builder.CreateMemCpy(destination, MaybeAlign(), source, MaybeAlign(), length, true);
  builder.CreateRetVoid();

  llvmModule->print(errs(), nullptr);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(*llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    using namespace jlm::llvm;

    auto controlFlowGraph =
        dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("f"))->cfg();
    auto jlmBasicBlock =
        dynamic_cast<const jlm::llvm::BasicBlock *>(controlFlowGraph->entry()->OutEdge(0)->sink());

    size_t numMemCpyThreeAddressCodes = 0;
    size_t numMemCpyVolatileThreeAddressCodes = 0;
    for (auto it = jlmBasicBlock->begin(); it != jlmBasicBlock->end(); ++it)
    {
      if (is<MemCpyVolatileOperation>(*it))
      {
        numMemCpyVolatileThreeAddressCodes++;
        auto ioStateAssignment = *std::next(it);
        auto memoryStateAssignment = *std::next(it, 2);

        EXPECT_TRUE(is<AssignmentOperation>(ioStateAssignment->operation()));
        EXPECT_TRUE(is<IOStateType>(ioStateAssignment->operand(0)->type()));

        EXPECT_TRUE(is<AssignmentOperation>(memoryStateAssignment->operation()));
        EXPECT_TRUE(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
      else if (is<MemCpyNonVolatileOperation>(*it))
      {
        numMemCpyThreeAddressCodes++;
        auto memoryStateAssignment = *std::next(it, 1);

        EXPECT_TRUE(is<AssignmentOperation>(memoryStateAssignment->operation()));
        EXPECT_TRUE(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
    }

    EXPECT_EQ(numMemCpyThreeAddressCodes, 1u);
    EXPECT_EQ(numMemCpyVolatileThreeAddressCodes, 2u);
  }
}

TEST(LlvmModuleConversionTests, MemSetConversion)
{
  using namespace llvm;

  // Arrange
  LLVMContext context;
  std::unique_ptr<Module> llvmModule(new Module("module", context));

  auto int8Type = Type::getInt8Ty(context);
  auto int64Type = Type::getInt64Ty(context);
  auto pointerType = PointerType::getUnqual(context);
  auto voidType = Type::getVoidTy(context);

  auto functionType =
      FunctionType::get(voidType, ArrayRef<Type *>({ pointerType, int8Type, int64Type }), false);
  auto function =
      Function::Create(functionType, GlobalValue::ExternalLinkage, "f", llvmModule.get());
  auto destination = function->getArg(0);
  auto value = function->getArg(1);
  auto length = function->getArg(2);

  auto llvmBasicBlock = BasicBlock::Create(context, "BasicBlock", function);

  IRBuilder builder(llvmBasicBlock);
  builder.CreateMemSet(destination, value, length, MaybeAlign());
  builder.CreateRetVoid();

  llvmModule->print(errs(), nullptr);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(*llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    using namespace jlm::llvm;

    auto controlFlowGraph =
        dynamic_cast<const FunctionNode *>(ipgModule->ipgraph().find("f"))->cfg();
    auto jlmBasicBlock =
        dynamic_cast<const jlm::llvm::BasicBlock *>(controlFlowGraph->entry()->OutEdge(0)->sink());

    size_t numMemsetThreeAddressCodes = 0;
    for (auto it = jlmBasicBlock->begin(); it != jlmBasicBlock->end(); ++it)
    {
      if (is<MemSetNonVolatileOperation>(*it))
      {
        numMemsetThreeAddressCodes++;
        auto memoryStateAssignment = *std::next(it, 1);

        EXPECT_TRUE(is<AssignmentOperation>(memoryStateAssignment->operation()));
        EXPECT_TRUE(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
    }

    EXPECT_EQ(numMemsetThreeAddressCodes, 1u);
  }
}
