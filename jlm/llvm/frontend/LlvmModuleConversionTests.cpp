/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
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
