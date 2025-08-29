/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
#include <jlm/llvm/ir/print.hpp>

static void
TestFMulAdd()
{
  // Arrange
  using namespace llvm;

  LLVMContext context;
  Module llvmModule("module", context);

  {
    auto doubleType = Type::getDoubleTy(context);

    const auto functionArguments = std::vector<Type *>({ doubleType, doubleType, doubleType });
    const auto functionType = FunctionType::get(doubleType, functionArguments, false);
    const auto function =
        Function::Create(functionType, GlobalValue::ExternalLinkage, "f", &llvmModule);

    const auto basicBlock = BasicBlock::Create(context, "basicBlock", function);

    IRBuilder builder(basicBlock);
    const auto returnValue = builder.CreateIntrinsic(
        Intrinsic::fmuladd,
        { doubleType },
        { function->getArg(0), function->getArg(1), function->getArg(2) });
    builder.CreateRet(returnValue);
  }

  jlm::tests::print(llvmModule);

  // Act
  const auto ipgModule = jlm::llvm::ConvertLlvmModule(llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    const auto controlFlowGraph =
        dynamic_cast<const jlm::llvm::FunctionNode *>(ipgModule->ipgraph().find("f"))->cfg();
    const auto basicBlock =
        dynamic_cast<const jlm::llvm::BasicBlock *>(controlFlowGraph->entry()->OutEdge(0)->sink());
    const auto fMulAdd = *std::next(basicBlock->rbegin(), 1);
    assert(jlm::llvm::is<jlm::llvm::FMulAddIntrinsicOperation>(fMulAdd));
  }
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/SpecializedArithmeticIntrinsicOperationTests-TestFMulAdd",
    TestFMulAdd);
