/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

static int
LoadConversion()
{
  using namespace llvm;

  // Arrange
  llvm::LLVMContext context;
  std::unique_ptr<Module> llvmModule(new Module("module", context));

  auto int64Type = Type::getInt64Ty(context);
  auto pointerType = Type::getInt64PtrTy(context);

  auto functionType = FunctionType::get(int64Type, ArrayRef<Type *>({ pointerType }), false);
  auto function =
      Function::Create(functionType, GlobalValue::ExternalLinkage, "f", llvmModule.get());

  auto basicBlock = BasicBlock::Create(context, "BasicBlock", function);

  IRBuilder<> builder(basicBlock);
  auto loadedValue1 = builder.CreateLoad(int64Type, function->arg_begin(), true);
  auto loadedValue2 = builder.CreateLoad(int64Type, function->arg_begin(), false);
  auto loadedValue3 = builder.CreateLoad(int64Type, function->arg_begin(), true);
  auto sum1 = builder.CreateAdd(loadedValue1, loadedValue2);
  auto sum2 = builder.CreateAdd(sum1, loadedValue3);
  builder.CreateRet(sum2);

  jlm::tests::print(*llvmModule);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(*llvmModule);
  print(*ipgModule, stdout);

  // Assert
  {
    using namespace jlm::llvm;

    auto controlFlowGraph =
        dynamic_cast<const function_node *>(ipgModule->ipgraph().find("f"))->cfg();
    auto basicBlock =
        dynamic_cast<const basic_block *>(controlFlowGraph->entry()->outedge(0)->sink());

    size_t numLoadThreeAddressCodes = 0;
    size_t numLoadVolatileThreeAddressCodes = 0;
    for (auto it = basicBlock->begin(); it != basicBlock->end(); it++)
    {
      if (is<LoadVolatileOperation>(*it))
      {
        numLoadVolatileThreeAddressCodes++;
        auto ioStateAssignment = *std::next(it);
        auto memoryStateAssignment = *std::next(it, 2);

        assert(is<assignment_op>(ioStateAssignment->operation()));
        assert(is<iostatetype>(ioStateAssignment->operand(0)->type()));

        assert(is<assignment_op>(memoryStateAssignment->operation()));
        assert(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
      else if (is<LoadNonVolatileOperation>(*it))
      {
        numLoadThreeAddressCodes++;
        auto memoryStateAssignment = *std::next(it, 1);

        assert(is<assignment_op>(memoryStateAssignment->operation()));
        assert(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
    }

    assert(numLoadThreeAddressCodes == 1);
    assert(numLoadVolatileThreeAddressCodes == 2);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/LoadTests-LoadConversion", LoadConversion)
