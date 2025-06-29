/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/MemCpy.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

static void
MemCpyConversion()
{
  using namespace llvm;

  // Arrange
  llvm::LLVMContext context;
  std::unique_ptr<Module> llvmModule(new Module("module", context));

  auto int64Type = Type::getInt64Ty(context);
  auto pointerType = llvm::PointerType::getUnqual(context);
  auto voidType = Type::getVoidTy(context);

  auto functionType =
      FunctionType::get(voidType, ArrayRef<Type *>({ pointerType, pointerType, int64Type }), false);
  auto function =
      Function::Create(functionType, GlobalValue::ExternalLinkage, "f", llvmModule.get());
  auto destination = function->getArg(0);
  auto source = function->getArg(1);
  auto length = function->getArg(2);

  auto basicBlock = BasicBlock::Create(context, "BasicBlock", function);

  IRBuilder<> builder(basicBlock);
  builder.CreateMemCpy(destination, MaybeAlign(), source, MaybeAlign(), length, true);
  builder.CreateMemCpy(destination, MaybeAlign(), source, MaybeAlign(), length, false);
  builder.CreateMemCpy(destination, MaybeAlign(), source, MaybeAlign(), length, true);
  builder.CreateRetVoid();

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
        dynamic_cast<const jlm::llvm::BasicBlock *>(controlFlowGraph->entry()->OutEdge(0)->sink());

    size_t numMemCpyThreeAddressCodes = 0;
    size_t numMemCpyVolatileThreeAddressCodes = 0;
    for (auto it = basicBlock->begin(); it != basicBlock->end(); it++)
    {
      if (is<MemCpyVolatileOperation>(*it))
      {
        numMemCpyVolatileThreeAddressCodes++;
        auto ioStateAssignment = *std::next(it);
        auto memoryStateAssignment = *std::next(it, 2);

        assert(is<AssignmentOperation>(ioStateAssignment->operation()));
        assert(is<IOStateType>(ioStateAssignment->operand(0)->type()));

        assert(is<AssignmentOperation>(memoryStateAssignment->operation()));
        assert(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
      else if (is<MemCpyNonVolatileOperation>(*it))
      {
        numMemCpyThreeAddressCodes++;
        auto memoryStateAssignment = *std::next(it, 1);

        assert(is<AssignmentOperation>(memoryStateAssignment->operation()));
        assert(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
    }

    assert(numMemCpyThreeAddressCodes == 1);
    assert(numMemCpyVolatileThreeAddressCodes == 2);
  }
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/MemCpyTests-MemCpyConversion", MemCpyConversion)
