/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

static int
StoreConversion()
{
  using namespace llvm;

  // Arrange
  llvm::LLVMContext context;
  std::unique_ptr<Module> llvmModule(new Module("module", context));

  auto int64Type = Type::getInt64Ty(context);
  auto pointerType = llvm::PointerType::getUnqual(context);
  auto voidType = Type::getVoidTy(context);

  auto functionType =
      FunctionType::get(voidType, ArrayRef<Type *>({ int64Type, pointerType }), false);
  auto function =
      Function::Create(functionType, GlobalValue::ExternalLinkage, "f", llvmModule.get());
  auto valueArgument = function->getArg(0);
  auto addressArgument = function->getArg(1);

  auto basicBlock = BasicBlock::Create(context, "BasicBlock", function);

  IRBuilder<> builder(basicBlock);
  builder.CreateStore(valueArgument, addressArgument, true);
  builder.CreateStore(valueArgument, addressArgument, false);
  builder.CreateStore(valueArgument, addressArgument, true);
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
        dynamic_cast<const basic_block *>(controlFlowGraph->entry()->OutEdge(0)->sink());

    size_t numStoreThreeAddressCodes = 0;
    size_t numStoreVolatileThreeAddressCodes = 0;
    for (auto it = basicBlock->begin(); it != basicBlock->end(); it++)
    {
      if (is<StoreVolatileOperation>(*it))
      {
        numStoreVolatileThreeAddressCodes++;
        auto ioStateAssignment = *std::next(it);
        auto memoryStateAssignment = *std::next(it, 2);

        assert(is<AssignmentOperation>(ioStateAssignment->operation()));
        assert(is<IOStateType>(ioStateAssignment->operand(0)->type()));

        assert(is<AssignmentOperation>(memoryStateAssignment->operation()));
        assert(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
      else if (is<StoreNonVolatileOperation>(*it))
      {
        numStoreThreeAddressCodes++;
        auto memoryStateAssignment = *std::next(it, 1);

        assert(is<AssignmentOperation>(memoryStateAssignment->operation()));
        assert(is<MemoryStateType>(memoryStateAssignment->operand(0)->type()));
      }
    }

    assert(numStoreThreeAddressCodes == 1);
    assert(numStoreVolatileThreeAddressCodes == 2);
  }

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/StoreTests-StoreConversion", StoreConversion)
