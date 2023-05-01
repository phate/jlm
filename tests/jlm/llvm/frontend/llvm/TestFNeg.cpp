/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/print.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

template<class OP> static bool
Contains(const jlm::ipgraph_module & module, const std::string & fctname)
{
  using namespace jlm;

  bool hasInstruction = false;
  auto controlFlowGraph = dynamic_cast<const function_node*>(module.ipgraph().find("f"))->cfg();
  auto basicBlock = dynamic_cast<const basic_block*>(controlFlowGraph->entry()->outedge(0)->sink());
  for (auto threeAddressCode : *basicBlock)
    hasInstruction = hasInstruction || is<OP>(threeAddressCode);

  return hasInstruction;
}

static void
TestFNegScalar()
{
  auto Setup = [](llvm::LLVMContext & context)
  {
    using namespace llvm;

    std::unique_ptr<Module> module(new Module("module", context));

    auto doubleType = Type::getDoubleTy(context);

    auto functionArguments = std::vector<Type*>({doubleType});
    auto functionType = FunctionType::get(doubleType, functionArguments, false);
    auto function = Function::Create(functionType, GlobalValue::ExternalLinkage, "f", module.get());

    auto basicBlock = BasicBlock::Create(context, "basicBlock", function);

    IRBuilder<> builder(basicBlock);
    auto returnValue = builder.CreateFNeg(function->arg_begin());
    builder.CreateRet(returnValue);

    return module;
  };

  llvm::LLVMContext context;
  auto llvmModule = Setup(context);
  jlm::print(*llvmModule);

  auto ipgModule = jlm::ConvertLlvmModule(*llvmModule);
  jlm::print(*ipgModule, stdout);

  assert(Contains<jlm::fpneg_op>(*ipgModule, "f"));
}

static void
TestFNegVector()
{
  auto Setup = [](llvm::LLVMContext & context)
  {
    using namespace llvm;

    std::unique_ptr<Module> module(new Module("module", context));

    auto vectorType = VectorType::get(Type::getDoubleTy(context), 2, false);

    auto functionArguments = std::vector<Type*>({vectorType});
    auto functionType = FunctionType::get(vectorType, functionArguments, false);
    auto function = Function::Create(functionType, GlobalValue::ExternalLinkage, "f", module.get());

    auto basicBlock = BasicBlock::Create(context, "basicBlock", function);

    IRBuilder<> builder(basicBlock);
    auto returnValue = builder.CreateFNeg(function->arg_begin());
    builder.CreateRet(returnValue);

    return module;
  };

  llvm::LLVMContext context;
  auto llvmModule = Setup(context);
  jlm::print(*llvmModule);

  auto ipgModule = jlm::ConvertLlvmModule(*llvmModule);
  jlm::print(*ipgModule, stdout);

  assert(Contains<jlm::vectorunary_op>(*ipgModule, "f"));
}

static int
Test()
{
  TestFNegScalar();
  TestFNegVector();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/TestFNeg", Test)