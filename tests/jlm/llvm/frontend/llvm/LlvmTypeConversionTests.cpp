/*
 * Copyright 2024 Halvor Linder Henriksen <halvor_lh@hotmail.no>
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

static void
TestTypeConversion(
    llvm::LLVMContext & context,
    llvm::Type * llvm_type,
    jlm::llvm::fpsize jlm_type_size)
{
  using namespace llvm;

  std::unique_ptr<Module> llvmModule(new Module("module", context));

  auto pointer_type = ::llvm::PointerType::getUnqual(llvm_type);

  auto function_type = FunctionType::get(llvm_type, ArrayRef<Type *>({ pointer_type }), false);
  auto function =
      Function::Create(function_type, GlobalValue::ExternalLinkage, "f", llvmModule.get());

  auto bb = BasicBlock::Create(context, "bb", function);

  auto arg0 = function->getArg(0);

  IRBuilder<> builder(bb);

  auto loadedValue1 = builder.CreateLoad(llvm_type, arg0, false);
  builder.CreateRet(loadedValue1);

  // Act
  auto ipgModule = jlm::llvm::ConvertLlvmModule(*llvmModule);
  print(*ipgModule, stdout);

  {
    using namespace jlm::llvm;
    auto controlFlowGraph =
        dynamic_cast<const function_node *>(ipgModule->ipgraph().find("f"))->cfg();
    auto basicBlock =
        dynamic_cast<const basic_block *>(controlFlowGraph->entry()->outedge(0)->sink());

    /*     Generated ipg bb looks something like this:
        <double, iostate, mem> f <ptr, iostate, mem>
        {
        0xbd0675417b80:  _io_ _s_

        0xbd0675418110:
          _r_ = undef
          tv0, tv1 = Load , _s_
          ASSIGN _s_, tv1
          ASSIGN _r_, tv0
          [0xbd0675417c00]

        0xbd0675417c00: _r_ _io_ _s_
        } */

    auto tac_it = basicBlock->begin();
    auto return_declaration = *tac_it;
    auto load = *std::next(tac_it);
    auto return_assignment = *std::next(tac_it, 3);

    std::cout << return_assignment->operation().debug_string() << std::endl;

    assert(is<fptype>(return_declaration->result(0)->type()));
    auto t0 = dynamic_cast<const fptype &>(return_declaration->result(0)->type());
    assert(t0.size() == jlm_type_size);

    assert(is<fptype>(load->result(0)->type()));
    auto t1 = dynamic_cast<const fptype &>(load->result(0)->type());
    assert(t1.size() == jlm_type_size);

    assert(is<fptype>(return_assignment->operand(0)->type()));
    auto t2 = dynamic_cast<const fptype &>(return_assignment->operand(0)->type());
    assert(t2.size() == jlm_type_size);

    assert(is<fptype>(return_assignment->operand(1)->type()));
    auto t3 = dynamic_cast<const fptype &>(return_assignment->operand(1)->type());
    assert(t3.size() == jlm_type_size);
  }
}

static int
TypeConversion()
{
  llvm::LLVMContext context;

  TestTypeConversion(context, ::llvm::Type::getHalfTy(context), jlm::llvm::fpsize::half);
  TestTypeConversion(context, ::llvm::Type::getFloatTy(context), jlm::llvm::fpsize::flt);
  TestTypeConversion(context, ::llvm::Type::getDoubleTy(context), jlm::llvm::fpsize::dbl);
  TestTypeConversion(context, ::llvm::Type::getX86_FP80Ty(context), jlm::llvm::fpsize::x86fp80);
  TestTypeConversion(context, ::llvm::Type::getFP128Ty(context), jlm::llvm::fpsize::fp128);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/LlvmTypeConversionTests-TypeConversion",
    TypeConversion)
