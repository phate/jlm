/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

template<class OP>
static bool
contains(const jlm::llvm::InterProceduralGraphModule & module, const std::string & fctname)
{
  using namespace jlm::llvm;

  bool has_select = false;
  auto cfg = dynamic_cast<const FunctionNode *>(module.ipgraph().find(fctname))->cfg();
  auto bb = dynamic_cast<const BasicBlock *>(cfg->entry()->OutEdge(0)->sink());
  for (auto tac : *bb)
    has_select = has_select || is<OP>(tac);

  return has_select;
}

static void
test_scalar_select()
{
  auto setup = [](llvm::LLVMContext & ctx)
  {
    using namespace llvm;

    std::unique_ptr<Module> module(new Module("module", ctx));

    auto int1 = Type::getIntNTy(ctx, 1);
    auto int64 = Type::getIntNTy(ctx, 64);

    auto fctargs = std::vector<Type *>({ int1, int64, int64 });
    auto fcttype = FunctionType::get(int64, fctargs, false);
    auto fct = Function::Create(fcttype, GlobalValue::ExternalLinkage, "f", module.get());

    auto bb = BasicBlock::Create(ctx, "bb", fct);

    IRBuilder<> builder(bb);
    auto r = builder.CreateSelect(fct->arg_begin(), fct->arg_begin() + 1, fct->arg_begin() + 2);
    builder.CreateRet(r);

    return module;
  };

  llvm::LLVMContext ctx;
  auto llmod = setup(ctx);
  jlm::tests::print(*llmod);

  auto ipgmod = jlm::llvm::ConvertLlvmModule(*llmod);
  print(*ipgmod, stdout);

  assert(contains<jlm::llvm::SelectOperation>(*ipgmod, "f"));
}

static void
test_vector_select()
{
  auto setup = [](llvm::LLVMContext & ctx)
  {
    using namespace llvm;

    std::unique_ptr<Module> module(new Module("module", ctx));

    auto vint1 = VectorType::get(Type::getIntNTy(ctx, 1), 4, false);
    auto vint64 = VectorType::get(Type::getIntNTy(ctx, 64), 4, false);

    auto fctargs = std::vector<Type *>({ vint1, vint64, vint64 });
    auto fcttype = FunctionType::get(vint64, fctargs, false);
    auto fct = Function::Create(fcttype, GlobalValue::ExternalLinkage, "f", module.get());

    auto bb = BasicBlock::Create(ctx, "bb", fct);

    IRBuilder<> builder(bb);
    auto r = builder.CreateSelect(fct->arg_begin(), fct->arg_begin() + 1, fct->arg_begin() + 2);
    builder.CreateRet(r);

    return module;
  };

  llvm::LLVMContext ctx;
  auto llmod = setup(ctx);
  jlm::tests::print(*llmod);

  auto ipgmod = jlm::llvm::ConvertLlvmModule(*llmod);
  print(*ipgmod, stdout);

  assert(contains<jlm::llvm::VectorSelectOperation>(*ipgmod, "f"));
}

static void
test()
{
  test_scalar_select();
  test_vector_select();
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-select", test)
