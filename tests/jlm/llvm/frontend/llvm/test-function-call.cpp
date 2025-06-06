/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

static void
test_function_call()
{
  auto setup = [](llvm::LLVMContext & ctx)
  {
    using namespace llvm;

    std::unique_ptr<Module> module(new Module("module", ctx));

    auto int64 = Type::getIntNTy(ctx, 64);
    auto clrtype = FunctionType::get(Type::getVoidTy(ctx), {}, false);
    auto caller = Function::Create(clrtype, GlobalValue::ExternalLinkage, "caller", module.get());

    auto bb = BasicBlock::Create(ctx, "bb", caller);

    auto c = ConstantInt::get(int64, 45);
    auto cletype = FunctionType::get(int64, ArrayRef<Type *>(std::vector<Type *>(2, int64)), false);
    auto callee = module->getOrInsertFunction("callee", cletype);

    IRBuilder<> builder(bb);
    builder.CreateCall(callee, ArrayRef<Value *>(std::vector<Value *>(2, c)));
    builder.CreateRetVoid();

    return module;
  };

  auto verify = [](const jlm::llvm::ipgraph_module & module)
  {
    using namespace jlm::llvm;

    ControlFlowGraph * cfg = nullptr;
    for (auto & node : module.ipgraph())
    {
      if (node.name() == "caller")
      {
        cfg = dynamic_cast<const function_node &>(node).cfg();
        break;
      }
    }

    auto bb = dynamic_cast<const BasicBlock *>(cfg->entry()->OutEdge(0)->sink());
    assert(is<CallOperation>(*std::next(bb->rbegin(), 2)));
  };

  llvm::LLVMContext ctx;
  auto llmod = setup(ctx);
  jlm::tests::print(*llmod);

  auto ipgmod = jlm::llvm::ConvertLlvmModule(*llmod);
  print(*ipgmod, stdout);

  verify(*ipgmod);
}

static void
test_malloc_call()
{
  auto setup = [](llvm::LLVMContext & ctx)
  {
    using namespace llvm;

    std::unique_ptr<Module> module(new Module("module", ctx));

    auto int8 = Type::getIntNTy(ctx, 8);
    auto int64 = Type::getIntNTy(ctx, 64);
    auto ptrint8 = PointerType::get(int8, 0);

    auto clrtype = FunctionType::get(Type::getVoidTy(ctx), {}, false);
    auto caller = Function::Create(clrtype, GlobalValue::ExternalLinkage, "caller", module.get());

    auto bb = BasicBlock::Create(ctx, "bb", caller);

    auto c = ConstantInt::get(int64, 45);
    auto malloctype = FunctionType::get(ptrint8, ArrayRef<Type *>(int64), false);
    auto malloc = module->getOrInsertFunction("malloc", malloctype);

    IRBuilder<> builder(bb);
    builder.CreateCall(malloc, ArrayRef<Value *>(c));
    builder.CreateRetVoid();

    return module;
  };

  auto verify = [](const jlm::llvm::ipgraph_module & module)
  {
    using namespace jlm::llvm;

    ControlFlowGraph * cfg = nullptr;
    for (auto & node : module.ipgraph())
    {
      if (node.name() == "caller")
      {
        cfg = dynamic_cast<const jlm::llvm::function_node &>(node).cfg();
        break;
      }
    }

    auto bb = dynamic_cast<const BasicBlock *>(cfg->entry()->OutEdge(0)->sink());
    assert(is<MemoryStateMergeOperation>(*std::next(bb->rbegin())));
    assert(is<malloc_op>((*std::next(bb->rbegin(), 2))));
  };

  llvm::LLVMContext ctx;
  auto llmod = setup(ctx);
  jlm::tests::print(*llmod);

  auto ipgmod = jlm::llvm::ConvertLlvmModule(*llmod);
  print(*ipgmod, stdout);

  verify(*ipgmod);
}

static void
test_free_call()
{
  auto setup = [](llvm::LLVMContext & ctx)
  {
    using namespace llvm;

    std::unique_ptr<Module> module(new Module("module", ctx));

    auto int8 = Type::getIntNTy(ctx, 8);
    auto ptrint8 = PointerType::get(int8, 0);

    auto clrtype = FunctionType::get(Type::getVoidTy(ctx), { ptrint8 }, false);
    auto caller = Function::Create(clrtype, GlobalValue::ExternalLinkage, "caller", module.get());

    auto bb = BasicBlock::Create(ctx, "bb", caller);

    auto freetype = FunctionType::get(Type::getVoidTy(ctx), ArrayRef<Type *>(ptrint8), false);
    auto free = module->getOrInsertFunction("free", freetype);

    IRBuilder<> builder(bb);
    builder.CreateCall(free, ArrayRef<Value *>(caller->getArg(0)));
    builder.CreateRetVoid();

    return module;
  };

  auto verify = [](const jlm::llvm::ipgraph_module & module)
  {
    using namespace jlm::llvm;

    ControlFlowGraph * cfg = nullptr;
    for (auto & node : module.ipgraph())
    {
      if (node.name() == "caller")
      {
        cfg = dynamic_cast<const function_node &>(node).cfg();
        break;
      }
    }

    auto bb = dynamic_cast<const BasicBlock *>(cfg->entry()->OutEdge(0)->sink());
    assert(is<AssignmentOperation>(*bb->rbegin()));
    assert(is<AssignmentOperation>(*std::next(bb->rbegin())));
    assert(is<FreeOperation>(*std::next(bb->rbegin(), 2)));
  };

  llvm::LLVMContext ctx;
  auto llvmmod = setup(ctx);
  jlm::tests::print(*llvmmod);

  auto ipgmod = jlm::llvm::ConvertLlvmModule(*llvmmod);
  print(*ipgmod, stdout);

  verify(*ipgmod);
}

static int
test()
{
  test_function_call();
  test_malloc_call();
  test_free_call();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-function-call", test)
