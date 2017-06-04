/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/construction/module.hpp>
#include <jlm/destruction/destruction.hpp>
#include <jlm/IR/module.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

static inline void
test_direct_call()
{
	using namespace llvm;

	LLVMContext ctx;
	Module module("module", ctx);
	auto callee0_t = FunctionType::get(Type::getVoidTy(ctx), false);
	auto callee1_t = FunctionType::get(Type::getInt32Ty(ctx), {Type::getInt32Ty(ctx)}, false);
	auto caller_t = FunctionType::get(Type::getInt32Ty(ctx), {Type::getInt32Ty(ctx)}, false);

	auto callee0 = Function::Create(callee0_t, Function::ExternalLinkage, "callee0", &module);
	auto callee1 = Function::Create(callee1_t, Function::ExternalLinkage, "callee1", &module);
	auto caller = Function::Create(caller_t, Function::ExternalLinkage, "caller", &module);

	auto entry_callee0 = BasicBlock::Create(ctx, "entry", callee0, nullptr);
	{
		IRBuilder<> builder(entry_callee0);
		builder.CreateRetVoid();
	}

	auto entry_callee1 = BasicBlock::Create(ctx, "entry", callee1, nullptr);
	{
		IRBuilder<> builder(entry_callee1);
		builder.CreateRet(callee1->arg_begin());
	}

	auto entry_caller = BasicBlock::Create(ctx, "entry", caller, nullptr);
	{
		IRBuilder<> builder(entry_caller);
		auto r = builder.CreateCall(callee1, {caller->arg_begin()});
		builder.CreateCall(callee0);
		builder.CreateRet(r);
	}

	module.dump();

	using namespace jlm;

	auto m = convert_module(module);
	auto rvsdg = construct_rvsdg(*m);

	jive::view(rvsdg->root(), stdout);
}

static inline void
test_indirect_call()
{
	using namespace llvm;

	LLVMContext ctx;
	Module module("module", ctx);
	auto int32 = Type::getInt32Ty(ctx);
	auto ftype = FunctionType::get(int32, {int32}, false);
	auto ptrtype = PointerType::get(ftype, 0);
	auto ftype2 = FunctionType::get(int32, {ptrtype, int32}, false);

	auto f = Function::Create(ftype2, Function::ExternalLinkage, "indirect_call", &module);
	auto bb = BasicBlock::Create(ctx, "entry", f, nullptr);

	IRBuilder<> builder(bb);
	auto v = builder.CreateCall(f->arg_begin(), {std::next(f->arg_begin())});
	builder.CreateRet(v);

	module.dump();

	using namespace jlm;

	auto m = convert_module(module);
	auto rvsdg = construct_rvsdg(*m);

	jive::view(rvsdg->root(), stdout);

}

static int
verify()
{
	test_direct_call();
	test_indirect_call();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-call", verify)
