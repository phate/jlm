/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/ir/module.hpp>
#include <jlm/ir/view.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>
#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/llvm2jlm/module.hpp>
#include <jlm/rvsdg2jlm/rvsdg2jlm.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

static int
verify()
{
	using namespace llvm;

	LLVMContext ctx;
	Module module("module", ctx);

	auto int64 = Type::getInt64Ty(ctx);
	auto ftype = FunctionType::get(int64, {int64, int64}, false);

	auto f = Function::Create(ftype, Function::ExternalLinkage, "f", &module);
	f->arg_begin()->setName("x");
	std::next(f->arg_begin())->setName("y");

	auto bb1 = BasicBlock::Create(ctx, "entry", f, nullptr);
	auto bb2 = BasicBlock::Create(ctx, "", f, nullptr);
	auto bb3 = BasicBlock::Create(ctx, "", f, nullptr);
	auto bb4 = BasicBlock::Create(ctx, "", f, nullptr);

	{
		IRBuilder<> builder(bb1);
		auto cmp = builder.CreateICmpEQ(f->arg_begin(), std::next(f->arg_begin()));
		builder.CreateCondBr(cmp, bb2, bb3);
	}

	{
		IRBuilder<> builder(bb2);
		builder.CreateBr(bb4);
	}

	{
		IRBuilder<> builder(bb3);
		builder.CreateBr(bb4);
	}

	{
		IRBuilder<> builder(bb4);
		auto phi = builder.CreatePHI(int64, 2);
		phi->addIncoming(ConstantInt::get(int64, 42), bb2);
		phi->addIncoming(f->arg_begin(), bb3);
		builder.CreateRet(phi);

	}

	module.dump();

	auto jm = jlm::convert_module(module);
	jlm::view(*jm, stdout);

	auto rvsdg = jlm::construct_rvsdg(*jm);
	jive::view(rvsdg->root(), stdout);

	jm = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);
	jlm::view(*jm, stdout);

	auto lm = jlm::jlm2llvm::convert(*jm, ctx);
	lm->dump();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-phi", verify);
