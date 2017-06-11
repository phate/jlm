/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/destruction/destruction.hpp>
#include <jlm/IR/module.hpp>
#include <jlm/llvm2jlm/module.hpp>

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
	auto ftype = FunctionType::get(Type::getVoidTy(ctx), false);
	auto f = Function::Create(ftype, Function::ExternalLinkage, "main", &module);
	auto bb1 = BasicBlock::Create(ctx, "bb1", f, nullptr);
	auto bb2 = BasicBlock::Create(ctx, "bb2", f, nullptr);

	{
		IRBuilder<> builder(bb1);
		builder.CreateUnreachable();
	}

	{
		IRBuilder<> builder(bb2);
		builder.CreateRetVoid();
	}

	module.dump();

	using namespace jlm;

	auto m = convert_module(module);
	auto rvsdg = construct_rvsdg(*m);

	jive::view(rvsdg->root(), stdout);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-unreachable", verify);
