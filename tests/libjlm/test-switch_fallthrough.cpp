/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/ir/module.hpp>
#include <jlm/ir/rvsdg.hpp>
#include <jlm/jlm2rvsdg/module.hpp>
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
	auto ftype = FunctionType::get(Type::getInt32Ty(ctx), {Type::getInt32Ty(ctx)}, false);
	auto f = Function::Create(ftype, Function::ExternalLinkage, "f", &module);

	auto prolog = BasicBlock::Create(ctx, "prolog", f, nullptr);
	auto bb = BasicBlock::Create(ctx, "bb", f, nullptr);
	auto def = BasicBlock::Create(ctx, "default", f, nullptr);
	auto epilog = BasicBlock::Create(ctx, "epilog", f, nullptr);

	/* prolog */
	IRBuilder<> irb1(prolog);
	auto swi = irb1.CreateSwitch(f->arg_begin(), def);
	swi->addCase(ConstantInt::get(Type::getInt32Ty(ctx), 2), epilog);
	swi->addCase(ConstantInt::get(Type::getInt32Ty(ctx), 3), bb);

	/* bb */
	IRBuilder<> irb2(bb);
	irb2.CreateBr(def);

	/* default */
	IRBuilder<> irb3(def);
	auto phi = irb3.CreatePHI(Type::getInt32Ty(ctx), 2);
	phi->addIncoming(ConstantInt::get(Type::getInt32Ty(ctx), 4), prolog);
	phi->addIncoming(ConstantInt::get(Type::getInt32Ty(ctx), 7), bb);
	irb3.CreateBr(epilog);

	/* epilog */
	IRBuilder<> irb4(epilog);
	phi = irb4.CreatePHI(Type::getInt32Ty(ctx), 3);
	phi->addIncoming(phi, def);
	phi->addIncoming(ConstantInt::get(Type::getInt32Ty(ctx), 2), prolog);
	irb4.CreateRet(phi);

	module.dump();

	using namespace jlm;

	auto m = convert_module(module);
	auto rvsdg = construct_rvsdg(*m);

	jive::view(rvsdg->graph()->root(), stdout);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-switch_fallthrough", verify);
