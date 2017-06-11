/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>
#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/llvm2jlm/module.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <assert.h>

static int
verify()
{
	using namespace llvm;

	LLVMContext ctx;
	Module module("module", ctx);
	auto int32 = Type::getInt32Ty(ctx);
	auto ftype = FunctionType::get(int32, {int32, int32}, false);
	auto f = Function::Create(ftype, Function::ExternalLinkage, "max", &module);
	auto bb = BasicBlock::Create(ctx, "entry", f, nullptr);

	IRBuilder<> builder(bb);
	auto c = builder.CreateICmpUGT(f->arg_begin(), std::next(f->arg_begin()));
	auto r  = builder.CreateSelect(c, f->arg_begin(), std::next(f->arg_begin()));
	builder.CreateRet(r);

	module.dump();

	using namespace jlm;

	auto m = convert_module(module);
	auto rvsdg = construct_rvsdg(*m);

	jive::view(rvsdg->root(), stdout);

	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(32, 13));
	bitliteral yl(jive::bits::value_repr(32, 14));

	auto result = eval(rvsdg.get(), "max", {&xl, &yl, &state})->copy();

	const fctliteral * fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 14);

	/* exchange arguments */
	result = eval(rvsdg.get(), "max", {&yl, &xl, &state})->copy();

	fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == 14);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-select", verify);
