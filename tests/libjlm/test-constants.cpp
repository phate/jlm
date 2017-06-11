/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
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

#include <assert.h>

static inline void
verify_constantInt()
{
	using namespace llvm;

	LLVMContext ctx;
	Module module("module", ctx);
	auto ftype = FunctionType::get(Type::getInt32Ty(ctx), false);
	auto f = Function::Create(ftype, Function::ExternalLinkage, "", &module);
	auto bb = BasicBlock::Create(ctx, "f", f, nullptr);

	IRBuilder<> builder(bb);
	builder.CreateRet(ConstantInt::get(Type::getInt32Ty(ctx), 32));

	module.dump();

	using namespace jlm;

	auto m = convert_module(module);
	auto rvsdg = construct_rvsdg(*m);

	jive::view(rvsdg->root(), stdout);
}

static inline int
verify_constantFP()
{
	/* FIXME: insert checks for all types */

	return 0;
}

static inline int
verify_constantPointerNull()
{
	/* FIXME: insert checks */

	return 0;
}

static inline int
verify_globalVariable()
{
	/* FIXME: insert checks */

	return 0;
}

static inline void
verify_undefValue()
{
	using namespace llvm;

	LLVMContext ctx;
	Module module("module", ctx);
	auto ftype = FunctionType::get(Type::getInt32Ty(ctx), false);
	auto f = Function::Create(ftype, Function::ExternalLinkage, "", &module);
	auto bb = BasicBlock::Create(ctx, "f", f, nullptr);

	IRBuilder<> builder(bb);
	builder.CreateRet(UndefValue::get(Type::getInt32Ty(ctx)));

	module.dump();

	using namespace jlm;

	auto m = convert_module(module);
	auto rvsdg = construct_rvsdg(*m);

	jive::view(rvsdg->root(), stdout);
}

static inline int
verify_constantAggregateZeroStruct()
{
	/* FIXME: insert checks */

	return 0;
}

static int
verify()
{
	verify_constantFP();
	verify_constantInt();
	verify_constantPointerNull();
	verify_globalVariable();
	verify_undefValue();
	verify_constantAggregateZeroStruct();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-constants", verify);
