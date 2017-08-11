/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
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

typedef std::function<llvm::Value*(llvm::IRBuilder<>&, llvm::Value*, llvm::Value*)> create_binop_t;

static inline void
test_binop(const create_binop_t & create_binop)
{
	using namespace llvm;

	LLVMContext ctx;
	auto flttype = Type::getFloatTy(ctx);
	auto ftype = FunctionType::get(flttype, {flttype, flttype}, false);

	std::unique_ptr<Module> module(new llvm::Module("module", ctx));
	auto f = Function::Create(ftype, Function::ExternalLinkage, "f", module.get());
	auto bb = BasicBlock::Create(ctx, "entry", f, nullptr);

	IRBuilder<> builder(bb);
	auto v = create_binop(builder, f->arg_begin(), std::next(f->arg_begin()));
	builder.CreateRet(v);
	module->dump();

	auto m = jlm::convert_module(*module);
	auto rvsdg = jlm::construct_rvsdg(*m);
	jive::view(rvsdg->graph()->root(), stdout);
}

static inline void
test_add()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateFAdd(lhs, rhs);
	};

	test_binop(create);
}

static inline void
test_sub()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateFSub(lhs, rhs);
	};

	test_binop(create);
}
static inline void
test_mul()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateFMul(lhs, rhs);
	};

	test_binop(create);
}
static inline void
test_div()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateFDiv(lhs, rhs);
	};

	test_binop(create);
}

static int
verify()
{
	/* FIXME: reactive tests */
//	test_add();
//	test_sub();
//	test_mul();
//	test_div();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-fltops", verify);
