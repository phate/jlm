/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/evaluator/eval.h>
#include <jive/evaluator/literal.h>
#include <jive/types/bitstring/arithmetic.h>
#include <jive/types/bitstring/comparison.h>
#include <jive/types/bitstring/type.h>
#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/IR/module.hpp>
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

#include <assert.h>

typedef std::function<llvm::Value*(llvm::IRBuilder<>&, llvm::Value*, llvm::Value*)> create_binop_t;

static inline void
verify_binop(const jive::graph * rvsdg, uint64_t lhs, uint64_t rhs, uint64_t r)
{
	using namespace jive::evaluator;

	memliteral state;
	bitliteral xl(jive::bits::value_repr(64, lhs));
	bitliteral yl(jive::bits::value_repr(64, rhs));

	auto result = eval(rvsdg, "f", {&xl, &yl, &state})->copy();

	auto fctlit = dynamic_cast<const fctliteral*>(result.get());
	assert(fctlit->nresults() == 2);
	assert(dynamic_cast<const bitliteral*>(&fctlit->result(0))->value_repr() == r);
}

static inline void
test_binop(const create_binop_t & create_binop, uint64_t lhs, uint64_t rhs, uint64_t r)
{
	using namespace llvm;

	LLVMContext ctx;
	auto int64 = Type::getInt64Ty(ctx);
	auto ftype = FunctionType::get(int64, {int64, int64}, false);

	std::unique_ptr<Module> module(new llvm::Module("module", ctx));
	auto f = Function::Create(ftype, Function::ExternalLinkage, "f", module.get());
	auto bb = BasicBlock::Create(ctx, "entry", f, nullptr);

	IRBuilder<> builder(bb);
	auto v = create_binop(builder, f->arg_begin(), std::next(f->arg_begin()));
	builder.CreateRet(v);
	module->dump();

	auto m = jlm::convert_module(*module);
	auto rvsdg = jlm::construct_rvsdg(*m);
	jive::view(rvsdg->root(), stdout);

	verify_binop(rvsdg.get(), lhs, rhs, r);

	m = jlm::rvsdg2jlm::rvsdg2jlm(*rvsdg);
	module = jlm::jlm2llvm::convert(*m, ctx);
	module->dump();
}

static inline void
test_add()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateAdd(lhs, rhs);
	};

	test_binop(create, 3, 4, 7);
}

static inline void
test_and()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateAnd(lhs, rhs);
	};

	test_binop(create, 3, 6, 2);
}

static inline void
test_ashr()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateAShr(lhs, rhs);
	};

	test_binop(create, 0x8000000000000001, 1, 0xC000000000000000);
}

static inline void
test_sub()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateSub(lhs, rhs);
	};

	test_binop(create, 5, 3, 2);
}

static inline void
test_udiv()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateUDiv(lhs, rhs);
	};

	test_binop(create, 16, 4, 4);
}

static inline void
test_sdiv()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateSDiv(lhs, rhs);
	};

	test_binop(create, -16, 4, -4);
}

static inline void
test_urem()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateURem(lhs, rhs);
	};

	test_binop(create, 16, 5, 1);
}

static inline void
test_srem()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateSRem(lhs, rhs);
	};

	test_binop(create, -16, 5, -1);
}

static inline void
test_shl()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateShl(lhs, rhs);
	};

	test_binop(create, 1, 1, 2);
}

static inline void
test_lshr()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateLShr(lhs, rhs);
	};

	test_binop(create, 2, 1, 1);
}

static inline void
test_or()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateOr(lhs, rhs);
	};

	test_binop(create, 3, 6, 7);
}

static inline void
test_xor()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateXor(lhs, rhs);
	};

	test_binop(create, 3, 6, 5);
}

static inline void
test_mul()
{
	auto create = [](llvm::IRBuilder<> & irb, llvm::Value * lhs, llvm::Value * rhs)
	{
		return irb.CreateMul(lhs, rhs);
	};

	test_binop(create, 3, 4, 12);
}

static int
verify()
{
	test_add();
	test_and();
	test_ashr();
	test_sub();
	test_udiv();
	test_sdiv();
	test_urem();
	test_srem();
	test_shl();
	test_lshr();
	test_or();
	test_xor();
	test_mul();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-bitops", verify);
