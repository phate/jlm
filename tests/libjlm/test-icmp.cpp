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

#include <assert.h>

typedef std::function<llvm::Value*(llvm::IRBuilder<>&, llvm::Value*, llvm::Value*)> create_binop_t;

static inline void
verify_icmp(const jive::graph * rvsdg, uint64_t lhs, uint64_t rhs, uint64_t r)
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
test_icmp(const llvm::CmpInst::Predicate & p, uint64_t lhs, uint64_t rhs, uint64_t r)
{
	using namespace llvm;

	LLVMContext ctx;
	auto int1 = Type::getInt1Ty(ctx);
	auto int64 = Type::getInt64Ty(ctx);
	auto ftype = FunctionType::get(int1, {int64, int64}, false);

	std::unique_ptr<Module> module(new llvm::Module("module", ctx));
	auto f = Function::Create(ftype, Function::ExternalLinkage, "f", module.get());
	auto bb = BasicBlock::Create(ctx, "entry", f, nullptr);

	IRBuilder<> builder(bb);
	auto v = builder.CreateICmp(p, f->arg_begin(), std::next(f->arg_begin()));
	builder.CreateRet(v);
	module->dump();

	auto m = jlm::convert_module(*module);
	auto rvsdg = jlm::construct_rvsdg(*m);
	jive::view(rvsdg->graph()->root(), stdout);

	verify_icmp(rvsdg->graph(), lhs, rhs, r);
}

static int
verify()
{
	using namespace llvm;

	test_icmp(CmpInst::ICMP_SLT, -3, 4, 1);
	test_icmp(CmpInst::ICMP_ULT, 3, 4, 1);
	test_icmp(CmpInst::ICMP_SLE, -3, -3, 1);
	test_icmp(CmpInst::ICMP_ULE, -2, -3, 0);
	test_icmp(CmpInst::ICMP_EQ, 4, 5, 0);
	test_icmp(CmpInst::ICMP_NE, 4, 5, 1);
	test_icmp(CmpInst::ICMP_SGT, -4, -5, 1);
	test_icmp(CmpInst::ICMP_UGT, 4, 5, 0);
	test_icmp(CmpInst::ICMP_SGE, -4, -4, 1);
	test_icmp(CmpInst::ICMP_UGE, 4, 4, 1);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-icmp", verify);
