/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jive/view.h>
#include <jive/vsdg/graph.h>

#include <jlm/jlm2rvsdg/module.hpp>
#include <jlm/IR/module.hpp>
#include <jlm/llvm2jlm/module.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

static inline void
test_fcmp(const llvm::CmpInst::Predicate & p)
{
	using namespace llvm;

	LLVMContext ctx;
	auto int1 = Type::getInt1Ty(ctx);
	auto flttype = Type::getFloatTy(ctx);
	auto ftype = FunctionType::get(int1, {flttype, flttype}, false);

	std::unique_ptr<Module> module(new llvm::Module("module", ctx));
	auto f = Function::Create(ftype, Function::ExternalLinkage, "f", module.get());
	auto bb = BasicBlock::Create(ctx, "entry", f, nullptr);

	IRBuilder<> builder(bb);
	auto v = builder.CreateFCmp(p, f->arg_begin(), std::next(f->arg_begin()));
	builder.CreateRet(v);
	module->dump();

	auto m = jlm::convert_module(*module);
	auto rvsdg = jlm::construct_rvsdg(*m);
	jive::view(rvsdg->root(), stdout);

}

static int
verify()
{
	using namespace llvm;

	test_fcmp(CmpInst::FCMP_FALSE);
	test_fcmp(CmpInst::FCMP_OEQ);
	test_fcmp(CmpInst::FCMP_OGT);
	test_fcmp(CmpInst::FCMP_OGE);
	test_fcmp(CmpInst::FCMP_OLT);
	test_fcmp(CmpInst::FCMP_OLE);
	test_fcmp(CmpInst::FCMP_ONE);
	test_fcmp(CmpInst::FCMP_ORD);
	test_fcmp(CmpInst::FCMP_UNO);
	test_fcmp(CmpInst::FCMP_UEQ);
	test_fcmp(CmpInst::FCMP_UGT);
	test_fcmp(CmpInst::FCMP_UGE);
	test_fcmp(CmpInst::FCMP_ULT);
	test_fcmp(CmpInst::FCMP_ULE);
	test_fcmp(CmpInst::FCMP_UNE);
	test_fcmp(CmpInst::FCMP_TRUE);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-fcmp", verify);
