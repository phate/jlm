/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
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

typedef std::function<llvm::Value*(llvm::IRBuilder<>&, llvm::Value*, llvm::Type*)> create_cast_t;

static inline void
test_cast(
	llvm::LLVMContext & ctx,
	const create_cast_t & create_cast,
	llvm::Type * argtype,
	llvm::Type * rtype)
{
	using namespace llvm;

	auto ftype = FunctionType::get(rtype, {argtype}, false);

	std::unique_ptr<Module> module(new llvm::Module("module", ctx));
	auto f = Function::Create(ftype, Function::ExternalLinkage, "f", module.get());
	auto bb = BasicBlock::Create(ctx, "entry", f, nullptr);

	IRBuilder<> builder(bb);
	auto v = create_cast(builder, f->arg_begin(), rtype);
	builder.CreateRet(v);
	module->dump();

	auto m = jlm::convert_module(*module);
	auto rvsdg = jlm::construct_rvsdg(*m);
	jive::view(rvsdg->root(), stdout);
}

static int
verify_bitcast()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateBitCast(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt32PtrTy(ctx), Type::getInt32PtrTy(ctx));

	return 0;
}

static int
verify_fpext()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateFPExt(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getFloatTy(ctx), Type::getDoubleTy(ctx));

	return 0;
}

static int
verify_fptosi()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateFPToSI(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getFloatTy(ctx), Type::getInt32Ty(ctx));

	return 0;
}

static int
verify_fptoui()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateFPToUI(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getFloatTy(ctx), Type::getInt32Ty(ctx));

	return 0;
}

static int
verify_fptrunc()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateFPTrunc(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getDoubleTy(ctx), Type::getFloatTy(ctx));

	return 0;
}

static int
verify_inttoptr()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateIntToPtr(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt64Ty(ctx), Type::getInt64PtrTy(ctx));

	return 0;
}

static int
verify_ptrtoint()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreatePtrToInt(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt64PtrTy(ctx), Type::getInt64Ty(ctx));

	return 0;
}

static int
verify_sext()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateSExt(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt32Ty(ctx), Type::getInt64Ty(ctx));

	return 0;
}

static int
verify_sitofp()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateSIToFP(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt32Ty(ctx), Type::getFloatTy(ctx));

	return 0;
}

static int
verify_trunc()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateTrunc(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt64Ty(ctx), Type::getInt32Ty(ctx));

	return 0;
}

static int
verify_uitofp()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateUIToFP(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt32Ty(ctx), Type::getFloatTy(ctx));

	return 0;
}

static int
verify_zext()
{
	using namespace llvm;

	auto create = [](IRBuilder<> & irb, Value * v, Type *t)
	{
		return irb.CreateZExt(v, t);
	};

	LLVMContext ctx;
	test_cast(ctx, create, Type::getInt16Ty(ctx), Type::getInt64Ty(ctx));

	return 0;
}

static int
verify()
{
	verify_bitcast();
	verify_fpext();
	verify_fptosi();
	verify_fptoui();
	verify_fptrunc();
	verify_inttoptr();
	verify_ptrtoint();
	verify_sext();
	verify_sitofp();
	verify_trunc();
	verify_uitofp();
	verify_zext();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-casts", verify);
