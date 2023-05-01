/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

static int
test()
{
	auto setup = [](llvm::LLVMContext & ctx)
	{
		using namespace llvm;

		std::unique_ptr<Module> module(new Module("module", ctx));

		auto int64 = Type::getIntNTy(ctx, 64);

		auto fcttype = FunctionType::get(int64, {}, false);
		auto fct = Function::Create(fcttype, GlobalValue::ExternalLinkage, "f", module.get());

		auto bb1 = BasicBlock::Create(ctx, "", fct);
		auto bb2 = BasicBlock::Create(ctx, "", fct);

		IRBuilder<> builder(bb1);
		builder.CreateBr(bb2);
		builder.SetInsertPoint(bb2);
		builder.CreateBr(bb2);

		return module;
	};

	llvm::LLVMContext ctx;
	auto llvmModule = setup(ctx);
	jlm::print(*llvmModule);

	auto ipgModule = jlm::ConvertLlvmModule(*llvmModule);
	jlm::print(*ipgModule, stdout);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-endless-loop", test)
