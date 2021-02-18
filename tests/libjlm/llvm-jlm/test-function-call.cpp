/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/ir/print.hpp>
#include <jlm/ir/operators/call.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/llvm2jlm/module.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>


static void
test_function_call()
{
	auto setup = [](llvm::LLVMContext & ctx) {
		using namespace llvm;

		std::unique_ptr<Module> module(new Module("module", ctx));

		auto int64 = Type::getIntNTy(ctx, 64);
		auto clrtype= FunctionType::get(Type::getVoidTy(ctx), {}, false);
		auto caller = Function::Create(clrtype, GlobalValue::ExternalLinkage, "caller", module.get());

		auto bb = BasicBlock::Create(ctx, "bb", caller);

		auto c = ConstantInt::get(int64, 45);
		auto cletype = FunctionType::get(int64, ArrayRef<Type*>(std::vector<Type*>(2, int64)), false);
		auto callee = module->getOrInsertFunction("callee", cletype);

		IRBuilder<> builder(bb);
		builder.CreateCall(callee, ArrayRef<Value*>(std::vector<Value*>(2, c)));
		builder.CreateRetVoid();

		return module;
	};

	auto verify = [](const jlm::ipgraph_module & module) {
		using namespace jlm;

		jlm::cfg * cfg = nullptr;
		for (auto & node : module.ipgraph()) {
			if (node.name() == "caller") {
				cfg = dynamic_cast<const function_node&>(node).cfg();
				break;
			}
		}

		auto bb = dynamic_cast<const basic_block*>(cfg->entry()->outedge(0)->sink());
		assert(is<call_op>(*std::next(bb->rbegin(), 3)));
	};


	llvm::LLVMContext ctx;
	auto llmod = setup(ctx);
	jlm::print(*llmod);

	auto ipgmod = jlm::convert_module(*llmod);
	jlm::print(*ipgmod, stdout);

	verify(*ipgmod);
}

void
test_malloc_call()
{
	auto setup = [](llvm::LLVMContext & ctx) {
		using namespace llvm;

		std::unique_ptr<Module> module(new Module("module", ctx));

		auto int8 = Type::getIntNTy(ctx, 8);
		auto int64 = Type::getIntNTy(ctx, 64);
		auto ptrint8 = PointerType::get(int8, 0);

		auto clrtype= FunctionType::get(Type::getVoidTy(ctx), {}, false);
		auto caller = Function::Create(clrtype, GlobalValue::ExternalLinkage, "caller", module.get());

		auto bb = BasicBlock::Create(ctx, "bb", caller);

		auto c = ConstantInt::get(int64, 45);
		auto malloctype = FunctionType::get(ptrint8, ArrayRef<Type*>(int64), false);
		auto malloc = module->getOrInsertFunction("malloc", malloctype);

		IRBuilder<> builder(bb);
		builder.CreateCall(malloc, ArrayRef<Value*>(c));
		builder.CreateRetVoid();

		return module;
	};

	auto verify = [](const jlm::ipgraph_module & module) {
		using namespace jlm;

		jlm::cfg * cfg = nullptr;
		for (auto & node : module.ipgraph()) {
			if (node.name() == "caller") {
				cfg = dynamic_cast<const function_node&>(node).cfg();
				break;
			}
		}

		auto bb = dynamic_cast<const basic_block*>(cfg->entry()->outedge(0)->sink());
		assert(is<memstatemux_op>(bb->last()->operation()));
		assert(is<malloc_op>((*std::next(bb->rbegin()))->operation()));
	};

	llvm::LLVMContext ctx;
	auto llmod = setup(ctx);
	jlm::print(*llmod);

	auto ipgmod = jlm::convert_module(*llmod);
	jlm::print(*ipgmod, stdout);

	verify(*ipgmod);
}

static int
test()
{
	test_function_call();
	test_malloc_call();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/llvm-jlm/test-function-call", test)
