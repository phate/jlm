/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#include "test-registry.hpp"

#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/print.hpp>
#include <jlm/jlm2llvm/jlm2llvm.hpp>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>

#include <iostream>

static void
print(const llvm::Module & module)
{
	llvm::raw_os_ostream os(std::cout);
	module.print(os, nullptr);
}

static void
test_malloc()
{
	auto setup = []() {
		using namespace jlm;

		jive::memtype mt;
		ptrtype pt(jive::bit8);
		auto im = ipgraph_module::create(filepath(""), "", "");

		auto cfg = cfg::create(*im);
		auto bb = basic_block::create(*cfg);
		cfg->exit()->divert_inedges(bb);
		bb->add_outedge(cfg->exit());

		auto size = cfg->entry()->append_argument(argument::create("size", jive::bit64));

		bb->append_last(malloc_op::create(size));

		cfg->exit()->append_result(bb->last()->result(0));
		cfg->exit()->append_result(bb->last()->result(1));

		jive::fcttype ft({&jive::bit64}, {&pt, &mt});
		auto f = function_node::create(im->ipgraph(), "f", ft, linkage::external_linkage);
		f->add_cfg(std::move(cfg));

		return im;
	};

	auto verify = [](const llvm::Module & m) {
		using namespace llvm;

		auto f = m.getFunction("f");
		auto & bb = f->getEntryBlock();

		assert(bb.getInstList().size() == 2);
		assert(bb.getFirstNonPHI()->getOpcode() == llvm::Instruction::Call);
		assert(bb.getTerminator()->getOpcode() == llvm::Instruction::Ret);
	};

	auto im = setup();
	jlm::print(*im, stdout);

	llvm::LLVMContext ctx;
	auto lm = jlm::jlm2llvm::convert(*im, ctx);
	print(*lm);

	verify(*lm);
}

static int
test()
{
	test_malloc();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/jlm-llvm/test-function-calls", test)
