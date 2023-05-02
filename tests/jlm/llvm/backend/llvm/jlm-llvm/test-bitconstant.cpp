/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

#include <llvm/IR/LLVMContext.h>

static int
test()
{
	const char * bs = "0100000000" "0000000000" "0000000000" "0000000000" "0000000000" "0000000000" \
		"00001";

	using namespace jlm;

	jive::bittype bt65(65);
	FunctionType ft({}, {&bt65});

	jive::bitvalue_repr vr(bs);

	ipgraph_module im(filepath(""), "", "");

	auto cfg = cfg::create(im);
	auto bb = basic_block::create(*cfg);
	bb->append_last(tac::create(jive::bitconstant_op(vr), {}));
	auto c = bb->last()->result(0);

	cfg->exit()->divert_inedges(bb);
	bb->add_outedge(cfg->exit());
	cfg->exit()->append_result(c);

	auto f = function_node::create(im.ipgraph(), "f", ft, linkage::external_linkage);
	f->add_cfg(std::move(cfg));

	print(im, stdout);

	llvm::LLVMContext ctx;
	auto lm = jlm2llvm::convert(im, ctx);

	print(*lm);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/backend/llvm/jlm-llvm/test-bitconstant", test)
