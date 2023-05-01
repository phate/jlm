/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

static int
test()
{
	using namespace jlm;

	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto bb0 = basic_block::create(cfg);
	auto bb1 = basic_block::create(cfg);

	jive::ctlconstant_op op(jive::ctlvalue_repr(1, 2));
	bb0->append_last(tac::create(op, {}));
	bb0->append_last(branch_op::create(2, bb0->last()->result(0)));

	cfg.exit()->divert_inedges(bb0);
	bb0->add_outedge(bb1);
	bb0->add_outedge(cfg.exit());
	bb1->add_outedge(bb1);

	print_ascii(cfg, stdout);

	purge(cfg);

	assert(cfg.nnodes() == 2);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg-purge", test)
