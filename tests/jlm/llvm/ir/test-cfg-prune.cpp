/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	test_op op({}, {&vt});

	/* setup cfg */

	ipgraph_module im(filepath(""), "", "");

	jlm::cfg cfg(im);
	auto arg = cfg.entry()->append_argument(argument::create("arg", vt));
	auto bb0 = basic_block::create(cfg);
	auto bb1 = basic_block::create(cfg);

	bb0->append_last(tac::create(op, {}));
	bb1->append_last(phi_op::create({{bb0->last()->result(0), bb0}, {arg, cfg.entry()}}, vt));

	cfg.exit()->divert_inedges(bb1);
	bb0->add_outedge(bb1);
	bb1->add_outedge(cfg.exit());
	cfg.exit()->append_result(bb1->last()->result(0));

	print_ascii(cfg, stdout);

	/* verify pruning */

	prune(cfg);
	print_ascii(cfg, stdout);

	assert(cfg.nnodes() == 1);

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg-prune", test)
