/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/print.hpp>

static void
test_single_operand_phi()
{
	using namespace jlm;

	valuetype vt;

	ipgraph_module im(filepath(""), "", "");

	jlm::cfg cfg(im);
	auto arg = cfg.entry()->append_argument(argument::create("arg", vt));

	auto bb0 = basic_block::create(cfg);
	bb0->append_first(phi_op::create({{arg, cfg.entry()}}, vt));

	cfg.exit()->divert_inedges(bb0);
	bb0->add_outedge(cfg.exit());
	cfg.exit()->append_result(bb0->last()->result(0));

	print_ascii(cfg, stdout);

	assert(is_valid(cfg));
}

static int
test()
{
	test_single_operand_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-cfg-validity", test)
