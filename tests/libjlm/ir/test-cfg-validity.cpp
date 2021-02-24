/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/print.hpp>

static void
test_single_operand_phi()
{
	using namespace jlm;

	valuetype vt;

	ipgraph_module im(filepath(""), "", "");
	auto p = im.create_tacvariable(vt, "p");

	jlm::cfg cfg(im);
	auto arg = cfg.entry()->append_argument(argument::create("arg", vt));
	cfg.exit()->append_result(p);

	auto bb0 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb0);
	bb0->add_outedge(cfg.exit());

	taclist tlbb0;
	tlbb0.append_last(phi_op::create({{arg, cfg.entry()}}, p));
	bb0->append_first(tlbb0);

	print_ascii(cfg, stdout);

	assert(is_valid(cfg));
}

static int
test()
{
	test_single_operand_phi();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/test-cfg-validity", test)
