/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/print.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	test_op op({}, {&vt});

	/* setup cfg */

	ipgraph_module im(filepath(""), "", "");
	auto c = im.create_tacvariable(vt, "c");
	auto p = im.create_tacvariable(vt, "p");

	jlm::cfg cfg(im);
	auto arg = cfg.entry()->append_argument(argument::create("arg", vt));
	cfg.exit()->append_result(p);
	auto bb0 = basic_block::create(cfg);
	auto bb1 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb0->add_outedge(bb1);
	bb1->add_outedge(cfg.exit());

	taclist tlbb0, tlbb1;
	tlbb0.append_last(tac::create(op, {}, {c}));
	tlbb1.append_last(phi_op::create({{c,bb0}, {arg, cfg.entry()}}, p));

	bb0->append_first(tlbb0);
	bb1->append_first(tlbb1);

	print_ascii(cfg, stdout);

	/* verify pruning */

	prune(cfg);
	print_ascii(cfg, stdout);

	assert(cfg.nnodes() == 1);

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/ir/test-cfg-prune", test)
