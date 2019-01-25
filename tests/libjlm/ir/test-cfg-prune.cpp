/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/cfg-structure.hpp>
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/print.hpp>

static int
test()
{
	using namespace jlm;

	valuetype vt;
	test_op op({}, {&vt});

	/* setup cfg */

	jlm::module module("", "");
	auto arg = module.create_variable(vt, "arg");
	auto c = module.create_variable(vt, "c");
	auto p = module.create_variable(vt, "p");

	jlm::cfg cfg(module);
	cfg.entry()->append_argument(arg);
	cfg.exit()->append_result(p);
	auto bb0 = basic_block::create(cfg);
	auto bb1 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb0->add_outedge(bb1);
	bb1->add_outedge(cfg.exit());

	taclist tlbb0, tlbb1;
	tlbb0.append_last(tac::create(op, {}, {c}));
	tlbb1.append_last(create_phi_tac({{c,bb0}, {arg, cfg.entry()}}, p));

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
