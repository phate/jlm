/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/module.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>
#include <jlm/jlm/ir/ssa.hpp>
#include <jlm/jlm/ir/print.hpp>

static inline void
test_two_phis()
{
	using namespace jlm;

	jlm::valuetype vt;
	jlm::module module("", "");

	auto v1 = module.create_variable(vt, "vbl1");
	auto v2 = module.create_variable(vt, "vbl2");
	auto v3 = module.create_variable(vt, "vbl3");
	auto v4 = module.create_variable(vt, "vbl4");

	auto r1 = module.create_variable(vt, "r1");
	auto r2 = module.create_variable(vt, "r2");

	jlm::cfg cfg(module);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);
	auto bb3 = basic_block::create(cfg);
	auto bb4 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb1->add_outedge(bb3);
	bb2->add_outedge(bb4);
	bb3->add_outedge(bb4);
	bb4->add_outedge(cfg.exit());

	bb2->append_last(create_testop_tac({}, {v1}));
	bb2->append_last(create_testop_tac({}, {v3}));
	bb3->append_last(create_testop_tac({}, {v2}));
	bb3->append_last(create_testop_tac({}, {v4}));

	bb4->append_last(create_phi_tac({{v1, bb2}, {v2, bb3}}, r1));
	bb4->append_last(create_phi_tac({{v3, bb2}, {v4, bb3}}, r2));

//	jlm::view_ascii(cfg, stdout);

	jlm::destruct_ssa(cfg);

//	jlm::view_ascii(cfg, stdout);
}

static int
verify()
{
	test_two_phis();

	return 0;
}

JLM_UNIT_TEST_REGISTER("libjlm/test-ssa-destruction", verify);
