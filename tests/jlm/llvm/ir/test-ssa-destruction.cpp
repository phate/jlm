/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"
#include "test-registry.hpp"
#include "test-types.hpp"

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/ssa.hpp>
#include <jlm/llvm/ir/print.hpp>

static inline void
test_two_phis()
{
	using namespace jlm;

	jlm::valuetype vt;
	ipgraph_module module(filepath(""), "", "");

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

	bb2->append_last(create_testop_tac({}, {&vt}));
	auto v1 = bb2->last()->result(0);

	bb2->append_last(create_testop_tac({}, {&vt}));
	auto v3 = bb2->last()->result(0);

	bb3->append_last(create_testop_tac({}, {&vt}));
	auto v2 = bb3->last()->result(0);

	bb3->append_last(create_testop_tac({}, {&vt}));
	auto v4 = bb3->last()->result(0);

	bb4->append_last(phi_op::create({{v1, bb2}, {v2, bb3}}, vt));
	bb4->append_last(phi_op::create({{v3, bb2}, {v4, bb3}}, vt));

	jlm::print_ascii(cfg, stdout);

	jlm::destruct_ssa(cfg);

	jlm::print_ascii(cfg, stdout);
}

static int
verify()
{
	test_two_phis();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-ssa-destruction", verify)
