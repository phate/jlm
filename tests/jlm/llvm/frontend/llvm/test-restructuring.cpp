/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/frontend/ControlFlowRestructuring.hpp>
#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

#include <assert.h>

static inline void
test_acyclic_structured()
{
	using namespace jlm;

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

//	jlm::view_ascii(cfg, stdout);

	size_t nnodes = cfg.nnodes();
  RestructureBranches(&cfg);

//	jlm::view_ascii(cfg, stdout);

	assert(nnodes == cfg.nnodes());
}

static inline void
test_acyclic_unstructured()
{
	using namespace jlm;

	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);
	auto bb3 = basic_block::create(cfg);
	auto bb4 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb1->add_outedge(bb3);
	bb2->add_outedge(bb3);
	bb2->add_outedge(bb4);
	bb3->add_outedge(bb4);
	bb4->add_outedge(cfg.exit());

//	jlm::view_ascii(cfg, stdout);

  RestructureBranches(&cfg);

//	jlm::view_ascii(cfg, stdout);

	assert(is_proper_structured(cfg));
}

static inline void
test_dowhile()
{
	using namespace jlm;

	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);
	auto bb3 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb2->add_outedge(bb2);
	bb2->add_outedge(bb3);
	bb3->add_outedge(bb1);
	bb3->add_outedge(cfg.exit());

//	jlm::view_ascii(cfg, stdout);

	size_t nnodes = cfg.nnodes();
  RestructureControlFlow(&cfg);

//	jlm::view_ascii(cfg, stdout);

	assert(nnodes == cfg.nnodes());
	assert(bb2->outedge(0)->sink() == bb2);
	assert(bb3->outedge(0)->sink() == bb1);
}

static inline void
test_while()
{
	using namespace jlm;

	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(cfg.exit());
	bb1->add_outedge(bb2);
	bb2->add_outedge(bb1);

//	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(&cfg);

	/* FIXME: Nodes are not printed in the right order */
//	jlm::view_ascii(cfg, stdout);

	assert(is_proper_structured(cfg));
}

static inline void
test_irreducible()
{
	using namespace jlm;

	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);
	auto bb3 = basic_block::create(cfg);
	auto bb4 = basic_block::create(cfg);
	auto bb5 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb1->add_outedge(bb3);
	bb2->add_outedge(bb4);
	bb2->add_outedge(bb3);
	bb3->add_outedge(bb2);
	bb3->add_outedge(bb5);
	bb4->add_outedge(cfg.exit());
	bb5->add_outedge(cfg.exit());

//	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(&cfg);

//	jlm::view_ascii(cfg, stdout);
	assert(is_proper_structured(cfg));
}

static inline void
test_acyclic_unstructured_in_dowhile()
{
	using namespace jlm;

	ipgraph_module module(filepath(""), "", "");

	jlm::cfg cfg(module);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);
	auto bb3 = basic_block::create(cfg);
	auto bb4 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(bb3);
	bb1->add_outedge(bb2);
	bb2->add_outedge(bb3);
	bb2->add_outedge(bb4);
	bb3->add_outedge(bb4);
	bb4->add_outedge(bb1);
	bb4->add_outedge(cfg.exit());

//	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(&cfg);

//	jlm::view_ascii(cfg, stdout);
	assert(is_proper_structured(cfg));
}

static inline void
test_lor_before_dowhile()
{
	using namespace jlm;

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
	bb2->add_outedge(bb3);
	bb3->add_outedge(bb4);
	bb4->add_outedge(cfg.exit());
	bb4->add_outedge(bb4);

//	jlm::view_ascii(cfg, stdout);

  RestructureControlFlow(&cfg);

//	jlm::view_ascii(cfg, stdout);
	assert(is_proper_structured(cfg));
}

static void
test_static_endless_loop()
{
	using namespace jlm;

	ipgraph_module im(filepath(""), "", "");

	jlm::cfg cfg(im);
	auto bb1 = basic_block::create(cfg);
	auto bb2 = basic_block::create(cfg);

	cfg.exit()->divert_inedges(bb1);
	bb1->add_outedge(bb2);
	bb1->add_outedge(bb1);
	bb1->add_outedge(cfg.exit());
	bb2->add_outedge(bb2);

//	jlm::print_dot(cfg, stdout);

  RestructureControlFlow(&cfg);

//	jlm::print_dot(cfg, stdout);
	assert(is_proper_structured(cfg));
}

static int
verify()
{
	test_acyclic_structured();
	test_acyclic_unstructured();
	test_dowhile();
	test_while();
	test_irreducible();
	test_acyclic_unstructured_in_dowhile();
	test_lor_before_dowhile();
	test_static_endless_loop();

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/frontend/llvm/test-restructuring", verify)
