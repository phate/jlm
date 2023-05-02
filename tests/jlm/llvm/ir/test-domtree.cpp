/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/domtree.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>

#include <unordered_set>

template<size_t N> static void
check(
	const jlm::domnode * dnode,
	const jlm::cfg_node * node,
	const std::unordered_set<const jlm::cfg_node*> & children)
{
	assert(dnode->node() == node);
	assert(dnode->nchildren() == N);
	for (auto & child : *dnode)
		assert(children.find(child->node()) != children.end());
}

static const jlm::domnode *
get_child(
	const jlm::domnode * root,
	const jlm::cfg_node * node)
{
	for (const auto & child : *root) {
		if (child->node() == node)
			return child.get();
	}

	assert(0);
}


static int
test()
{
	using namespace jlm;

	ipgraph_module im(filepath(""), "", "");

	/* setup cfg */

	jlm::cfg cfg(im);
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

	/* verify domtree */

	auto root = domtree(cfg);
	check<1>(root.get(), cfg.entry(), {bb1});

	auto dtbb1 = root->child(0);
	check<3>(dtbb1, bb1, {bb2, bb3, bb4});

	auto dtbb2 = get_child(dtbb1, bb2);
	check<0>(dtbb2, bb2, {});

	auto dtbb3 = get_child(dtbb1, bb3);
	check<0>(dtbb3, bb3, {});

	auto dtbb4 = get_child(dtbb1, bb4);
	check<1>(dtbb4, bb4, {cfg.exit()});

	auto dtexit = dtbb4->child(0);
	check<0>(dtexit, cfg.exit(), {});

	return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-domtree", test)
