/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/cfg-node.hpp>
#include <jlm/ir/ipgraph.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/tac.hpp>

#include <algorithm>
#include <deque>
#include <sstream>
#include <unordered_map>

#include <stdio.h>
#include <stdlib.h>
#include <sstream>

namespace jlm {

/* cfg entry node */

entry_node::~entry_node()
{}

/* cfg exit node */

exit_node::~exit_node()
{}

/* cfg */

cfg::cfg(ipgraph_module & im)
: module_(im)
{
	entry_ = std::unique_ptr<entry_node>(new entry_node(*this));
	exit_ = std::unique_ptr<exit_node>(new exit_node(*this));
	entry_->add_outedge(exit_.get());
}

cfg::iterator
cfg::remove_node(cfg::iterator & nodeit)
{
	if (&nodeit->cfg() != this)
		throw jlm::error("node does not belong to this CFG.");

	for (auto it = nodeit->begin_inedges(); it != nodeit->end_inedges(); it++)
		if ((*it)->source() != nodeit.node())
			throw jlm::error("cannot remove node. It has still incoming edges.");

	nodeit->remove_outedges();
	std::unique_ptr<basic_block> tmp(nodeit.node());
	auto rit = iterator(std::next(nodes_.find(tmp)));
	nodes_.erase(tmp);
	tmp.release();
	return rit;
}

/* supporting functions */

std::vector<cfg_node*>
postorder(const jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_closed(cfg));

	std::function<void(
		cfg_node*,
		std::unordered_set<cfg_node*>&,
		std::vector<cfg_node*>&
	)> traverse = [&](
		cfg_node * node,
		std::unordered_set<cfg_node*> & visited,
		std::vector<cfg_node*> & nodes)
	{
		visited.insert(node);
		for (size_t n = 0; n < node->noutedges(); n++) {
			auto edge = node->outedge(n);
			if (visited.find(edge->sink()) == visited.end())
				traverse(edge->sink(), visited, nodes);
		}

		nodes.push_back(node);
	};

	std::vector<cfg_node*> nodes;
	std::unordered_set<cfg_node*> visited;
	traverse(cfg.entry(), visited, nodes);

	return nodes;
}

std::vector<cfg_node*>
reverse_postorder(const jlm::cfg & cfg)
{
	auto nodes = postorder(cfg);
	std::reverse(nodes.begin(), nodes.end());
	return nodes;
}

size_t
ntacs(const jlm::cfg & cfg)
{
	size_t ntacs = 0;
	for (auto & node : cfg) {
		if (auto bb = dynamic_cast<const basic_block*>(&node))
			ntacs += bb->ntacs();
	}

	return ntacs;
}

}
