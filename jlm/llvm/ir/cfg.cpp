/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <algorithm>
#include <deque>
#include <unordered_map>

namespace jlm {

/* argument */

argument::~argument()
{}

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
	auto & cfg = nodeit->cfg();

	for (auto & inedge : nodeit->inedges()) {
		if (inedge->source() != nodeit.node())
			throw jlm::error("cannot remove node. It has still incoming edges.");
	}

	nodeit->remove_outedges();
	std::unique_ptr<basic_block> tmp(nodeit.node());
	auto rit = iterator(std::next(cfg.nodes_.find(tmp)));
	cfg.nodes_.erase(tmp);
	tmp.release();
	return rit;
}

cfg::iterator
cfg::remove_node(basic_block * bb)
{
	auto & cfg = bb->cfg();

	auto it = cfg.find_node(bb);
	return remove_node(it);
}

/* supporting functions */

std::vector<cfg_node*>
postorder(const jlm::cfg & cfg)
{
	JLM_ASSERT(is_closed(cfg));

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

std::vector<cfg_node*>
breadth_first(const jlm::cfg & cfg)
{
	std::deque<jlm::cfg_node*> next({cfg.entry()});
	std::vector<jlm::cfg_node*> nodes({cfg.entry()});
	std::unordered_set<jlm::cfg_node*> visited({cfg.entry()});
	while (!next.empty()) {
		auto node = next.front();
		next.pop_front();

		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			if (visited.find(it->sink()) == visited.end()) {
				visited.insert(it->sink());
				next.push_back(it->sink());
				nodes.push_back(it->sink());
			}
		}
	}

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
