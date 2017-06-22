/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/cfg_node.hpp>
#include <jlm/ir/clg.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/tac.hpp>

#include <algorithm>
#include <deque>
#include <sstream>
#include <unordered_map>

#include <stdio.h>
#include <stdlib.h>
#include <sstream>

namespace jlm {

/* entry attribute */

static inline jlm::cfg_node *
create_entry_node(jlm::cfg * cfg)
{
	jlm::entry_attribute attr;
	return cfg->create_node(attr);
}

entry_attribute::~entry_attribute()
{}

std::string
entry_attribute::debug_string() const noexcept
{
	return "ENTRY";
}

std::unique_ptr<attribute>
entry_attribute::copy() const
{
	return std::unique_ptr<attribute>(new entry_attribute(*this));
}

/* exit attribute */

static inline jlm::cfg_node *
create_exit_node(jlm::cfg * cfg)
{
	jlm::exit_attribute attr;
	return cfg->create_node(attr);
}

exit_attribute::~exit_attribute()
{}

std::string
exit_attribute::debug_string() const noexcept
{
	return "EXIT";
}

std::unique_ptr<attribute>
exit_attribute::copy() const
{
	return std::unique_ptr<attribute>(new exit_attribute(*this));
}

/* cfg */

cfg::cfg(jlm::module & module)
: module_(module)
{
	entry_ = create_entry_node(this);
	exit_ = create_exit_node(this);
	entry_->add_outedge(exit_);
}

cfg_node *
cfg::create_node(const attribute & attr)
{
	auto node = std::make_unique<cfg_node>(*this, attr);
	auto tmp = node.get();
	nodes_.insert(std::move(node));
	return tmp;
}

std::string
cfg::convert_to_dot() const
{
	std::string dot("digraph cfg{\n");
	for (const auto & node : nodes_) {
		dot += strfmt((intptr_t)node.get());
		dot += strfmt("[shape = box, label = \"", node->debug_string(), "\"];\n");
		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			dot += strfmt((intptr_t)it->source(), " -> ", (intptr_t)it->sink());
			dot += strfmt("[label = \"", it->index(), "\"];\n");
		}
	}
	dot += "}\n";

	return dot;
}

void
cfg::remove_node(cfg_node * node)
{
	node->remove_inedges();
	node->remove_outedges();

	std::unique_ptr<cfg_node> tmp(node);
	std::unordered_set<std::unique_ptr<cfg_node>>::const_iterator it = nodes_.find(tmp);
	JLM_DEBUG_ASSERT(it != nodes_.end());
	nodes_.erase(it);
	tmp.release();
}

void
cfg::prune()
{
	JLM_DEBUG_ASSERT(is_valid(*this));

	/* find all nodes that are dominated by the entry node */
	std::unordered_set<cfg_node*> to_visit({entry_node()});
	std::unordered_set<cfg_node*> visited;
	while (!to_visit.empty()) {
		cfg_node * node = *to_visit.begin();
		to_visit.erase(to_visit.begin());
		JLM_DEBUG_ASSERT(visited.find(node) == visited.end());
		visited.insert(node);
		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			if (visited.find(it->sink()) == visited.end()
			&& to_visit.find(it->sink()) == to_visit.end())
				to_visit.insert(it->sink());
		}
	}

	/* remove all nodes not dominated by the entry node */
	std::unordered_set<std::unique_ptr<cfg_node>>::iterator it = nodes_.begin();
	while (it != nodes_.end()) {
		if (visited.find((*it).get()) == visited.end()) {
			cfg_node * node = (*it).get();
			node->remove_inedges();
			node->remove_outedges();
			it = nodes_.erase(it);
		} else
			it++;
	}

	JLM_DEBUG_ASSERT(is_closed(*this));
}

}

void
jive_cfg_view(const jlm::cfg & cfg)
{
	FILE * file = popen("tee /tmp/cfg.dot | dot -Tps > /tmp/cfg.ps ; gv /tmp/cfg.ps", "w");
	auto dot = cfg.convert_to_dot();
	fwrite(dot.c_str(), dot.size(), 1, file);
	pclose(file);
}
