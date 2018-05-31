/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/ir/aggregation.hpp>
#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/cfg-structure.hpp>
#include <jlm/jlm/ir/cfg-node.hpp>
#include <jlm/jlm/ir/tac.hpp>

#include <algorithm>
#include <deque>
#include <unordered_map>

namespace jlm {

/* aggnode class */

aggnode::~aggnode()
{}

/* entryaggnode class */

entryaggnode::~entryaggnode()
{}

std::string
entryaggnode::debug_string() const
{
	return "entry";
}

/* exitaggnode class */

exitaggnode::~exitaggnode()
{}

std::string
exitaggnode::debug_string() const
{
	return "exit";
}

/* blockaggnode class */

blockaggnode::~blockaggnode()
{}

std::string
blockaggnode::debug_string() const
{
	return "block";
}

/* linearaggnode class */

linearaggnode::~linearaggnode()
{}

std::string
linearaggnode::debug_string() const
{
	return "linear";
}

/* branchaggnode class */

branchaggnode::~branchaggnode()
{}

std::string
branchaggnode::debug_string() const
{
	return "branch";
}

/* loopaggnode class */

loopaggnode::~loopaggnode()
{}

std::string
loopaggnode::debug_string() const
{
	return "loop";
}

/* aggregation */

static inline bool
is_loop(const cfg_node * node) noexcept
{
	return node->ninedges() == 2
	    && node->noutedges() == 2
	    && node->has_selfloop_edge();
}

static inline bool
is_branch(const cfg_node * split) noexcept
{
	if (split->noutedges() < 2)
		return false;

	if (split->outedge(0)->sink()->noutedges() != 1)
		return false;

	auto join = split->outedge(0)->sink()->outedge(0)->sink();
	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++) {
		if (it->sink()->ninedges() != 1)
			return false;
		if (it->sink()->noutedges() != 1)
			return false;
		if (it->sink()->outedge(0)->sink() != join)
			return false;
	}

	return true;
}

static inline bool
is_linear(const cfg_node * node) noexcept
{
	if (node->noutedges() != 1)
		return false;

	auto exit = node->outedge(0)->sink();
	if (exit->ninedges() != 1)
		return false;

	return true;
}

static inline jlm::cfg_node *
reduce_linear(
	jlm::cfg_node * entry,
	std::unordered_set<cfg_node*> & to_visit,
	std::unordered_map<cfg_node*, std::unique_ptr<aggnode>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(entry->noutedges() == 1);
	JLM_DEBUG_ASSERT(map.find(entry) != map.end());

	auto exit = entry->outedge(0)->sink();
	JLM_DEBUG_ASSERT(exit->ninedges() == 1);
	JLM_DEBUG_ASSERT(map.find(exit) != map.end());

	/* perform reduction */
	auto reduction = create_basic_block_node(entry->cfg());
	entry->divert_inedges(reduction);
	for (auto it = exit->begin_outedges(); it != exit->end_outedges(); it++)
		reduction->add_outedge(it->sink());
	exit->remove_outedges();

	map[reduction] = linearaggnode::create(std::move(map[entry]), std::move(map[exit]));
	map.erase(entry);
	map.erase(exit);
	to_visit.erase(entry);
	to_visit.erase(exit);
	to_visit.insert(reduction);

	return reduction;
}

static inline jlm::cfg_node *
reduce_loop(
	jlm::cfg_node * node,
	std::unordered_set<cfg_node*> & to_visit,
	std::unordered_map<cfg_node*, std::unique_ptr<aggnode>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(is_loop(node));
	JLM_DEBUG_ASSERT(map.find(node) != map.end());

	/* perform reduction */
	auto reduction = create_basic_block_node(node->cfg());
	for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
		if (it->is_selfloop()) {
			node->remove_outedge(it->index());
			break;
		}
	}
	reduction->add_outedge(node->outedge(0)->sink());
	node->remove_outedges();
	node->divert_inedges(reduction);

	map[reduction] = loopaggnode::create(std::move(map[node]));
	map.erase(node);
	to_visit.erase(node);
	to_visit.insert(reduction);

	return reduction;
}

static inline jlm::cfg_node *
reduce_branch(
	jlm::cfg_node * split,
	std::unordered_set<cfg_node*> & to_visit,
	std::unordered_map<cfg_node*, std::unique_ptr<aggnode>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(split->noutedges() > 1);
	JLM_DEBUG_ASSERT(split->outedge(0)->sink()->noutedges() == 1);
	JLM_DEBUG_ASSERT(map.find(split) != map.end());

	auto join = split->outedge(0)->sink()->outedge(0)->sink();
	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++) {
		JLM_DEBUG_ASSERT(it->sink()->ninedges() == 1);
		JLM_DEBUG_ASSERT(map.find(it->sink()) != map.end());
		JLM_DEBUG_ASSERT(it->sink()->noutedges() == 1);
		JLM_DEBUG_ASSERT(it->sink()->outedge(0)->sink() == join);
	}

	/* perform reduction */
	auto reduction = create_basic_block_node(split->cfg());
	split->divert_inedges(reduction);
	join->remove_inedges();
	reduction->add_outedge(join);

	auto branch = branchaggnode::create(std::move(map[split]));
	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++) {
		branch->add_child(std::move(map[it->sink()]));
		map.erase(it->sink());
		to_visit.erase(it->sink());
	}

	map[reduction] = std::move(branch);
	map.erase(split);
	to_visit.erase(split);
	to_visit.insert(reduction);

	return reduction;
}

static inline bool
reduce(
	jlm::cfg_node * node,
	std::unordered_set<cfg_node*> & to_visit,
	std::unordered_map<cfg_node*, std::unique_ptr<aggnode>> & map)
{
	if (is_loop(node)) {
		reduce_loop(node, to_visit, map);
		return true;
	}

	if (is_branch(node)) {
		reduce_branch(node, to_visit, map);
		return true;
	}

	if (is_linear(node)) {
		reduce_linear(node, to_visit, map);
		return true;
	}

	return false;
}

static inline void
aggregate(
	std::unordered_set<jlm::cfg_node*> & to_visit,
	std::unordered_map<cfg_node*, std::unique_ptr<aggnode>> & map)
{
	auto it = to_visit.begin();
	while (it != to_visit.end())	{
		bool reduced = reduce(*it, to_visit, map);
		it = reduced ? to_visit.begin() : std::next(it);
	}

	JLM_DEBUG_ASSERT(to_visit.size() == 1);
}

std::unique_ptr<aggnode>
aggregate(jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_proper_structured(cfg));

	/* insert all aggregation leaves into the map */
	std::unordered_set<cfg_node*> to_visit;
	std::unordered_map<jlm::cfg_node*, std::unique_ptr<aggnode>> map;
	for (auto & node : cfg) {
		if (is_basic_block(node.attribute()))
			map[&node] = blockaggnode::create(std::move(*static_cast<basic_block*>(&node.attribute())));
		else if (is_entry_node(&node))
			map[&node] = entryaggnode::create(*static_cast<const jlm::entry*>(&node.attribute()));
		else if (is_exit_node(&node))
			map[&node] = exitaggnode::create(*static_cast<const jlm::exit*>(&node.attribute()));
		else
			JLM_DEBUG_ASSERT(0);
		to_visit.insert(&node);
	}

	aggregate(to_visit, map);
	JLM_DEBUG_ASSERT(map.size() == 1);

	return std::move(std::move(map.begin()->second));
}

}
