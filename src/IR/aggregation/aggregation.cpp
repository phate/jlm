/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/aggregation/aggregation.hpp>
#include <jlm/IR/aggregation/node.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/tac.hpp>

#include <deque>
#include <algorithm>

namespace jlm {
namespace agg {

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
	for (const auto & outedge : split->outedges()) {
		if (outedge->sink()->ninedges() != 1)
			return false;
		if (outedge->sink()->noutedges() != 1)
			return false;
		if (outedge->sink()->outedge(0)->sink() != join)
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
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
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
	for (const auto & outedge : exit->outedges())
		reduction->add_outedge(outedge->sink(), outedge->index());
	exit->remove_outedges();

	map[reduction] = create_linear_node(std::move(map[entry]), std::move(map[exit]));
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
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(is_loop(node));
	JLM_DEBUG_ASSERT(map.find(node) != map.end());

	/* perform reduction */
	auto reduction = create_basic_block_node(node->cfg());
	for (auto & outedge : node->outedges()) {
		if (outedge->is_selfloop()) {
			node->remove_outedge(outedge);
			break;
		}
	}
	reduction->add_outedge(node->outedge(0)->sink(), 0);
	node->remove_outedges();
	node->divert_inedges(reduction);

	map[reduction] = create_loop_node(std::move(map[node]));
	map.erase(node);
	to_visit.erase(node);
	to_visit.insert(reduction);

	return reduction;
}

static inline jlm::cfg_node *
reduce_branch(
	jlm::cfg_node * split,
	std::unordered_set<cfg_node*> & to_visit,
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(split->noutedges() > 1);
	JLM_DEBUG_ASSERT(split->outedge(0)->sink()->noutedges() == 1);
	JLM_DEBUG_ASSERT(map.find(split) != map.end());

	auto join = split->outedge(0)->sink()->outedge(0)->sink();
	for (const auto & outedge : split->outedges()) {
		JLM_DEBUG_ASSERT(outedge->sink()->ninedges() == 1);
		JLM_DEBUG_ASSERT(map.find(outedge->sink()) != map.end());
		JLM_DEBUG_ASSERT(outedge->sink()->noutedges() == 1);
		JLM_DEBUG_ASSERT(outedge->sink()->outedge(0)->sink() == join);
	}

	/* perform reduction */
	auto reduction = create_basic_block_node(split->cfg());
	split->divert_inedges(reduction);
	join->remove_inedges();
	reduction->add_outedge(join, 0);

	auto branch = create_branch_node(std::move(map[split]));
	for (const auto & outedge : split->outedges()) {
		branch->add_child(std::move(map[outedge->sink()]));
		map.erase(outedge->sink());
		to_visit.erase(outedge->sink());
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
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
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
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
{
	auto it = to_visit.begin();
	while (it != to_visit.end())	{
		bool reduced = reduce(*it, to_visit, map);
		it = reduced ? to_visit.begin() : std::next(it);
	}

	JLM_DEBUG_ASSERT(to_visit.size() == 1);
}

std::unique_ptr<agg::node>
aggregate(jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(cfg.is_structured());

	/* insert all aggregation leaves into the map */
	std::unordered_set<cfg_node*> to_visit;
	std::unordered_map<jlm::cfg_node*, std::unique_ptr<agg::node>> map;
	for (auto & node : cfg) {
		if (is_basic_block(&node))
			map[&node] = create_block_node(*static_cast<const basic_block*>(&node.attribute()));
		else if (is_entry_node(&node))
			map[&node] = create_entry_node(*static_cast<const entry_attribute*>(&node.attribute()));
		else if (is_exit_node(&node))
			map[&node] = create_exit_node(*static_cast<const exit_attribute*>(&node.attribute()));
		else
			JLM_DEBUG_ASSERT(0);
		to_visit.insert(&node);
	}

	aggregate(to_visit, map);
	JLM_DEBUG_ASSERT(map.size() == 1);

	return std::move(std::move(map.begin()->second));
}

}}
