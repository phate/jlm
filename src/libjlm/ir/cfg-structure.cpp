/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/cfg.hpp>
#include <jlm/jlm/ir/cfg-structure.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>

#include <algorithm>
#include <unordered_map>

/* Tarjan's SCC algorithm */

static void
strongconnect(
	jlm::cfg_node * node,
	jlm::cfg_node * exit,
	std::unordered_map<jlm::cfg_node*, std::pair<size_t,size_t>> & map,
	std::vector<jlm::cfg_node*> & node_stack,
	size_t & index,
	std::vector<std::unordered_set<jlm::cfg_node*>> & sccs)
{
	map.emplace(node, std::make_pair(index, index));
	node_stack.push_back(node);
	index++;

	if (node != exit) {
		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			auto successor = it->sink();
			if (map.find(successor) == map.end()) {
				/* successor has not been visited yet; recurse on it */
				strongconnect(successor, exit, map, node_stack, index, sccs);
				map[node].second = std::min(map[node].second, map[successor].second);
			} else if (std::find(node_stack.begin(), node_stack.end(), successor) != node_stack.end()) {
				/* successor is in stack and hence in the current SCC */
				map[node].second = std::min(map[node].second, map[successor].first);
			}
		}
	}

	if (map[node].second == map[node].first) {
		std::unordered_set<jlm::cfg_node*> scc;
		jlm::cfg_node * w;
		do {
			w = node_stack.back();
			node_stack.pop_back();
			scc.insert(w);
		} while (w != node);

		if (scc.size() != 1 || (*scc.begin())->has_selfloop_edge())
			sccs.push_back(scc);
	}
}

static inline std::unique_ptr<jlm::cfg>
copy_structural(const jlm::cfg & in)
{
	std::unique_ptr<jlm::cfg> out(new jlm::cfg(in.module()));
	out->entry_node()->remove_outedge(0);

	/* create all nodes */
	std::unordered_map<const jlm::cfg_node*, jlm::cfg_node*> node_map;
	for (const auto & node : in) {
		if (&node == in.entry_node()) {
			node_map[&node] = out->entry_node();
		} else if (&node == in.exit_node()) {
			node_map[&node] = out->exit_node();
		} else {
			JLM_DEBUG_ASSERT(is_basic_block(node.attribute()));
			node_map[&node] = create_basic_block_node(out.get());
		}
	}

	/* establish control flow */
	for (const auto & node : in) {
		for (auto it = node.begin_outedges(); it != node.end_outedges(); it++)
			node_map[&node]->add_outedge(node_map[it->sink()]);
	}

	return out;
}

static inline bool
is_loop(const jlm::cfg_node * node) noexcept
{
	return node->ninedges() == 2
	    && node->noutedges() == 2
	    && node->has_selfloop_edge();
}

static inline bool
is_linear_reduction(const jlm::cfg_node * node) noexcept
{
	if (node->noutedges() != 1)
		return false;

	if (node->outedge(0)->sink()->ninedges() != 1)
		return false;

	return true;
}

static inline jlm::cfg_node *
find_join(const jlm::cfg_node * split) noexcept
{
	JLM_DEBUG_ASSERT(split->noutedges() > 1);
	auto s1 = split->outedge(0)->sink();
	auto s2 = split->outedge(1)->sink();

	jlm::cfg_node * join = nullptr;
	if (s1->noutedges() == 1 && s1->outedge(0)->sink() == s2)
		join = s2;
	else if (s2->noutedges() == 1 && s2->outedge(0)->sink() == s1)
		join = s1;
	else if (s1->noutedges() == 1 && s2->noutedges() == 1
	     && (s1->outedge(0)->sink() == s2->outedge(0)->sink()))
		join = s1->outedge(0)->sink();

	return join;
}

static inline bool
is_branch(const jlm::cfg_node * split) noexcept
{
	if (split->noutedges() < 2)
		return false;

	auto join = find_join(split);
	if (join == nullptr || join->ninedges() != split->noutedges())
		return false;

	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++) {
		if (it->sink() == join)
			continue;

		auto node = it->sink();
		if (node->ninedges() != 1)
			return false;
		if (node->noutedges() != 1 || node->outedge(0)->sink() != join)
			return false;
	}

	return true;
}

static inline bool
is_proper_branch(const jlm::cfg_node * split) noexcept
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
is_T1(const jlm::cfg_node * node) noexcept
{
	for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
		if (it->source() == it->sink())
			return true;
	}

	return false;
}

static inline bool
is_T2(const jlm::cfg_node * node) noexcept
{
	if (node->ninedges() == 0)
		return false;

	auto source = (*node->begin_inedges())->source();
	for (auto it = node->begin_inedges(); it != node->end_inedges(); it++) {
		if ((*it)->source() != source)
			return false;
	}

	return true;
}

static inline void
reduce_loop(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_loop(node));
	auto cfg = node->cfg();

	auto reduction = create_basic_block_node(cfg);
	for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
		if (it->is_selfloop()) {
			node->remove_outedge(it->index());
			break;
		}
	}

	reduction->add_outedge(node->outedge(0)->sink());
	node->remove_outedges();
	node->divert_inedges(reduction);

	to_visit.erase(node);
	to_visit.insert(reduction);
}

static inline void
reduce_linear(
	jlm::cfg_node * entry,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_linear_reduction(entry));
	auto exit = entry->outedge(0)->sink();
	auto cfg = entry->cfg();

	auto reduction = create_basic_block_node(cfg);
	entry->divert_inedges(reduction);
	for (auto it = exit->begin_outedges(); it != exit->end_outedges(); it++)
		reduction->add_outedge(it->sink());
	exit->remove_outedges();

	to_visit.erase(entry);
	to_visit.erase(exit);
	to_visit.insert(reduction);
}

static inline void
reduce_branch(
	jlm::cfg_node * split,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_branch(split));
	auto join = find_join(split);
	auto cfg = split->cfg();

	auto reduction = create_basic_block_node(cfg);
	split->divert_inedges(reduction);
	reduction->add_outedge(join);
	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++) {
		if (it->sink() != join) {
			it->sink()->remove_outedges();
			to_visit.erase(it->sink());
		}
	}
	split->remove_outedges();

	to_visit.erase(split);
	to_visit.insert(reduction);
}

static inline void
reduce_proper_branch(
	jlm::cfg_node * split,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_proper_branch(split));
	auto join = split->outedge(0)->sink()->outedge(0)->sink();

	auto reduction = create_basic_block_node(split->cfg());
	split->divert_inedges(reduction);
	join->remove_inedges();
	reduction->add_outedge(join);
	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++)
		to_visit.erase(it->sink());

	to_visit.erase(split);
	to_visit.insert(reduction);
}

static inline void
reduce_T1(jlm::cfg_node * node)
{
	JLM_DEBUG_ASSERT(is_T1(node));

	for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
		if (it->source() == it->sink()) {
			node->remove_outedge(it->index());
			break;
		}
	}
}

static inline void
reduce_T2(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_DEBUG_ASSERT(is_T2(node));

	auto p = (*node->begin_inedges())->source();
	p->divert_inedges(node);
	p->remove_outedges();
	to_visit.erase(p);
}

static inline bool
reduce_proper_structured(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	if (is_loop(node)) {
		reduce_loop(node, to_visit);
		return true;
	}

	if (is_proper_branch(node)) {
		reduce_proper_branch(node, to_visit);
		return true;
	}

	if (is_linear_reduction(node)) {
		reduce_linear(node, to_visit);
		return true;
	}

	return false;
}

static inline bool
reduce_structured(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	if (is_loop(node)) {
		reduce_loop(node, to_visit);
		return true;
	}

	if (is_branch(node)) {
		reduce_branch(node, to_visit);
		return true;
	}

	if (is_linear_reduction(node)) {
		reduce_linear(node, to_visit);
		return true;
	}

	return false;
}

static inline bool
reduce_reducible(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	if (is_T1(node)) {
		reduce_T1(node);
		return true;
	}

	if (is_T2(node)) {
		reduce_T2(node, to_visit);
		return true;
	}

	return false;
}

namespace jlm {

static bool
check_phis(const basic_block * bb)
{
	for (auto it = bb->begin(); it != bb->end(); it++) {
		auto tac = *it;
		if (!is<phi_op>(tac))
			continue;

		/*
			Ensure that all phi nodes do not have for the same basic block
			multiple incoming variables.
		*/
		auto phi = static_cast<const phi_op*>(&tac->operation());
		std::unordered_map<cfg_node*, const variable*> map;
		for (size_t n = 0; n < tac->ninputs(); n++) {
			auto mit = map.find(phi->node(n));
			if (mit != map.end() && mit->second != tac->input(n))
				return false;

			if (mit == map.end())
				map[phi->node(n)] = tac->input(n);
		}

		/* ensure all phi nodes are at the beginning of a basic block */
		if (tac != bb->first() && !is<phi_op>(*std::prev(it)))
			return false;

	}

	return true;
}

bool
is_valid(const jlm::cfg & cfg)
{
	for (const auto & node : cfg) {
		if (&node == cfg.exit_node()) {
			if (!node.no_successor())
				return false;
			continue;
		}

		if (&node == cfg.entry_node()) {
			if (!node.no_predecessor())
				return false;
			if (!node.single_successor())
				return false;
			if (node.outedge(0)->index() != 0)
				return false;
			continue;
		}

		if (node.no_successor())
			return false;

		JLM_DEBUG_ASSERT(is_basic_block(node.attribute()));
		auto bb = static_cast<const basic_block*>(&node.attribute());
		if (!check_phis(bb))
			return false;
	}

	return true;
}

bool
is_closed(const jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_valid(cfg));

	for (const auto & node : cfg) {
		if (&node == cfg.entry_node())
			continue;

		if (node.no_predecessor())
			return false;
	}

	return true;
}

bool
is_linear(const jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_closed(cfg));

	for (const auto & node : cfg) {
		if (&node == cfg.entry_node() || &node == cfg.exit_node())
			continue;

		if (!node.single_successor() || !node.single_predecessor())
			return false;
	}

	return true;
}

std::vector<std::unordered_set<cfg_node*>>
find_sccs(const jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_closed(cfg));

	size_t index = 0;
	std::vector<cfg_node*> node_stack;
	std::vector<std::unordered_set<cfg_node*>> sccs;
	std::unordered_map<cfg_node*, std::pair<size_t,size_t>> map;
	strongconnect(cfg.entry_node(), cfg.exit_node(), map, node_stack, index, sccs);

	return sccs;
}

static inline bool
reduce(
	const jlm::cfg & cfg,
	const std::function<bool(jlm::cfg_node*, std::unordered_set<jlm::cfg_node*>&)> & f)
{
	JLM_DEBUG_ASSERT(is_closed(cfg));
	auto c = copy_structural(cfg);

	std::unordered_set<cfg_node*> to_visit;
	for (auto & node : *c)
		to_visit.insert(&node);

	auto it = to_visit.begin();
	while (it != to_visit.end()) {
		bool reduced = f(*it, to_visit);
		it = reduced ? to_visit.begin() : std::next(it);
	}

	return to_visit.size() == 1;
}

bool
is_structured(const jlm::cfg & cfg)
{
	return reduce(cfg, reduce_structured);
}

bool
is_proper_structured(const jlm::cfg & cfg)
{
	return reduce(cfg, reduce_proper_structured);
}

bool
is_reducible(const jlm::cfg & cfg)
{
	return reduce(cfg, reduce_reducible);
}

void
straighten(jlm::cfg & cfg)
{
	auto it = cfg.begin();
	while (it != cfg.end()) {
		if (is_linear_reduction(it.node())
		&& is_basic_block(it.node()->attribute())
		&& is_basic_block(it->outedge(0)->sink()->attribute())) {
			append_first(it->outedge(0)->sink(), it.node());
			it->divert_inedges(it->outedge(0)->sink());
			it = cfg.remove_node(it);
		} else {
			it++;
		}
	}
}

void
purge(jlm::cfg & cfg)
{
	auto it = cfg.begin();
	while (it != cfg.end()) {
		auto bb = dynamic_cast<const jlm::basic_block*>(&it.node()->attribute());

		if (bb && bb->ntacs() == 0) {
			JLM_DEBUG_ASSERT(it.node()->noutedges() == 1);
			it.node()->divert_inedges(it.node()->outedge(0)->sink());
			it = cfg.remove_node(it);
		} else {
			it++;
		}
	}
}

void
prune(jlm::cfg & cfg)
{
	JLM_DEBUG_ASSERT(is_valid(cfg));

	/* find all nodes that are dominated by the entry node */
	std::unordered_set<cfg_node*> visited;
	std::unordered_set<cfg_node*> to_visit({cfg.entry_node()});
	while (!to_visit.empty()) {
		auto node = *to_visit.begin();
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
	auto it = cfg.begin();
	while (it != cfg.end()) {
		if (visited.find(it.node()) == visited.end()) {
			it->remove_inedges();
			it = cfg.remove_node(it);
		} else {
			it++;
		}
	}

	JLM_DEBUG_ASSERT(is_closed(cfg));
}

}
