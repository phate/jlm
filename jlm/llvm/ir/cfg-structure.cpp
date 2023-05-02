/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <algorithm>
#include <unordered_map>

namespace jlm {

/* scc class */

scc::constiterator
scc::begin() const
{
	return constiterator(nodes_.begin());
}

scc::constiterator
scc::end() const
{
	return constiterator(nodes_.end());
}

/* sccstructure class */

bool
sccstructure::is_tcloop() const
{
	return nenodes() == 1
	    && nredges() == 1
	    && nxedges() == 1
	    && (*redges().begin())->source() == (*xedges().begin())->source();
}

std::unique_ptr<sccstructure>
sccstructure::create(const jlm::scc & scc)
{
	auto sccstruct = std::make_unique<sccstructure>();

	for (auto & node : scc) {
		for (auto & inedge : node.inedges()) {
			if (!scc.contains(inedge->source())) {
				sccstruct->eedges_.insert(inedge);
				if (sccstruct->enodes_.find(&node) == sccstruct->enodes_.end())
					sccstruct->enodes_.insert(&node);
			}
		}

		for (auto it = node.begin_outedges(); it != node.end_outedges(); it++) {
			if (!scc.contains(it->sink())) {
				sccstruct->xedges_.insert(it.edge());
				if (sccstruct->xnodes_.find(it->sink()) == sccstruct->xnodes_.end())
					sccstruct->xnodes_.insert(it->sink());
			}
		}
	}

	for (auto & node : scc) {
		for (auto it = node.begin_outedges(); it != node.end_outedges(); it++) {
			if (sccstruct->enodes_.find(it->sink()) != sccstruct->enodes_.end())
				sccstruct->redges_.insert(it.edge());
		}
	}

	return sccstruct;
}

/**
* Tarjan's SCC algorithm
*/
static void
strongconnect(
	jlm::cfg_node * node,
	jlm::cfg_node * exit,
	std::unordered_map<jlm::cfg_node*, std::pair<size_t,size_t>> & map,
	std::vector<jlm::cfg_node*> & node_stack,
	size_t & index,
	std::vector<jlm::scc> & sccs)
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
		std::unordered_set<jlm::cfg_node*> set;
		jlm::cfg_node * w;
		do {
			w = node_stack.back();
			node_stack.pop_back();
			set.insert(w);
		} while (w != node);

		if (set.size() != 1 || (*set.begin())->has_selfloop_edge())
			sccs.push_back(jlm::scc(set));
	}
}

std::vector<jlm::scc>
find_sccs(const jlm::cfg & cfg)
{
	JLM_ASSERT(is_closed(cfg));

	return find_sccs(cfg.entry(), cfg.exit());
}

std::vector<jlm::scc>
find_sccs(cfg_node * entry, cfg_node * exit)
{
	size_t index = 0;
	std::vector<scc> sccs;
	std::vector<cfg_node*> node_stack;
	std::unordered_map<cfg_node*, std::pair<size_t,size_t>> map;
	strongconnect(entry, exit, map, node_stack, index, sccs);

	return sccs;
}

}

static inline std::unique_ptr<jlm::cfg>
copy_structural(const jlm::cfg & in)
{
	JLM_ASSERT(is_valid(in));

	std::unique_ptr<jlm::cfg> out(new jlm::cfg(in.module()));
	out->entry()->remove_outedge(0);

	/* create all nodes */
	std::unordered_map<const jlm::cfg_node*, jlm::cfg_node*> node_map({
	  {in.entry(), out->entry()}, {in.exit(), out->exit()}
	});

	for (const auto & node : in) {
		JLM_ASSERT(jlm::is<jlm::basic_block>(&node));
		node_map[&node] = jlm::basic_block::create(*out);
	}

	/* establish control flow */
	node_map[in.entry()]->add_outedge(node_map[in.entry()->outedge(0)->sink()]);
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
	JLM_ASSERT(split->noutedges() > 1);
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

	auto source = (*node->inedges().begin())->source();
	for (auto & inedge : node->inedges()) {
		if (inedge->source() != source)
			return false;
	}

	return true;
}

static inline void
reduce_loop(
	jlm::cfg_node * node,
	std::unordered_set<jlm::cfg_node*> & to_visit)
{
	JLM_ASSERT(is_loop(node));
	auto & cfg = node->cfg();

	auto reduction = jlm::basic_block::create(cfg);
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
	JLM_ASSERT(is_linear_reduction(entry));
	auto exit = entry->outedge(0)->sink();
	auto & cfg = entry->cfg();

	auto reduction = jlm::basic_block::create(cfg);
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
	JLM_ASSERT(is_branch(split));
	auto join = find_join(split);
	auto & cfg = split->cfg();

	auto reduction = jlm::basic_block::create(cfg);
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
	JLM_ASSERT(is_proper_branch(split));
	auto join = split->outedge(0)->sink()->outedge(0)->sink();

	auto reduction = jlm::basic_block::create(split->cfg());
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
	JLM_ASSERT(is_T1(node));

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
	JLM_ASSERT(is_T2(node));

	auto p = (*node->inedges().begin())->source();
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
has_valid_phis(const basic_block & bb)
{
	for (auto it = bb.begin(); it != bb.end(); it++) {
		auto tac = *it;
		if (!is<phi_op>(tac))
			continue;

		/*
			Ensure the number of phi operands equals the number of incoming edges
		*/
		if (tac->noperands() != bb.ninedges())
			return false;

		/*
			Ensure all phi nodes are at the beginning of a basic block
		*/
		if (tac != bb.first() && !is<phi_op>(*std::prev(it)))
			return false;

		/*
			Ensure that a phi node does not have for the same basic block
			multiple incoming variables.
		*/
		auto phi = static_cast<const phi_op*>(&tac->operation());
		std::unordered_map<cfg_node*, const variable*> map;
		for (size_t n = 0; n < tac->noperands(); n++) {
			auto mit = map.find(phi->node(n));
			if (mit != map.end() && mit->second != tac->operand(n))
				return false;

			if (mit == map.end())
				map[phi->node(n)] = tac->operand(n);
		}
	}

	return true;
}

static bool
is_valid_basic_block(const basic_block & bb)
{
	if (bb.no_successor())
		return false;

	if (!has_valid_phis(bb))
		return false;

	return true;
}

static bool
has_valid_entry(const jlm::cfg & cfg)
{
	if (!cfg.entry()->no_predecessor())
		return false;

	if (cfg.entry()->noutedges() != 1)
		return false;

	return true;
}

static bool
has_valid_exit(const jlm::cfg & cfg)
{
	return cfg.exit()->no_successor();
}

bool
is_valid(const jlm::cfg & cfg)
{
	if (!has_valid_entry(cfg))
		return false;

	if (!has_valid_exit(cfg))
		return false;

	/* check basic blocks */
	for (const auto & node : cfg) {
		JLM_ASSERT(is<basic_block>(&node));
		auto & bb = *static_cast<const basic_block*>(&node);
		if (!is_valid_basic_block(bb))
			return false;
	}

	return true;
}

bool
is_closed(const jlm::cfg & cfg)
{
	JLM_ASSERT(is_valid(cfg));

	for (const auto & node : cfg) {
		if (node.no_predecessor())
			return false;
	}

	return true;
}

bool
is_linear(const jlm::cfg & cfg)
{
	JLM_ASSERT(is_closed(cfg));

	for (const auto & node : cfg) {
		if (!node.single_successor() || !node.single_predecessor())
			return false;
	}

	return true;
}

static inline bool
reduce(
	const jlm::cfg & cfg,
	const std::function<bool(jlm::cfg_node*, std::unordered_set<jlm::cfg_node*>&)> & f)
{
	JLM_ASSERT(is_closed(cfg));
	auto c = copy_structural(cfg);

	std::unordered_set<cfg_node*> to_visit({c->entry(), c->exit()});
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
		&& is<basic_block>(it.node())
		&& is<basic_block>(it->outedge(0)->sink())) {
			static_cast<basic_block*>(it->outedge(0)->sink())->append_first(it.node()->tacs());
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
	JLM_ASSERT(is_valid(cfg));

	auto it = cfg.begin();
	while (it != cfg.end()) {
		auto bb = it.node();

		/*
			Ignore basic blocks with instructions
		*/
		if (bb->ntacs() != 0) {
			it++;
			continue;
		}

		JLM_ASSERT(bb->noutedges() == 1);
		auto outedge = bb->outedge(0);
		/*
			Ignore endless loops
		*/
		if (outedge->sink() == bb) {
			it++;
			continue;
		}

		bb->divert_inedges(outedge->sink());
		it = cfg.remove_node(it);
	}

	JLM_ASSERT(is_valid(cfg));
}

/*
* @brief Find all nodes dominated by the entry node.
*/
static std::unordered_set<const cfg_node*>
compute_livenodes(const jlm::cfg & cfg)
{
	std::unordered_set<const cfg_node*> visited;
	std::unordered_set<cfg_node*> to_visit({cfg.entry()});
	while (!to_visit.empty()) {
		auto node = *to_visit.begin();
		to_visit.erase(to_visit.begin());
		JLM_ASSERT(visited.find(node) == visited.end());
		visited.insert(node);
		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			if (visited.find(it->sink()) == visited.end()
			&& to_visit.find(it->sink()) == to_visit.end())
				to_visit.insert(it->sink());
		}
	}

	return visited;
}

/*
* @brief Find all nodes that are NOT dominated by the entry node.
*/
static std::unordered_set<cfg_node*>
compute_deadnodes(jlm::cfg & cfg)
{
	auto livenodes = compute_livenodes(cfg);

	std::unordered_set<cfg_node*> deadnodes;
	for (auto & node : cfg) {
		if (livenodes.find(&node) == livenodes.end())
			deadnodes.insert(&node);
	}

	JLM_ASSERT(deadnodes.find(cfg.entry()) == deadnodes.end());
	JLM_ASSERT(deadnodes.find(cfg.exit()) == deadnodes.end());
	return deadnodes;
}

/*
* @brief Returns all basic blocks that are live and a sink
*	of a dead node.
*/
static std::unordered_set<basic_block*>
compute_live_sinks(const std::unordered_set<cfg_node*> & deadnodes)
{
	std::unordered_set<basic_block*> sinks;
	for (auto & node : deadnodes) {
		for (size_t n = 0; n < node->noutedges(); n++) {
			auto sink = dynamic_cast<basic_block*>(node->outedge(n)->sink());
			if (sink && deadnodes.find(sink) == deadnodes.end())
				sinks.insert(sink);
		}
	}

	return sinks;
}

static void
update_phi_operands(
	jlm::tac & phitac,
	const std::unordered_set<cfg_node*> & deadnodes)
{
	JLM_ASSERT(is<phi_op>(&phitac));
	auto phi = static_cast<const phi_op*>(&phitac.operation());

	std::vector<cfg_node*> nodes;
	std::vector<const variable*> operands;
	for (size_t n = 0; n < phitac.noperands(); n++) {
		if (deadnodes.find(phi->node(n)) == deadnodes.end()) {
			operands.push_back(phitac.operand(n));
			nodes.push_back(phi->node(n));
		}
	}

	phitac.replace(phi_op(nodes, phi->type()), operands);
}

static void
update_phi_operands(
	const std::unordered_set<basic_block*> & sinks,
	const std::unordered_set<cfg_node*> & deadnodes)
{
	for (auto & sink : sinks) {
		for (auto & tac : *sink) {
			if (!is<phi_op>(tac))
				break;

			update_phi_operands(*tac, deadnodes);
		}
	}
}

static void
remove_deadnodes(const std::unordered_set<cfg_node*> & deadnodes)
{
	for (auto & node : deadnodes) {
		node->remove_inedges();
		JLM_ASSERT(is<basic_block>(node));
		node->cfg().remove_node(static_cast<basic_block*>(node));
	}
}

void
prune(jlm::cfg & cfg)
{
	JLM_ASSERT(is_valid(cfg));

	auto deadnodes = compute_deadnodes(cfg);
	auto sinks = compute_live_sinks(deadnodes);
	update_phi_operands(sinks, deadnodes);
	remove_deadnodes(deadnodes);

	JLM_ASSERT(is_closed(cfg));
}

}
