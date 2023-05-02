/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/aggregation.hpp>
#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/tac.hpp>

#include <algorithm>
#include <deque>
#include <unordered_map>

namespace jlm {

/* aggnode class */

aggnode::~aggnode()
{}

void
aggnode::normalize(aggnode & node)
{
	std::function<std::vector<std::unique_ptr<aggnode>>(aggnode&)> reduce = [&](aggnode & node)
	{
		JLM_ASSERT(is<linearaggnode>(&node));

		std::vector<std::unique_ptr<aggnode>> children;
		for (size_t n = 0; n < node.children_.size(); n++) {
			auto & child = node.children_[n];

			if (is<linearaggnode>(child.get())) {
				auto tmp = reduce(*child);
				std::move(tmp.begin(), tmp.end(), std::back_inserter(children));
			} else {
				children.push_back(std::move(child));
			}
		}

		return children;
	};

	if (is<linearaggnode>(&node)) {
		auto children = reduce(node);

		node.remove_children();
		for (auto & child : children)
			node.add_child(std::move(child));
	}

	for (auto & child : node)
		normalize(child);
}

/* entryaggnode class */

entryaggnode::~entryaggnode()
{}

entryaggnode::constiterator
entryaggnode::begin() const
{
	return constiterator(arguments_.begin());
}

entryaggnode::constiterator
entryaggnode::end() const
{
	return constiterator(arguments_.end());
}

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

/** Aggregation map
*
* It associates CFG nodes with aggregation subtrees.
*/
class aggregation_map final {
public:
	bool
	contains(jlm::cfg_node * node) const
	{
		return map_.find(node) != map_.end();
	}

	std::unique_ptr<aggnode>&
	lookup(cfg_node * node)
	{
		JLM_ASSERT(contains(node));

		return map_.at(node);
	}

	void
	insert(cfg_node * node, std::unique_ptr<aggnode> anode)
	{
		map_[node] = std::move(anode);
	}

	void
	remove(cfg_node * node)
	{
		map_.erase(node);
	}

	static std::unique_ptr<aggregation_map>
	create(jlm::cfg & cfg)
	{
		auto exit = cfg.exit();
		auto entry = cfg.entry();
		auto map = std::make_unique<aggregation_map>();

		map->map_[entry] = entryaggnode::create(entry->arguments());
		map->map_[exit] = exitaggnode::create(exit->results());
		for (auto & node : cfg) {
			auto bb = static_cast<basic_block*>(&node);
			map->map_[&node] = blockaggnode::create(std::move(bb->tacs()));
		}

		return map;
	}

private:
	std::unordered_map<cfg_node*, std::unique_ptr<aggnode>> map_;
};

static bool
is_sese_basic_block(const cfg_node * node) noexcept
{
	return node->ninedges() == 1
	    && node->noutedges() == 1;
}

static bool
is_branch_split(const cfg_node * node) noexcept
{
	return node->noutedges() > 1;
}

static bool
is_branch_join(const cfg_node * node) noexcept
{
	return node->ninedges() > 1;
}

static bool
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

static bool
is_linear(const cfg_node * node) noexcept
{
	if (node->noutedges() != 1)
		return false;

	auto exit = node->outedge(0)->sink();
	if (exit->ninedges() != 1)
		return false;

	return true;
}

static cfg_node *
aggregate(cfg_node*, cfg_node*, aggregation_map&);

/**
* Reduces a tail-controlled loop subgraph to a single node and creates an aggregation subtree for
* the subgraph.
*/
static void
reduce_loop(
	const sccstructure & sccstruct,
	aggregation_map & map)
{
	JLM_ASSERT(sccstruct.is_tcloop());

	auto exit = (*sccstruct.xedges().begin())->source();
	auto entry = *sccstruct.enodes().begin();

	auto redge = *sccstruct.redges().begin();
	redge->source()->remove_outedge(redge->index());

	auto sese = aggregate(entry, exit, map);
	auto loop = loopaggnode::create(std::move(map.lookup(sese)));
	map.insert(sese, std::move(loop));
}

/**
* Reduces a branch subgraph to a single node and creates an aggregation subtree for the subgraph.
* A branch subgraph of the following form:
* \dot
* 	digraph {
* 		Split -> A;
* 		Split -> B;
* 		Split -> C;
* 		A -> Join;
* 		B -> Join;
* 		C -> Join;
* 	}
* \enddot
*
* is reduced to the following aggregation subtree:
* \dot
* 	digraph {
* 		Split -> Linear;
* 		A -> Branch;
* 		B -> Branch;
* 		C -> Branch;
* 		Branch -> Linear;
* 	}
* \enddot
*
* Only the split node and the individual branch nodes are reduced. The join node is not reduced.
*/
static cfg_node *
reduce_branch(
	cfg_node * split,
	cfg_node ** entry,
	cfg_node ** exit,
	aggregation_map & map)
{
	/* sanity checks */
	JLM_ASSERT(split->noutedges() > 1);
	JLM_ASSERT(split->outedge(0)->sink()->noutedges() == 1);
	JLM_ASSERT(map.contains(split));

	auto join = split->outedge(0)->sink()->outedge(0)->sink();
	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++) {
		JLM_ASSERT(it->sink()->ninedges() == 1);
		JLM_ASSERT(map.contains(it->sink()));
		JLM_ASSERT(it->sink()->noutedges() == 1);
		JLM_ASSERT(it->sink()->outedge(0)->sink() == join);
	}

	/* perform reduction */
	auto sese = basic_block::create(split->cfg());
	split->divert_inedges(sese);
	sese->add_outedge(join);

	auto branch = branchaggnode::create();
	for (auto it = split->begin_outedges(); it != split->end_outedges(); it++) {
		it->sink()->remove_outedge(0);
		branch->add_child(std::move(map.lookup(it->sink())));
		map.remove(it->sink());
	}

	auto & child = map.lookup(split);
	map.insert(sese, linearaggnode::create(std::move(child), std::move(branch)));
	map.remove(split);

	/*
		We need to adjust the SEE subgraphs' entry, in case it was the same as the split and we just
		reduced it. We do not have to adjust the SESE subgraphs' exit node, as the branch join is not
		reduced in this function.
	*/
	*entry = split == *entry ? sese : *entry;
	return sese;
}

/**
* Reduce a linear subgraph to a single node and create an aggregation subtree for the subgraph.
* A linear subgraph of the following form:
* \dot
* 	digraph {
* 		A -> B;
* 	}
* \enddot
*
* is reduced to the following aggregation subtree:
* \dot
* 	digraph {
* 		A -> Linear;
* 		B -> Linear;
* 	}
* \enddot
*/
static cfg_node *
reduce_linear(
	cfg_node * source,
	cfg_node ** entry,
	cfg_node ** exit,
	aggregation_map & map)
{
	JLM_ASSERT(is_linear(source));
	auto sink = source->outedge(0)->sink();

	auto sese = basic_block::create(source->cfg());
	source->divert_inedges(sese);
	for (auto it = sink->begin_outedges(); it != sink->end_outedges(); it++)
		sese->add_outedge(it->sink());
	sink->remove_outedges();

	auto child0 = std::move(map.lookup(source));
	auto child1 = std::move(map.lookup(sink));
	map.insert(sese, linearaggnode::create(std::move(child0), std::move(child1)));
	map.remove(source);
	map.remove(sink);

	/*
		We need to adjust the SESE subgraphs' entry and exit, in case we have just reduced either of
		them.
	*/
	*entry = source == *entry ? sese : *entry;
	*exit = sink == *exit ? sese : *exit; 

	return sese;
}

/**
* Find all tail-controlled loops in an SESE subgraph and reduce each loop to a single node.
*/
static void
aggregate_loops(
	cfg_node * entry,
	cfg_node * exit,
	aggregation_map & map)
{
	auto sccs = find_sccs(entry, exit);
	for (auto scc : sccs) {
		auto sccstruct = sccstructure::create(scc);

		if (sccstruct->is_tcloop()) {
			reduce_loop(*sccstruct, map);
			continue;
		}

		JLM_UNREACHABLE("We should have never reached this point!");
	}
}

static void
aggregate_acyclic_sese(
	cfg_node * node,
	cfg_node ** entry,
	cfg_node ** exit,
	aggregation_map & map)
{
	/*
		We reduced the entire subgraph to a single node. We are done here.
	*/
	if (*entry == *exit) {
		JLM_ASSERT(node == *entry);
		return;
	}

	/*
		We traversed the entire subgraph until the end. Turn around.
	*/
	if (node == *exit)
		return;

	/*
		Reduce linear subgraph
	*/
	if (is_linear(node)) {
		auto sese = reduce_linear(node, entry, exit, map);
		aggregate_acyclic_sese(sese, entry, exit, map);
		return;
	}

	/*
		Reduce branch subgraph
	*/
	if (is_branch_split(node)) {
		/*
			First, greedily reduce all branches of the branch subgraph...
		*/
		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++)
			aggregate_acyclic_sese(it->sink(), entry, exit, map);

		/*
			..., then try to reduce the branch subgraph itself.
		*/
		if (is_branch(node)) {
			auto sese = reduce_branch(node, entry, exit, map);
			aggregate_acyclic_sese(sese, entry, exit, map);
			return;
		}

		JLM_UNREACHABLE("We should have never reached this point!");
	}

	/*
		It is only a single basic block with one incoming and outgoing edge, simply step over it.
	*/
	if (is_sese_basic_block(node)) {
		aggregate_acyclic_sese(node->outedge(0)->sink(), entry, exit, map);
		return;
	}

	/*
		It is a branch join, turn around to the branch split such that we can reduce it.
	*/
	if (is_branch_join(node))
		return;

	JLM_UNREACHABLE("We should have never reached this point!");
}

/**
* This function takes the \p entry and the \p exit of a single-entry/single-exit (SESE) subgraph,
* i.e. \p entry must always dominate \p exit, and reduces the subgraph to an aggregation subtree.
* The subgraph is then replaced by a single basic block in the CFG and this CFG is associated with
* the aggregation subtree in the aggregation map.
*
* The function consists of two recursively nested phases:
* 1. Loop aggregation
* 2. Acyclic SESE aggregation.
*
* The first phase finds all tail-controlled loops and recursively invokes the aggregation procedure
* on a loops' body to reduce it to a single node. Once all loops in the subgraph have been
* reduced, the Acyclic SESE aggregation reduces the rest of the acyclic graph into a tree.
*/
static cfg_node *
aggregate(
	cfg_node * entry,
	cfg_node * exit,
	aggregation_map & map)
{
	aggregate_loops(entry, exit, map);
	aggregate_acyclic_sese(entry, &entry, &exit, map);
	JLM_ASSERT(entry == exit);

	return entry;
}

std::unique_ptr<aggnode>
aggregate(jlm::cfg & cfg)
{
	JLM_ASSERT(is_proper_structured(cfg));

	auto map = aggregation_map::create(cfg);
	auto root = aggregate(cfg.entry(), cfg.exit(), *map);

	return std::move(map->lookup(root));
}

size_t
ntacs(const jlm::aggnode & root)
{
	size_t n = 0;
	for (auto & child : root)
		n += ntacs(child);

	if (auto bb = dynamic_cast<const blockaggnode*>(&root))
		n += bb->tacs().ntacs();

	return n;
}

}
