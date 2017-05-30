/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/aggregation/aggregation.hpp>
#include <jlm/IR/aggregation/node.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/tac.hpp>

#include <algorithm>

namespace jlm {
namespace agg {

/* reduction */

class reduction final : public attribute {
public:
	virtual
	~reduction()
	{}

	inline constexpr
	reduction(const jlm::cfg_node * entry, const jlm::cfg_node * exit)
	: attribute()
	, entry_(entry)
	, exit_(exit)
	{}

	inline const jlm::cfg_node *
	entry() const noexcept
	{
		return entry_;
	}

	inline const jlm::cfg_node *
	exit() const noexcept
	{
		return exit_;
	}

	virtual std::string
	debug_string() const noexcept
	{
		return "reduction";
	}

	virtual std::unique_ptr<attribute>
	copy() const
	{
		return std::make_unique<reduction>(*this);
	}

private:
	const jlm::cfg_node * entry_;
	const jlm::cfg_node * exit_;
};

static inline cfg_node *
create_reduction_node(const jlm::cfg_node * entry, const jlm::cfg_node * exit)
{
	jlm::agg::reduction reduction(entry, exit);
	return entry->cfg()->create_node(reduction);
}

static inline bool
is_reduction_node(const jlm::cfg_node * node)
{
	return dynamic_cast<const reduction*>(&node->attribute()) != nullptr;
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
is_branch_join(const cfg_node * node) noexcept
{
	return node->ninedges() > 1;
}

static inline bool
is_branch_split(const cfg_node * node) noexcept
{
	return node->noutedges() > 1;
}

static inline bool
is_linear_entry(const cfg_node * node) noexcept
{
	return node->noutedges() == 1;
}

static inline bool
is_linear_exit(const cfg_node * node) noexcept
{
	return node->ninedges() == 1;
}

static inline jlm::cfg_node *
reduce_linear(
	jlm::cfg_node * entry,
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(entry->noutedges() == 1);
	JLM_DEBUG_ASSERT(map.find(entry) != map.end());

	auto exit = entry->outedge(0)->sink();
	JLM_DEBUG_ASSERT(exit->ninedges() == 1);
	JLM_DEBUG_ASSERT(map.find(exit) != map.end());

	/* perform reduction */
	auto reduction = create_reduction_node(entry, exit);
	entry->divert_inedges(reduction);
	for (const auto & outedge : exit->outedges())
		reduction->add_outedge(outedge->sink(), outedge->index());
	exit->remove_outedges();

	map[reduction] = create_linear_node(std::move(map[entry]), std::move(map[exit]));
	map.erase(entry);
	map.erase(exit);

	return reduction;
}

static inline jlm::cfg_node *
reduce_loop(
	jlm::cfg_node * node,
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(is_loop(node));
	JLM_DEBUG_ASSERT(map.find(node) != map.end());

	/* perform reduction */
	auto reduction = create_reduction_node(node, node);
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

	return reduction;
}

static inline jlm::cfg_node *
reduce_branch(
	jlm::cfg_node * split,
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
{
	/* sanity checks */
	JLM_DEBUG_ASSERT(is_basic_block(split));
	JLM_DEBUG_ASSERT(split->noutedges() > 1);
	JLM_DEBUG_ASSERT(split->outedge(0)->sink()->noutedges() == 1);
	JLM_DEBUG_ASSERT(map.find(split) != map.end());

	auto join = split->outedge(0)->sink()->outedge(0)->sink();
	JLM_DEBUG_ASSERT(is_basic_block(join));
	JLM_DEBUG_ASSERT(join->noutedges() == 1);
	JLM_DEBUG_ASSERT(map.find(join) != map.end());

	for (const auto & outedge : split->outedges()) {
		JLM_DEBUG_ASSERT(outedge->sink()->ninedges() == 1);
		JLM_DEBUG_ASSERT(map.find(outedge->sink()) != map.end());
		JLM_DEBUG_ASSERT(outedge->sink()->noutedges() == 1);
		JLM_DEBUG_ASSERT(outedge->sink()->outedge(0)->sink() == join);
	}

	/* perform reduction */
	auto reduction = create_reduction_node(split, join);
	split->divert_inedges(reduction);
	reduction->add_outedge(join->outedge(0)->sink(), 0);
	join->remove_outedges();

	auto s = static_cast<const basic_block*>(&split->attribute());
	auto j = static_cast<const basic_block*>(&join->attribute());
	auto branch = create_branch_node(*s, *j);
	for (const auto & outedge : split->outedges()) {
		branch->add_child(std::move(map[outedge->sink()]));
		map.erase(outedge->sink());
	}

	map[reduction] = std::move(branch);
	map.erase(split);
	map.erase(join);

	return reduction;
}

static inline void
aggregate(
	jlm::cfg_node * node,
	std::unordered_map<cfg_node*, std::unique_ptr<agg::node>> & map)
{
	if (is_linear_entry(node) && is_loop(node->outedge(0)->sink())) {
		aggregate(node->outedge(0)->sink(), map);
		JLM_DEBUG_ASSERT(is_reduction_node(node->outedge(0)->sink()));
		auto reduction = reduce_linear(node, map);
		aggregate(reduction, map);
		return;
	}

	if (is_linear_entry(node) && is_branch_split(node->outedge(0)->sink())) {
		aggregate(node->outedge(0)->sink(), map);
		JLM_DEBUG_ASSERT(is_reduction_node(node->outedge(0)->sink()));
		auto reduction = reduce_linear(node, map);
		aggregate(reduction, map);
		return;
	}

	if (is_linear_entry(node) && is_linear_exit(node->outedge(0)->sink())) {
		aggregate(node->outedge(0)->sink(), map);
		reduce_linear(node, map);
		return;
	}

	if (is_loop(node)) {
		auto reduction = reduce_loop(node, map);
		aggregate(reduction, map);
		return;
	}

	if (is_branch_split(node)) {
		for (size_t n = 0; n < node->noutedges(); n++)
			aggregate(node->outedge(n)->sink(), map);
		auto reduction = reduce_branch(node, map);
		aggregate(reduction, map);
		return;
	}

	if (is_branch_join(node))
		return;
}

std::unique_ptr<agg::node>
aggregate(jlm::cfg & cfg)
{
	/* insert all aggregation leaves into the map */
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
	}

	aggregate(cfg.entry_node(), map);
	JLM_DEBUG_ASSERT(map.size() == 1);
	return std::move(std::move(map.begin()->second));
}

/* annotate */

static demand_set
annotate(const agg::node * node, const demand_set & pds, demand_map & dm);

static inline demand_set
annotate_basic_block(const basic_block & bb, const demand_set & pds)
{
	demand_set ds(pds);
	for (auto it = bb.rbegin(); it != bb.rend(); it++) {
		for (size_t n = 0; n < (*it)->noutputs(); n++)
			ds.erase((*it)->output(n));
		for (size_t n = 0; n < (*it)->ninputs(); n++)
			ds.insert((*it)->input(n));
	}

	return ds;
}

static inline demand_set
annotate_entry(const agg::node * node, const demand_set & pds, demand_map & dm)
{
	JLM_DEBUG_ASSERT(is_entry_structure(node->structure()));
	const auto & ea = static_cast<const entry*>(&node->structure())->attribute();

	demand_set ds(pds);
	for (size_t n = 0; n < ea.narguments(); n++)
		ds.erase(ea.argument(n));

	JLM_DEBUG_ASSERT(dm.find(node) == dm.end());
	dm[node] = ds;

	return ds;
}

static inline demand_set
annotate_exit(const agg::node * node, const demand_set & pds, demand_map & dm)
{
	JLM_DEBUG_ASSERT(is_exit_structure(node->structure()));
	const auto & xa = static_cast<const exit*>(&node->structure())->attribute();

	demand_set ds(pds);
	for (size_t n = 0; n < xa.nresults(); n++)
		ds.insert(xa.result(n));

	JLM_DEBUG_ASSERT(dm.find(node) == dm.end());
	dm[node] = ds;

	return ds;
}

static inline demand_set
annotate_block(const agg::node * node, const demand_set & pds, demand_map & dm)
{
	JLM_DEBUG_ASSERT(is_block_structure(node->structure()));
	const auto & bb = static_cast<const block*>(&node->structure())->basic_block();

	auto ds = annotate_basic_block(bb, pds);
	JLM_DEBUG_ASSERT(dm.find(node) == dm.end());
	dm[node] = ds;

	return ds;
}

static inline demand_set
annotate_linear(const agg::node * node, const demand_set & pds, demand_map & dm)
{
	JLM_DEBUG_ASSERT(is_linear_structure(node->structure()));

	demand_set ds(pds);
	for (ssize_t n = node->nchildren()-1; n >= 0; n--)
		ds = annotate(node->child(n), ds, dm);

	JLM_DEBUG_ASSERT(dm.find(node) == dm.end());
	dm[node] = ds;

	return ds;
}

static inline demand_set
annotate_branch(const agg::node * node, const demand_set & pds, demand_map & dm)
{
	JLM_DEBUG_ASSERT(is_branch_structure(node->structure()));
	const auto & branch = static_cast<const jlm::agg::branch*>(&node->structure());

	demand_set intersect;
	auto ds = annotate_basic_block(branch->join(), pds);
	for (ssize_t n = node->nchildren()-1; n >= 0; n--) {
		auto tmp = annotate(node->child(n), ds, dm);
		auto tmp2 = std::move(intersect);
		std::set_intersection(tmp.begin(), tmp.end(), tmp2.begin(), tmp2.end(),
			std::inserter(intersect, intersect.begin()));
	}
	intersect= annotate_basic_block(branch->split(), intersect);

	JLM_DEBUG_ASSERT(dm.find(node) == dm.end());
	dm[node] = intersect;

	return intersect;
}

static inline demand_set
annotate_loop(const agg::node * node, const demand_set & pds, demand_map & dm)
{
	JLM_DEBUG_ASSERT(is_loop_structure(node->structure()));
	JLM_DEBUG_ASSERT(node->nchildren() == 1);

	auto ds = annotate(node->child(0), pds, dm);
	if (ds != pds) {
		auto tmp = annotate(node->child(0), ds, dm);
		JLM_DEBUG_ASSERT(tmp == ds);
	}

	JLM_DEBUG_ASSERT(dm.find(node) == dm.end());
	dm[node] = ds;

	return ds;
}

static inline demand_set
annotate(const agg::node * node, const demand_set & pds, demand_map & dm)
{
	static std::unordered_map<
		std::type_index,
		std::function<demand_set(const agg::node*, const demand_set&, demand_map&)>
	> map({
	  {std::type_index(typeid(entry)), annotate_entry}
	, {std::type_index(typeid(exit)), annotate_exit}
	, {std::type_index(typeid(block)), annotate_block}
	, {std::type_index(typeid(linear)), annotate_linear}
	, {std::type_index(typeid(branch)), annotate_branch}
	, {std::type_index(typeid(loop)), annotate_loop}
	});

	auto it = dm.find(node);
	if (it != dm.end() && it->second == pds)
		return pds;

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(node->structure()))) != map.end());
	return map[std::type_index(typeid(node->structure()))](node, pds, dm);
}

demand_map
annotate(jlm::agg::node & root)
{
	demand_map dm;
	annotate(&root, demand_set(), dm);
	return dm;
}

}}
