/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/aggregation/annotation.hpp>
#include <jlm/IR/aggregation/node.hpp>
#include <jlm/IR/basic_block.hpp>

#include <algorithm>

namespace jlm {
namespace agg {

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
