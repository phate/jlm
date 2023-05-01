/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/cfg.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/domtree.hpp>

#include <unordered_map>

namespace jlm {

/* domnode class */

domnode *
domnode::add_child(std::unique_ptr<domnode> child)
{
	children_.push_back(std::move(child));
	auto c = children_.back().get();
	c->depth_ = depth()+1;
	c->parent_ = this;

	return c;
}

/* dominator computations */

static std::unique_ptr<domnode>
build_domtree(
	std::unordered_map<cfg_node*, cfg_node*> & doms,
	cfg_node * root)
{
	std::function<domnode*(cfg_node*, std::unordered_map<cfg_node*, domnode*>&)> build = [&](
		cfg_node * node,
		std::unordered_map<cfg_node*, domnode*> & map)
	{
		if (map.find(node) != map.end())
			return map[node];

		auto parent = build(doms[node], map);
		auto child = parent->add_child(domnode::create(node));
		map[node] = child;
		return child ;
	};

	auto & cfg = root->cfg();

	/* find leaves of tree */
	/* FIXME */
	std::unordered_set<cfg_node*> nodes({cfg.entry(), cfg.exit()});
	for (auto & node : cfg)
		nodes.insert(&node);
	for (auto & node : cfg)
		nodes.erase(doms[&node]);

	/* build tree bottom-up */
	std::unordered_map<cfg_node*, domnode*> map;
	auto domroot = domnode::create(root);
	map[root] = domroot.get();
	for (auto node : nodes)
		build(node, map);

	return domroot;
}

static cfg_node *
intersect(cfg_node * b1, cfg_node * b2,
	const std::unordered_map<cfg_node*, size_t> & indices,
	const std::unordered_map<cfg_node*, cfg_node*> & doms)
{
	while (indices.at(b1) != indices.at(b2)) {
		while (indices.at(b1) < indices.at(b2))
			b1 = doms.at(b1);
		while (indices.at(b2) < indices.at(b1))
			b2 = doms.at(b2);
	}

	return b1;
}

/*
	Keith D. Cooper et. al. - A Simple, Fast Dominance Algorithm
*/
std::unique_ptr<domnode>
domtree(jlm::cfg & cfg)
{
	JLM_ASSERT(is_closed(cfg));

	std::unordered_map<cfg_node*, cfg_node*> doms({
		{cfg.entry(), cfg.entry()}, {cfg.exit(), nullptr}
	});
	for (auto & node : cfg)
		doms[&node] = nullptr;

	size_t index = cfg.nnodes()+2;
	auto rporder = reverse_postorder(cfg);
	std::unordered_map<cfg_node*, size_t> indices;
	for(auto & node : rporder)
		indices[node] = index--;
	JLM_ASSERT(index == 0);

	bool changed = true;
	while (changed) {
		changed = false;
		for (auto & node : rporder) {
			if (node == cfg.entry())
				continue;

			/* find first processed predecessor */
			cfg_node * newidom = nullptr;
			for (auto & inedge : node->inedges()) {
				auto p = inedge->source();
				if (doms[p] != nullptr) {
					newidom = p;
					break;
				}
			}
			JLM_ASSERT(newidom != nullptr);

			auto pred = newidom;
			for (auto & inedge : node->inedges()) {
				auto p = inedge->source();
				if (p == pred)
					continue;

				if (doms[p] != nullptr)
					newidom = intersect(p, newidom, indices, doms);
			}

			if (doms[node] != newidom) {
				doms[node] = newidom;
				changed = true;
			}
		}
	}

	doms[cfg.entry()] = nullptr;
	return build_domtree(doms, cfg.entry());
}

}
