/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/types.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/alias-analyses/pointsto-graph.hpp>
#include <jlm/util/strfmt.hpp>

#include <jive/arch/addresstype.hpp>
#include <jive/rvsdg/node.hpp>
#include <jive/rvsdg/region.hpp>
#include <jive/rvsdg/structural-node.hpp>
#include <jive/rvsdg/traverser.hpp>

#include <typeindex>
#include <unordered_map>

namespace jlm {
namespace aa {

/* points-to graph */

ptg::ptg()
{
	memunknown_ = std::unique_ptr<jlm::aa::ptg::unknown>(new ptg::unknown(this));
}

ptg::allocnode_range
ptg::allocnodes()
{
	return allocnode_range(allocnodes_.begin(), allocnodes_.end());
}

ptg::allocnode_constrange
ptg::allocnodes() const
{
	return allocnode_constrange(allocnodes_.begin(), allocnodes_.end());
}

ptg::impnode_range
ptg::impnodes()
{
	return impnode_range(impnodes_.begin(), impnodes_.end());
}

ptg::impnode_constrange
ptg::impnodes() const
{
	return impnode_constrange(impnodes_.begin(), impnodes_.end());
}

ptg::regnode_range
ptg::regnodes()
{
	return regnode_range(regnodes_.begin(), regnodes_.end());
}

ptg::regnode_constrange
ptg::regnodes() const
{
	return regnode_constrange(regnodes_.begin(), regnodes_.end());
}

ptg::iterator
ptg::begin()
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return iterator(anodes.begin(), inodes.begin(), rnodes.begin(), anodes, inodes, rnodes);
}

ptg::constiterator
ptg::begin() const
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return constiterator(anodes.begin(), inodes.begin(), rnodes.begin(), anodes, inodes, rnodes);
}

ptg::iterator
ptg::end()
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return iterator(anodes.end(), inodes.end(), rnodes.end(), anodes, inodes, rnodes);
}

ptg::constiterator
ptg::end() const
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return constiterator(anodes.end(), inodes.end(), rnodes.end(), anodes, inodes, rnodes);
}

ptg::node *
ptg::add(std::unique_ptr<ptg::allocator> node)
{
	auto tmp = node.get();
	allocnodes_[node->node()] = std::move(node);

	return tmp;
}

ptg::node *
ptg::add(std::unique_ptr<ptg::regnode> node)
{
	auto tmp = node.get();
	regnodes_[node->output()] = std::move(node);

	return tmp;
}

ptg::node *
ptg::add(std::unique_ptr<ptg::impnode> node)
{
	auto tmp = node.get();
	impnodes_[node->argument()] = std::move(node);

	return tmp;
}

std::string
ptg::to_dot(const jlm::aa::ptg & ptg)
{
	auto shape = [](const ptg::node & node) {
		static std::unordered_map<std::type_index, std::string> shapes({
		  {typeid(allocator), "box"}
	  , {typeid(impnode),   "box"}
		, {typeid(regnode),   "oval"}
		, {typeid(unknown),   "box"}
		});

		if (shapes.find(typeid(node)) != shapes.end())
			return shapes[typeid(node)];

		JLM_UNREACHABLE("Unknown points-to graph node type.");
	};

	auto nodestring = [&](const ptg::node & node) {
		return strfmt("{ ", (intptr_t)&node, " ["
			, "label = \"", node.debug_string(), "\" "
			, "shape = \"", shape(node), "\"]; }\n");
	};

	auto edgestring = [](const ptg::node & node, const ptg::node & target)
	{
		return strfmt((intptr_t)&node, " -> ", (intptr_t)&target, "\n");
	};

	std::string dot("digraph ptg {\n");
	for (auto & node : ptg) {
		dot += nodestring(node);
		for (auto & target : node.targets())
			dot += edgestring(node, target);
	}
	dot += nodestring(ptg.memunknown());
	dot += "}\n";

	return dot;
}

/* points-to graph node */

ptg::node::~node()
{}

ptg::node::node_range
ptg::node::targets()
{
	return node_range(targets_.begin(), targets_.end());
}

ptg::node::node_constrange
ptg::node::targets() const
{
	return node_constrange(targets_.begin(), targets_.end());
}

ptg::node::node_range
ptg::node::sources()
{
	return node_range(sources_.begin(), sources_.end());
}

ptg::node::node_constrange
ptg::node::sources() const
{
	return node_constrange(sources_.begin(), sources_.end());
}

void
ptg::node::add_edge(ptg::node * target)
{
	if (Graph() != target->Graph())
		throw jlm::error("Points-to graph nodes are not in the same graph.");

	targets_.insert(target);
	target->sources_.insert(this);
}

void
ptg::node::remove_edge(ptg::node * target)
{
	if (Graph() != target->Graph())
		throw jlm::error("Points-to graph nodes are not in the same graph.");

	target->sources_.erase(this);
	targets_.erase(target);
}

/* points-to graph register node */

ptg::regnode::~regnode()
{}

std::string
ptg::regnode::debug_string() const
{
	auto node = jive::node_output::node(output());

	if (node != nullptr)
		return strfmt(node->operation().debug_string(), ":o", output()->index());

	node = output()->region()->node();
	if (node != nullptr)
		return strfmt(node->operation().debug_string(), ":a", output()->index());

	if (is_import(output())) {
		auto port = static_cast<const jlm::impport*>(&output()->port());
		return strfmt("import:", port->name());
	}

	return "REGNODE";
}

std::vector<const ptg::memnode*>
ptg::regnode::allocators(const ptg::regnode & node)
{
	/*
		FIXME: This function currently iterates through all pointstos of the regnode.
		Maybe we can be more efficient?
	*/
	std::vector<const ptg::memnode*> memnodes;
	for (auto & target : node.targets()) {
		if (auto memnode = dynamic_cast<const ptg::memnode*>(&target))
			memnodes.push_back(memnode);
	}

	return memnodes;
}

/* points-to graph memory node */

ptg::memnode::~memnode()
{}

/* points-to graph allocator node */

ptg::allocator::~allocator()
{}

std::string
ptg::allocator::debug_string() const
{
	return node()->operation().debug_string();
}

/* points-to graph import node */

ptg::impnode::~impnode()
{}

std::string
ptg::impnode::debug_string() const
{
	auto port = static_cast<const jlm::impport*>(&argument()->port());
	return port->name();
}

/* points-to graph unknown node */

ptg::unknown::~unknown()
{}

std::string
ptg::unknown::debug_string() const
{
	return "Unknown";
}

}}
