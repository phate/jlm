/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/strfmt.hpp>

#include <jive/arch/addresstype.hpp>
#include <jive/rvsdg/node.hpp>
#include <jive/rvsdg/structural-node.hpp>

#include <typeindex>
#include <unordered_map>

namespace jlm {
namespace aa {

/* points-to graph */

PointsToGraph::PointsToGraph()
{
	memunknown_ = std::unique_ptr<PointsToGraph::UnknownNode>(new PointsToGraph::UnknownNode(*this));
  ExternalMemory_ = ExternalMemoryNode::Create(*this);
}

PointsToGraph::allocnode_range
PointsToGraph::allocnodes()
{
	return {allocnodes_.begin(), allocnodes_.end()};
}

PointsToGraph::allocnode_constrange
PointsToGraph::allocnodes() const
{
	return {allocnodes_.begin(), allocnodes_.end()};
}

PointsToGraph::impnode_range
PointsToGraph::impnodes()
{
	return {impnodes_.begin(), impnodes_.end()};
}

PointsToGraph::impnode_constrange
PointsToGraph::impnodes() const
{
	return {impnodes_.begin(), impnodes_.end()};
}

PointsToGraph::regnode_range
PointsToGraph::regnodes()
{
	return {regnodes_.begin(), regnodes_.end()};
}

PointsToGraph::regnode_constrange
PointsToGraph::regnodes() const
{
	return {regnodes_.begin(), regnodes_.end()};
}

PointsToGraph::iterator
PointsToGraph::begin()
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return {anodes.begin(), inodes.begin(), rnodes.begin(), anodes, inodes, rnodes};
}

PointsToGraph::constiterator
PointsToGraph::begin() const
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return {anodes.begin(), inodes.begin(), rnodes.begin(), anodes, inodes, rnodes};
}

PointsToGraph::iterator
PointsToGraph::end()
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return {anodes.end(), inodes.end(), rnodes.end(), anodes, inodes, rnodes};
}

PointsToGraph::constiterator
PointsToGraph::end() const
{
	auto anodes = allocnodes();
	auto inodes = impnodes();
	auto rnodes = regnodes();
	return {anodes.end(), inodes.end(), rnodes.end(), anodes, inodes, rnodes};
}

PointsToGraph::AllocatorNode &
PointsToGraph::add(std::unique_ptr<PointsToGraph::AllocatorNode> node)
{
	auto tmp = node.get();
	allocnodes_[node->node()] = std::move(node);

	return *tmp;
}

PointsToGraph::RegisterNode &
PointsToGraph::add(std::unique_ptr<PointsToGraph::RegisterNode> node)
{
  auto tmp = node.get();
	regnodes_[node->output()] = std::move(node);

	return *tmp;
}

PointsToGraph::ImportNode &
PointsToGraph::add(std::unique_ptr<PointsToGraph::ImportNode> node)
{
	auto tmp = node.get();
	impnodes_[node->argument()] = std::move(node);

	return *tmp;
}

std::string
PointsToGraph::ToDot(const PointsToGraph & ptg)
{
	auto shape = [](const PointsToGraph::Node & node) {
		static std::unordered_map<std::type_index, std::string> shapes({
		  {typeid(AllocatorNode), "box"}
	  , {typeid(ImportNode),    "box"}
		, {typeid(RegisterNode),  "oval"}
		, {typeid(UnknownNode),   "box"}
    , {typeid(ExternalMemoryNode), "box"}
		});

		if (shapes.find(typeid(node)) != shapes.end())
			return shapes[typeid(node)];

		JLM_UNREACHABLE("Unknown points-to graph Node type.");
	};

	auto nodestring = [&](const PointsToGraph::Node & node) {
		return strfmt("{ ", (intptr_t)&node, " ["
			, "label = \"", node.debug_string(), "\" "
			, "shape = \"", shape(node), "\"]; }\n");
	};

	auto edgestring = [](const PointsToGraph::Node & node, const PointsToGraph::Node & target)
	{
		return strfmt((intptr_t)&node, " -> ", (intptr_t)&target, "\n");
	};

	std::string dot("digraph PointsToGraph {\n");
	for (auto & node : ptg) {
		dot += nodestring(node);
		for (auto & target : node.targets())
			dot += edgestring(node, target);
	}
	dot += nodestring(ptg.memunknown());
  dot += nodestring(ptg.GetExternalMemoryNode());
	dot += "}\n";

	return dot;
}

/* PointsToGraph::Node */

PointsToGraph::Node::~Node() = default;

PointsToGraph::Node::node_range
PointsToGraph::Node::targets()
{
	return {targets_.begin(), targets_.end()};
}

PointsToGraph::Node::node_constrange
PointsToGraph::Node::targets() const
{
	return {targets_.begin(), targets_.end()};
}

PointsToGraph::Node::node_range
PointsToGraph::Node::sources()
{
	return {sources_.begin(), sources_.end()};
}

PointsToGraph::Node::node_constrange
PointsToGraph::Node::sources() const
{
	return {sources_.begin(), sources_.end()};
}

void
PointsToGraph::Node::add_edge(PointsToGraph::MemoryNode & target)
{
	if (&Graph() != &target.Graph())
		throw jlm::error("Points-to graph nodes are not in the same graph.");

	targets_.insert(&target);
	target.sources_.insert(this);
}

void
PointsToGraph::Node::remove_edge(PointsToGraph::MemoryNode & target)
{
	if (&Graph() != &target.Graph())
		throw jlm::error("Points-to graph nodes are not in the same graph.");

	target.sources_.erase(this);
	targets_.erase(&target);
}

/* PointsToGraph::RegisterNode */

PointsToGraph::RegisterNode::~RegisterNode() = default;

std::string
PointsToGraph::RegisterNode::debug_string() const
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

std::vector<const PointsToGraph::MemoryNode*>
PointsToGraph::RegisterNode::allocators(const PointsToGraph::RegisterNode & node)
{
	/*
		FIXME: This function currently iterates through all pointstos of the RegisterNode.
		Maybe we can be more efficient?
	*/
	std::vector<const PointsToGraph::MemoryNode*> memnodes;
	for (auto & target : node.targets()) {
		if (auto memnode = dynamic_cast<const PointsToGraph::MemoryNode*>(&target))
			memnodes.push_back(memnode);
	}

	return memnodes;
}

/* PointsToGraph::MemoryNode class */

PointsToGraph::MemoryNode::~MemoryNode() = default;

/* PointsToGraph::AllocatorNode class */

PointsToGraph::AllocatorNode::~AllocatorNode() = default;

std::string
PointsToGraph::AllocatorNode::debug_string() const
{
	return node()->operation().debug_string();
}

/* PointsToGraph::ImportNode class */

PointsToGraph::ImportNode::~ImportNode() = default;

std::string
PointsToGraph::ImportNode::debug_string() const
{
	auto port = static_cast<const jlm::impport*>(&argument()->port());
	return port->name();
}

/* PointsToGraph::UnknownNode class */

PointsToGraph::UnknownNode::~UnknownNode() = default;

std::string
PointsToGraph::UnknownNode::debug_string() const
{
	return "Unknown";
}

/**
 * PointsToGraph::ExternalMemoryNode class
 */
PointsToGraph::ExternalMemoryNode::~ExternalMemoryNode()
= default;

std::string
PointsToGraph::ExternalMemoryNode::debug_string() const
{
  return "ExternalMemory";
}

}}
