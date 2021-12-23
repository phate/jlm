/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP
#define JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP

#include <jlm/common.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/ir/rvsdg-module.hpp>

#include <jive/rvsdg/node.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace jive {
	class argument;
	class node;
	class output;
}

namespace jlm {

class rvsdg_module;

namespace aa {

/** /brief PointsTo Graph
*
*/
class PointsToGraph final {
	class constiterator;
	class iterator;

public:
	class AllocatorNode;
	class ImportNode;
	class MemoryNode;
	class Node;
	class RegisterNode;
	class UnknownNode;

	using allocnodemap = std::unordered_map<const jive::node*, std::unique_ptr<PointsToGraph::AllocatorNode>>;
	using impnodemap = std::unordered_map<const jive::argument*, std::unique_ptr<PointsToGraph::ImportNode>>;
	using regnodemap = std::unordered_map<const jive::output*, std::unique_ptr<PointsToGraph::RegisterNode>>;

	using allocnode_range = iterator_range<allocnodemap::iterator>;
	using allocnode_constrange = iterator_range<allocnodemap::const_iterator>;

	using impnode_range = iterator_range<impnodemap::iterator>;
	using impnode_constrange = iterator_range<impnodemap::const_iterator>;

	using regnode_range = iterator_range<regnodemap::iterator>;
	using regnode_constrange = iterator_range<regnodemap::const_iterator>;

private:
	PointsToGraph();

public:
	PointsToGraph(const PointsToGraph&) = delete;

	PointsToGraph(PointsToGraph&&) = delete;

	PointsToGraph &
	operator=(const PointsToGraph&) = delete;

	PointsToGraph &
	operator=(PointsToGraph&&) = delete;

	allocnode_range
	allocnodes();

	allocnode_constrange
	allocnodes() const;

	impnode_range
	impnodes();

	impnode_constrange
	impnodes() const;

	regnode_range
	regnodes();

	regnode_constrange
	regnodes() const;

	iterator
	begin();

	constiterator
	begin() const;

	iterator
	end();

	constiterator
	end() const;

	size_t
	nallocnodes() const noexcept
	{
		return allocnodes_.size();
	}

	size_t
	nimpnodes() const noexcept
	{
		return impnodes_.size();
	}

	size_t
	nregnodes() const noexcept
	{
		return regnodes_.size();
	}

	size_t
	nnodes() const noexcept
	{
		return nallocnodes() + nimpnodes() + nregnodes();
	}

	PointsToGraph::UnknownNode &
	memunknown() const noexcept
	{
		return *memunknown_;
	}

	const PointsToGraph::AllocatorNode &
	GetAllocatorNode(const jive::node * node) const
	{
		auto it = allocnodes_.find(node);
		if (it == allocnodes_.end())
			throw error("Cannot find memory node in points-to graph.");

		return *it->second;
	}

	const PointsToGraph::ImportNode &
	GetImportNode(const jive::argument * argument) const
	{
		auto it = impnodes_.find(argument);
		if (it == impnodes_.end())
			throw error("Cannot find import node in points-to graph.");

		return *it->second;
	}

	const PointsToGraph::RegisterNode &
	GetRegisterNode(const jive::output * output) const
	{
		auto it = regnodes_.find(output);
		if (it == regnodes_.end())
			throw error("Cannot find register node in points-to graph.");

		return *it->second;
	}

	PointsToGraph::AllocatorNode &
	add(std::unique_ptr<PointsToGraph::AllocatorNode> node);

	PointsToGraph::RegisterNode &
	add(std::unique_ptr<PointsToGraph::RegisterNode> node);

	PointsToGraph::ImportNode &
	add(std::unique_ptr<PointsToGraph::ImportNode> node);

	static std::string
	ToDot(const PointsToGraph & ptg);

	static std::unique_ptr<PointsToGraph>
	Create()
	{
		return std::unique_ptr<PointsToGraph>(new PointsToGraph());
	}

private:
	impnodemap impnodes_;
	regnodemap regnodes_;
	allocnodemap allocnodes_;
	std::unique_ptr<PointsToGraph::UnknownNode> memunknown_;
};


/** \brief PointsTo graph node
*
*/
class PointsToGraph::Node {
	class constiterator;
	class iterator;

	using node_range = iterator_range<PointsToGraph::Node::iterator>;
	using node_constrange = iterator_range<PointsToGraph::Node::constiterator>;

public:
	virtual
	~Node();

  explicit
	Node(PointsToGraph & ptg)
	: ptg_(&ptg)
	{}

	Node(const Node&) = delete;

	Node(Node&&) = delete;

	Node&
	operator=(const Node&) = delete;

	Node&
	operator=(Node&&) = delete;

	node_range
	targets();

	node_constrange
	targets() const;

	node_range
	sources();

	node_constrange
	sources() const;

	PointsToGraph&
	Graph() const noexcept
	{
		return *ptg_;
	}

	size_t
	ntargets() const noexcept
	{
		return targets_.size();
	}

	size_t
	nsources() const noexcept
	{
		return sources_.size();
	}

	virtual std::string
	debug_string() const = 0;

	void
	add_edge(PointsToGraph::MemoryNode & target);

	void
	remove_edge(PointsToGraph::MemoryNode & target);

private:
	PointsToGraph * ptg_;
	std::unordered_set<PointsToGraph::Node*> targets_;
	std::unordered_set<PointsToGraph::Node*> sources_;
};


/** \brief PointsTo graph register node
*
*/
class PointsToGraph::RegisterNode final : public PointsToGraph::Node {
public:
	~RegisterNode() override;

private:
	RegisterNode(
    PointsToGraph & ptg,
    const jive::output * output)
    : Node(ptg)
    , output_(output)
	{}

public:
	const jive::output *
	output() const noexcept
	{
		return output_;
	}

	std::string
	debug_string() const override;

	/**
		FIXME: write documentation
	*/
	static std::vector<const PointsToGraph::MemoryNode*>
	allocators(const PointsToGraph::RegisterNode & node);

	static PointsToGraph::RegisterNode &
	create(
    PointsToGraph & ptg,
    const jive::output * output)
	{
		auto node = std::unique_ptr<PointsToGraph::RegisterNode>(new RegisterNode(ptg, output));
		return ptg.add(std::move(node));
	}

private:
	const jive::output * output_;
};


/** \brief PointsTo graph memory node
*
*/
class PointsToGraph::MemoryNode : public PointsToGraph::Node {
public:
	~MemoryNode() override;

protected:
  explicit
	MemoryNode(PointsToGraph & ptg)
	: Node(ptg)
	{}
};


/** \brief PointsTo graph allocator node
*
*/
class PointsToGraph::AllocatorNode final : public PointsToGraph::MemoryNode {
public:
	~AllocatorNode() override;

private:
	AllocatorNode(
    PointsToGraph & ptg,
    const jive::node * node)
    : MemoryNode(ptg)
    , node_(node)
	{}

public:
	const jive::node *
	node() const noexcept
	{
		return node_;
	}

	std::string
	debug_string() const override;

	static PointsToGraph::AllocatorNode &
	create(
    PointsToGraph & ptg,
    const jive::node * node)
	{
		auto n = std::unique_ptr<PointsToGraph::AllocatorNode>(new AllocatorNode(ptg, node));
		return ptg.add(std::move(n));
	}

private:
	const jive::node * node_;
};

/** \brief PointsTo graph import node
*
*/
class PointsToGraph::ImportNode final : public PointsToGraph::MemoryNode {
public:
	~ImportNode() override;

private:
	ImportNode(
    PointsToGraph & ptg,
    const jive::argument * argument)
    : MemoryNode(ptg)
    , argument_(argument)
	{
		JLM_ASSERT(dynamic_cast<const jlm::impport*>(&argument->port()));
	}

public:
	const jive::argument *
	argument() const noexcept
	{
		return argument_;
	}

	std::string
	debug_string() const override;

	static PointsToGraph::ImportNode &
	create(
    PointsToGraph & ptg,
    const jive::argument * argument)
	{
		auto n = std::unique_ptr<PointsToGraph::ImportNode>(new ImportNode(ptg, argument));
		return ptg.add(std::move(n));
	}

private:
	const jive::argument * argument_;
};

/** \brief PointsTo graph unknown node
*
*/
class PointsToGraph::UnknownNode final : public PointsToGraph::MemoryNode {
	friend PointsToGraph;

public:
	~UnknownNode() override;

private:
  explicit
	UnknownNode(PointsToGraph & ptg)
	: MemoryNode(ptg)
	{}

	std::string
	debug_string() const override;
};


/** \brief PointsTo graph node iterator
*/
class PointsToGraph::iterator final : public std::iterator<std::forward_iterator_tag,
	PointsToGraph::Node*, ptrdiff_t> {

	friend PointsToGraph;

	iterator(
		allocnodemap::iterator anit,
		impnodemap::iterator init,
		regnodemap::iterator rnit,
		/* FIXME: the full ranges are unnecessary here */
		const allocnode_range & anrange,
		const impnode_range & inrange,
		const regnode_range & rnrange)
	: anit_(anit)
	, init_(init)
	, rnit_(rnit)
	, anrange_(anrange)
	, inrange_(inrange)
	, rnrange_(rnrange)
	{}

public:
	PointsToGraph::Node *
	node() const noexcept
	{
		if (anit_ != anrange_.end())
			return anit_->second.get();

		if (init_ != inrange_.end())
			return init_->second.get();

		return rnit_->second.get();
	}

	PointsToGraph::Node &
	operator*() const
	{
		JLM_ASSERT(node() != nullptr);
		return *node();
	}

	PointsToGraph::Node *
	operator->() const
	{
		return node();
	}

	iterator &
	operator++()
	{
		if (anit_ != anrange_.end()) {
			++anit_;
			return *this;
		}

		if (init_ != inrange_.end()) {
			++init_;
			return *this;
		}

		++rnit_;
		return *this;
	}

	iterator
	operator++(int)
	{
		iterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const iterator & other) const
	{
		return anit_ == other.anit_
		    && init_ == other.init_
		    && rnit_ == other.rnit_;
	}

	bool
	operator!=(const iterator & other) const
	{
		return !operator==(other);
	}

private:
	allocnodemap::iterator anit_;
	impnodemap::iterator init_;
	regnodemap::iterator rnit_;

	allocnode_range anrange_;
	impnode_range inrange_;
	regnode_range rnrange_;
};

/** \brief PointsTo graph node const iterator
*/
class PointsToGraph::constiterator final : public std::iterator<std::forward_iterator_tag,
	const PointsToGraph::Node*, ptrdiff_t> {

	friend PointsToGraph;

	constiterator(
		allocnodemap::const_iterator anit,
		impnodemap::const_iterator init,
		regnodemap::const_iterator rnit,
		/* FIXME: the full ranges are unnecessary here */
		const allocnode_constrange & anrange,
		const impnode_constrange & inrange,
		const regnode_constrange & rnrange)
	: anit_(anit)
	, init_(init)
	, rnit_(rnit)
	, anrange_(anrange)
	, inrange_(inrange)
	, rnrange_(rnrange)
	{}

public:
	const PointsToGraph::Node *
	node() const noexcept
	{
		if (anit_ != anrange_.end())
			return anit_->second.get();

		if (init_ != inrange_.end())
			return init_->second.get();

		return rnit_->second.get();
	}

	const PointsToGraph::Node &
	operator*() const
	{
		JLM_ASSERT(node() != nullptr);
		return *node();
	}

	const PointsToGraph::Node *
	operator->() const
	{
		return node();
	}

	constiterator &
	operator++()
	{
		if (anit_ != anrange_.end()) {
			++anit_;
			return *this;
		}

		if (init_ != inrange_.end()) {
			++init_;
			return *this;
		}

		++rnit_;
		return *this;
	}

	constiterator
	operator++(int)
	{
		constiterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const constiterator & other) const
	{
		return anit_ == other.anit_
		    && init_ == other.init_
		    && rnit_ == other.rnit_;
	}

	bool
	operator!=(const constiterator & other) const
	{
		return !operator==(other);
	}

private:
	allocnodemap::const_iterator anit_;
	impnodemap::const_iterator init_;
	regnodemap::const_iterator rnit_;

	allocnode_constrange anrange_;
	impnode_constrange inrange_;
	regnode_constrange rnrange_;
};


/** \brief Points-to graph edge iterator
*/
class PointsToGraph::Node::iterator final : public std::iterator<std::forward_iterator_tag,
	PointsToGraph::Node*, ptrdiff_t> {

	friend PointsToGraph::Node;

	iterator(const std::unordered_set<PointsToGraph::Node*>::iterator & it)
	: it_(it)
	{}

public:
	PointsToGraph::Node *
	target() const noexcept
	{
		return *it_;
	}

	PointsToGraph::Node &
	operator*() const
	{
		JLM_ASSERT(target() != nullptr);
		return *target();
	}

	PointsToGraph::Node *
	operator->() const
	{
		return target();
	}

	iterator &
	operator++()
	{
		++it_;
		return *this;
	}

	iterator
	operator++(int)
	{
		iterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const iterator & other) const
	{
		return it_ == other.it_;
	}

	bool
	operator!=(const iterator & other) const
	{
		return !operator==(other);
	}

private:
	std::unordered_set<PointsToGraph::Node*>::iterator it_;
};


/** \brief Points-to graph edge const iterator
*/
class PointsToGraph::Node::constiterator final : public std::iterator<std::forward_iterator_tag,
	const PointsToGraph::Node*, ptrdiff_t> {

	friend PointsToGraph;

	constiterator(const std::unordered_set<PointsToGraph::Node*>::const_iterator & it)
	: it_(it)
	{}

public:
	const PointsToGraph::Node *
	target() const noexcept
	{
		return *it_;
	}

	const PointsToGraph::Node &
	operator*() const
	{
		JLM_ASSERT(target() != nullptr);
		return *target();
	}

	const PointsToGraph::Node *
	operator->() const
	{
		return target();
	}

	constiterator &
	operator++()
	{
		++it_;
		return *this;
	}

	constiterator
	operator++(int)
	{
		constiterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const constiterator & other) const
	{
		return it_ == other.it_;
	}

	bool
	operator!=(const constiterator & other) const
	{
		return !operator==(other);
	}

private:
	std::unordered_set<PointsToGraph::Node*>::const_iterator it_;
};

}}

#endif
