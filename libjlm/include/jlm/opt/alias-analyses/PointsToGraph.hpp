/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP
#define JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP

#include <jlm/common.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/ir/RvsdgModule.hpp>

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

class RvsdgModule;

namespace aa {

/** /brief PointsTo Graph
*
*/
class PointsToGraph final {
  class AllocatorNodeIterator;
  class AllocatorNodeConstIterator;

  class ImportNodeIterator;
  class ImportNodeConstIterator;

  class RegisterNodeIterator;
  class RegisterNodeConstIterator;

public:
  class AllocatorNode;
  class ImportNode;
  class MemoryNode;
  class Node;
  class RegisterNode;
  class UnknownNode;
  class ExternalMemoryNode;

  using AllocatorNodeMap = std::unordered_map<const jive::node*, std::unique_ptr<PointsToGraph::AllocatorNode>>;
  using ImportNodeMap = std::unordered_map<const jive::argument*, std::unique_ptr<PointsToGraph::ImportNode>>;
  using RegisterNodeMap = std::unordered_map<const jive::output*, std::unique_ptr<PointsToGraph::RegisterNode>>;

  using AllocatorNodeRange = iterator_range<AllocatorNodeIterator>;
  using AllocatorNodeConstRange = iterator_range<AllocatorNodeConstIterator>;

  using ImportNodeRange = iterator_range<ImportNodeIterator>;
  using ImportNodeConstRange = iterator_range<ImportNodeConstIterator>;

  using RegisterNodeRange = iterator_range<RegisterNodeIterator>;
  using RegisterNodeConstRange = iterator_range<RegisterNodeConstIterator>;

private:
  PointsToGraph();

public:
  PointsToGraph(const PointsToGraph&) = delete;

  PointsToGraph(PointsToGraph&&) = delete;

  PointsToGraph &
  operator=(const PointsToGraph&) = delete;

  PointsToGraph &
  operator=(PointsToGraph&&) = delete;

  AllocatorNodeRange
  AllocatorNodes();

  AllocatorNodeConstRange
  AllocatorNodes() const;

  ImportNodeRange
  ImportNodes();

  ImportNodeConstRange
  ImportNodes() const;

  RegisterNodeRange
  RegisterNodes();

  RegisterNodeConstRange
  RegisterNodes() const;

  size_t
  NumAllocatorNodes() const noexcept
  {
    return AllocatorNodes_.size();
  }

  size_t
  NumImportNodes() const noexcept
  {
    return ImportNodes_.size();
  }

  size_t
  NumRegisterNodes() const noexcept
  {
    return RegisterNodes_.size();
  }

  size_t
  NumNodes() const noexcept
  {
    return NumAllocatorNodes() + NumImportNodes() + NumRegisterNodes();
  }

  PointsToGraph::UnknownNode &
  GetUnknownMemoryNode() const noexcept
  {
    return *UnknownMemoryNode_;
  }

  ExternalMemoryNode &
  GetExternalMemoryNode() const noexcept
  {
    return *ExternalMemoryNode_;
  }

  const PointsToGraph::AllocatorNode &
  GetAllocatorNode(const jive::node & node) const
  {
    auto it = AllocatorNodes_.find(&node);
    if (it == AllocatorNodes_.end())
      throw error("Cannot find memory node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::ImportNode &
  GetImportNode(const jive::argument & argument) const
  {
    auto it = ImportNodes_.find(&argument);
    if (it == ImportNodes_.end())
      throw error("Cannot find import node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::RegisterNode &
  GetRegisterNode(const jive::output & output) const
  {
    auto it = RegisterNodes_.find(&output);
    if (it == RegisterNodes_.end())
      throw error("Cannot find register node in points-to graph.");

    return *it->second;
  }

  PointsToGraph::AllocatorNode &
  AddAllocatorNode(std::unique_ptr<PointsToGraph::AllocatorNode> node);

  PointsToGraph::RegisterNode &
  AddRegisterNode(std::unique_ptr<PointsToGraph::RegisterNode> node);

  PointsToGraph::ImportNode &
  AddImportNode(std::unique_ptr<PointsToGraph::ImportNode> node);

  static std::string
  ToDot(const PointsToGraph & pointsToGraph);

  static std::unique_ptr<PointsToGraph>
  Create()
  {
    return std::unique_ptr<PointsToGraph>(new PointsToGraph());
  }

private:
  ImportNodeMap ImportNodes_;
  RegisterNodeMap RegisterNodes_;
  AllocatorNodeMap AllocatorNodes_;
  std::unique_ptr<PointsToGraph::UnknownNode> UnknownMemoryNode_;
  std::unique_ptr<ExternalMemoryNode> ExternalMemoryNode_;
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
		return ptg.AddRegisterNode(std::move(node));
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
		return ptg.AddAllocatorNode(std::move(n));
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
		return ptg.AddImportNode(std::move(n));
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

class PointsToGraph::ExternalMemoryNode final : public PointsToGraph::MemoryNode {
  friend PointsToGraph;

public:
  ~ExternalMemoryNode() override;

private:
  explicit
  ExternalMemoryNode(PointsToGraph & pointsToGraph)
  : MemoryNode(pointsToGraph)
  {}

  static std::unique_ptr<ExternalMemoryNode>
  Create(PointsToGraph & pointsToGraph)
  {
    return std::unique_ptr<ExternalMemoryNode>(new ExternalMemoryNode(pointsToGraph));
  }

  std::string
  debug_string() const override;
};

/** \brief Points-to graph allocator node iterator
*/
class PointsToGraph::AllocatorNodeIterator final : public std::iterator<std::forward_iterator_tag,
  PointsToGraph::AllocatorNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  AllocatorNodeIterator(const AllocatorNodeMap::iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] PointsToGraph::AllocatorNode *
  AllocatorNode() const noexcept
  {
    return it_->second.get();
  }

  PointsToGraph::AllocatorNode &
  operator*() const
  {
    JLM_ASSERT(AllocatorNode() != nullptr);
    return *AllocatorNode();
  }

  PointsToGraph::AllocatorNode *
  operator->() const
  {
    return AllocatorNode();
  }

  AllocatorNodeIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  AllocatorNodeIterator
  operator++(int)
  {
    AllocatorNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const AllocatorNodeIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const AllocatorNodeIterator & other) const
  {
    return !operator==(other);
  }

private:
  AllocatorNodeMap::iterator it_;
};

/** \brief Points-to graph allocator node const iterator
*/
class PointsToGraph::AllocatorNodeConstIterator final : public std::iterator<std::forward_iterator_tag,
  const PointsToGraph::AllocatorNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  AllocatorNodeConstIterator(const AllocatorNodeMap::const_iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] const PointsToGraph::AllocatorNode *
  AllocatorNode() const noexcept
  {
    return it_->second.get();
  }

  const PointsToGraph::AllocatorNode &
  operator*() const
  {
    JLM_ASSERT(AllocatorNode() != nullptr);
    return *AllocatorNode();
  }

  const PointsToGraph::AllocatorNode *
  operator->() const
  {
    return AllocatorNode();
  }

  AllocatorNodeConstIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  AllocatorNodeConstIterator
  operator++(int)
  {
    AllocatorNodeConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const AllocatorNodeConstIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const AllocatorNodeConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  AllocatorNodeMap::const_iterator it_;
};

/** \brief Points-to graph import node iterator
*/
class PointsToGraph::ImportNodeIterator final : public std::iterator<std::forward_iterator_tag,
  PointsToGraph::ImportNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  ImportNodeIterator(const ImportNodeMap::iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] PointsToGraph::ImportNode *
  ImportNode() const noexcept
  {
    return it_->second.get();
  }

  PointsToGraph::ImportNode &
  operator*() const
  {
    JLM_ASSERT(ImportNode() != nullptr);
    return *ImportNode();
  }

  PointsToGraph::ImportNode *
  operator->() const
  {
    return ImportNode();
  }

  ImportNodeIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  ImportNodeIterator
  operator++(int)
  {
    ImportNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const ImportNodeIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const ImportNodeIterator & other) const
  {
    return !operator==(other);
  }

private:
  ImportNodeMap::iterator it_;
};

/** \brief Points-to graph import node const iterator
*/
class PointsToGraph::ImportNodeConstIterator final : public std::iterator<std::forward_iterator_tag,
  const PointsToGraph::ImportNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  ImportNodeConstIterator(const ImportNodeMap::const_iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] const PointsToGraph::ImportNode *
  ImportNode() const noexcept
  {
    return it_->second.get();
  }

  const PointsToGraph::ImportNode &
  operator*() const
  {
    JLM_ASSERT(ImportNode() != nullptr);
    return *ImportNode();
  }

  const PointsToGraph::ImportNode *
  operator->() const
  {
    return ImportNode();
  }

  ImportNodeConstIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  ImportNodeConstIterator
  operator++(int)
  {
    ImportNodeConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const ImportNodeConstIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const ImportNodeConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  ImportNodeMap::const_iterator it_;
};

/** \brief Points-to graph register node iterator
*/
class PointsToGraph::RegisterNodeIterator final : public std::iterator<std::forward_iterator_tag,
  PointsToGraph::RegisterNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  RegisterNodeIterator(const RegisterNodeMap::iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] PointsToGraph::RegisterNode *
  RegisterNode() const noexcept
  {
    return it_->second.get();
  }

  PointsToGraph::RegisterNode &
  operator*() const
  {
    JLM_ASSERT(RegisterNode() != nullptr);
    return *RegisterNode();
  }

  PointsToGraph::RegisterNode *
  operator->() const
  {
    return RegisterNode();
  }

  RegisterNodeIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  RegisterNodeIterator
  operator++(int)
  {
    RegisterNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const RegisterNodeIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const RegisterNodeIterator & other) const
  {
    return !operator==(other);
  }

private:
  RegisterNodeMap::iterator it_;
};

/** \brief Points-to graph register node const iterator
*/
class PointsToGraph::RegisterNodeConstIterator final : public std::iterator<std::forward_iterator_tag,
  const PointsToGraph::RegisterNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  RegisterNodeConstIterator(const RegisterNodeMap::const_iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] const PointsToGraph::RegisterNode *
  RegisterNode() const noexcept
  {
    return it_->second.get();
  }

  const PointsToGraph::RegisterNode &
  operator*() const
  {
    JLM_ASSERT(RegisterNode() != nullptr);
    return *RegisterNode();
  }

  const PointsToGraph::RegisterNode *
  operator->() const
  {
    return RegisterNode();
  }

  RegisterNodeConstIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  RegisterNodeConstIterator
  operator++(int)
  {
    RegisterNodeConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const RegisterNodeConstIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const RegisterNodeConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  RegisterNodeMap::const_iterator it_;
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
