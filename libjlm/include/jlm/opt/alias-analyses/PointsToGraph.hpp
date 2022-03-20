/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP
#define JLM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP

#include <jlm/common.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/ir/operators/alloca.hpp>
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
  class AllocaNodeIterator;
  class AllocaNodeConstIterator;

  class AllocatorNodeIterator;
  class AllocatorNodeConstIterator;

  class ImportNodeIterator;
  class ImportNodeConstIterator;

  class RegisterNodeIterator;
  class RegisterNodeConstIterator;

public:
  class AllocaNode;
  class AllocatorNode;
  class ImportNode;
  class MemoryNode;
  class Node;
  class RegisterNode;
  class UnknownMemoryNode;
  class ExternalMemoryNode;

  using AllocaNodeMap = std::unordered_map<const jive::node*, std::unique_ptr<PointsToGraph::AllocaNode>>;
  using AllocatorNodeMap = std::unordered_map<const jive::node*, std::unique_ptr<PointsToGraph::AllocatorNode>>;
  using ImportNodeMap = std::unordered_map<const jive::argument*, std::unique_ptr<PointsToGraph::ImportNode>>;
  using RegisterNodeMap = std::unordered_map<const jive::output*, std::unique_ptr<PointsToGraph::RegisterNode>>;

  using AllocaNodeRange = iterator_range<AllocaNodeIterator>;
  using AllocaNodeConstRange = iterator_range<AllocaNodeConstIterator>;

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

  AllocaNodeRange
  AllocaNodes();

  AllocaNodeConstRange
  AllocaNodes() const;

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
  NumAllocaNodes() const noexcept
  {
    return AllocaNodes_.size();
  }

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
    return NumAllocaNodes() + NumAllocatorNodes() + NumImportNodes() + NumRegisterNodes();
  }

  PointsToGraph::UnknownMemoryNode &
  GetUnknownMemoryNode() const noexcept
  {
    return *UnknownMemoryNode_;
  }

  ExternalMemoryNode &
  GetExternalMemoryNode() const noexcept
  {
    return *ExternalMemoryNode_;
  }

  const PointsToGraph::AllocaNode &
  GetAllocaNode(const jive::node & node) const
  {
    auto it = AllocaNodes_.find(&node);
    if (it == AllocaNodes_.end())
      throw error("Cannot find alloca node in points-to graph.");

    return *it->second;
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

  PointsToGraph::AllocaNode &
  AddAllocaNode(std::unique_ptr<PointsToGraph::AllocaNode> node);

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
  AllocaNodeMap AllocaNodes_;
  ImportNodeMap ImportNodes_;
  RegisterNodeMap RegisterNodes_;
  AllocatorNodeMap AllocatorNodes_;
  std::unique_ptr<PointsToGraph::UnknownMemoryNode> UnknownMemoryNode_;
  std::unique_ptr<ExternalMemoryNode> ExternalMemoryNode_;
};


/** \brief PointsTo graph node
*
*/
class PointsToGraph::Node {
  class ConstIterator;
  class Iterator;

  using NodeRange = iterator_range<PointsToGraph::Node::Iterator>;
  using NodeConstRange = iterator_range<PointsToGraph::Node::ConstIterator>;

public:
  virtual
  ~Node() noexcept;

  explicit
  Node(PointsToGraph & pointsToGraph)
    : PointsToGraph_(&pointsToGraph)
  {}

  Node(const Node&) = delete;

  Node(Node&&) = delete;

  Node&
  operator=(const Node&) = delete;

  Node&
  operator=(Node&&) = delete;

  NodeRange
  Targets();

  NodeConstRange
  Targets() const;

  NodeRange
  Sources();

  NodeConstRange
  Sources() const;

  PointsToGraph&
  Graph() const noexcept
  {
    return *PointsToGraph_;
  }

  size_t
  NumTargets() const noexcept
  {
    return Targets_.size();
  }

  size_t
  NumSources() const noexcept
  {
    return Sources_.size();
  }

  virtual std::string
  DebugString() const = 0;

  void
  AddEdge(PointsToGraph::MemoryNode & target);

  void
  RemoveEdge(PointsToGraph::MemoryNode & target);

private:
  PointsToGraph * PointsToGraph_;
  std::unordered_set<PointsToGraph::Node*> Targets_;
  std::unordered_set<PointsToGraph::Node*> Sources_;
};

/** \brief PointsTo graph register node
*
*/
class PointsToGraph::RegisterNode final : public PointsToGraph::Node {
public:
  ~RegisterNode() noexcept override;

private:
  RegisterNode(
    PointsToGraph & pointsToGraph,
    const jive::output & output)
    : Node(pointsToGraph)
    , Output_(&output)
  {}

public:
  const jive::output &
  GetOutput() const noexcept
  {
    return *Output_;
  }

  std::string
  DebugString() const override;

  /**
    FIXME: write documentation
  */
  static std::vector<const PointsToGraph::MemoryNode*>
  GetMemoryNodes(const PointsToGraph::RegisterNode & node);

  static PointsToGraph::RegisterNode &
  Create(
    PointsToGraph & pointsToGraph,
    const jive::output & output)
  {
    auto node = std::unique_ptr<PointsToGraph::RegisterNode>(new RegisterNode(pointsToGraph, output));
    return pointsToGraph.AddRegisterNode(std::move(node));
  }

private:
  const jive::output * Output_;
};

/** \brief PointsTo graph memory node
*
*/
class PointsToGraph::MemoryNode : public PointsToGraph::Node {
public:
  ~MemoryNode() noexcept override;

protected:
  explicit
  MemoryNode(PointsToGraph & pointsToGraph)
    : Node(pointsToGraph)
  {}
};

/** \brief PointsTo graph alloca node
 *
 */
class PointsToGraph::AllocaNode final : public PointsToGraph::MemoryNode {
public:
  ~AllocaNode() noexcept override;

private:
  AllocaNode(
    PointsToGraph & pointsToGraph,
  const jive::node & allocaNode)
  : MemoryNode(pointsToGraph)
  , AllocaNode_(&allocaNode)
  {
    JLM_ASSERT(is<alloca_op>(&allocaNode));
  }

public:
  const jive::node &
  GetAllocaNode() const noexcept
  {
    return *AllocaNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::AllocaNode &
  Create(
    PointsToGraph & pointsToGraph,
    const jive::node & node)
  {
    auto n = std::unique_ptr<PointsToGraph::AllocaNode>(new AllocaNode(pointsToGraph, node));
    return pointsToGraph.AddAllocaNode(std::move(n));
  }

private:
  const jive::node * AllocaNode_;
};

/** \brief PointsTo graph allocator node
*
*/
class PointsToGraph::AllocatorNode final : public PointsToGraph::MemoryNode {
public:
  ~AllocatorNode() noexcept override;

private:
  AllocatorNode(
    PointsToGraph & pointsToGraph,
    const jive::node & node)
    : MemoryNode(pointsToGraph)
    , Node_(&node)
  {}

public:
  const jive::node &
  GetNode() const noexcept
  {
    return *Node_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::AllocatorNode &
  Create(
    PointsToGraph & pointsToGraph,
    const jive::node & node)
  {
    auto n = std::unique_ptr<PointsToGraph::AllocatorNode>(new AllocatorNode(pointsToGraph, node));
    return pointsToGraph.AddAllocatorNode(std::move(n));
  }

private:
  const jive::node * Node_;
};

/** \brief PointsTo graph import node
*
*/
class PointsToGraph::ImportNode final : public PointsToGraph::MemoryNode {
public:
  ~ImportNode() noexcept override;

private:
  ImportNode(
    PointsToGraph & pointsToGraph,
    const jive::argument & argument)
    : MemoryNode(pointsToGraph)
    , Argument_(&argument)
  {
    JLM_ASSERT(dynamic_cast<const impport*>(&argument.port()));
  }

public:
  const jive::argument &
  GetArgument() const noexcept
  {
    return *Argument_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::ImportNode &
  Create(
    PointsToGraph & pointsToGraph,
    const jive::argument & argument)
  {
    auto n = std::unique_ptr<PointsToGraph::ImportNode>(new ImportNode(pointsToGraph, argument));
    return pointsToGraph.AddImportNode(std::move(n));
  }

private:
  const jive::argument * Argument_;
};

/** \brief PointsTo graph unknown node
*
*/
class PointsToGraph::UnknownMemoryNode final : public PointsToGraph::MemoryNode {
  friend PointsToGraph;

public:
  ~UnknownMemoryNode() noexcept override;

private:
  explicit
  UnknownMemoryNode(PointsToGraph & pointsToGraph)
    : MemoryNode(pointsToGraph)
  {}

  std::string
  DebugString() const override;

  static std::unique_ptr<UnknownMemoryNode>
  Create(PointsToGraph & pointsToGraph)
  {
    return std::unique_ptr<UnknownMemoryNode>(new UnknownMemoryNode(pointsToGraph));
  }
};

/** \brief PointsTo graph external memory node
 *
 */
class PointsToGraph::ExternalMemoryNode final : public PointsToGraph::MemoryNode {
  friend PointsToGraph;

public:
  ~ExternalMemoryNode() noexcept override;

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
  DebugString() const override;
};

/** \brief Points-to graph alloca node iterator
*/
class PointsToGraph::AllocaNodeIterator final : public std::iterator<std::forward_iterator_tag,
  PointsToGraph::AllocaNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  AllocaNodeIterator(const AllocaNodeMap::iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] PointsToGraph::AllocaNode *
  AllocaNode() const noexcept
  {
    return it_->second.get();
  }

  PointsToGraph::AllocaNode &
  operator*() const
  {
    JLM_ASSERT(AllocaNode() != nullptr);
    return *AllocaNode();
  }

  PointsToGraph::AllocaNode *
  operator->() const
  {
    return AllocaNode();
  }

  AllocaNodeIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  AllocaNodeIterator
  operator++(int)
  {
    AllocaNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const AllocaNodeIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const AllocaNodeIterator & other) const
  {
    return !operator==(other);
  }

private:
  AllocaNodeMap::iterator it_;
};

/** \brief Points-to graph alloca node const iterator
*/
class PointsToGraph::AllocaNodeConstIterator final : public std::iterator<std::forward_iterator_tag,
  const PointsToGraph::AllocaNode*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  AllocaNodeConstIterator(const AllocaNodeMap::const_iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] const PointsToGraph::AllocaNode *
  AllocaNode() const noexcept
  {
    return it_->second.get();
  }

  const PointsToGraph::AllocaNode &
  operator*() const
  {
    JLM_ASSERT(AllocaNode() != nullptr);
    return *AllocaNode();
  }

  const PointsToGraph::AllocaNode *
  operator->() const
  {
    return AllocaNode();
  }

  AllocaNodeConstIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  AllocaNodeConstIterator
  operator++(int)
  {
    AllocaNodeConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const AllocaNodeConstIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const AllocaNodeConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  AllocaNodeMap::const_iterator it_;
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
class PointsToGraph::Node::Iterator final : public std::iterator<std::forward_iterator_tag,
  PointsToGraph::Node*, ptrdiff_t> {

  friend PointsToGraph::Node;

  explicit
  Iterator(const std::unordered_set<PointsToGraph::Node*>::iterator & it)
    : It_(it)
  {}

public:
  [[nodiscard]] PointsToGraph::Node *
  GetNode() const noexcept
  {
    return *It_;
  }

  PointsToGraph::Node &
  operator*() const
  {
    JLM_ASSERT(GetNode() != nullptr);
    return *GetNode();
  }

  PointsToGraph::Node *
  operator->() const
  {
    return GetNode();
  }

  Iterator &
  operator++()
  {
    ++It_;
    return *this;
  }

  Iterator
  operator++(int)
  {
    Iterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const Iterator & other) const
  {
    return It_ == other.It_;
  }

  bool
  operator!=(const Iterator & other) const
  {
    return !operator==(other);
  }

private:
  std::unordered_set<PointsToGraph::Node*>::iterator It_;
};

/** \brief Points-to graph edge const iterator
*/
class PointsToGraph::Node::ConstIterator final : public std::iterator<std::forward_iterator_tag,
  const PointsToGraph::Node*, ptrdiff_t> {

  friend PointsToGraph;

  explicit
  ConstIterator(const std::unordered_set<PointsToGraph::Node*>::const_iterator & it)
    : It_(it)
  {}

public:
  [[nodiscard]] const PointsToGraph::Node *
  GetNode() const noexcept
  {
    return *It_;
  }

  const PointsToGraph::Node &
  operator*() const
  {
    JLM_ASSERT(GetNode() != nullptr);
    return *GetNode();
  }

  const PointsToGraph::Node *
  operator->() const
  {
    return GetNode();
  }

  ConstIterator &
  operator++()
  {
    ++It_;
    return *this;
  }

  ConstIterator
  operator++(int)
  {
    ConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const ConstIterator & other) const
  {
    return It_ == other.It_;
  }

  bool
  operator!=(const ConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  std::unordered_set<PointsToGraph::Node*>::const_iterator It_;
};

}}

#endif
