/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/HashSet.hpp>
#include <jlm/util/iterator_range.hpp>

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
  template<class DATATYPE, class ITERATORTYPE>
  class NodeIterator;

  template<class DATATYPE, class ITERATORTYPE>
  class NodeConstIterator;

public:
  class AllocaNode;
  class DeltaNode;
  class ImportNode;
  class LambdaNode;
  class MallocNode;
  class MemoryNode;
  class Node;
  class RegisterNode;
  class UnknownMemoryNode;
  class ExternalMemoryNode;

  using AllocaNodeMap = std::unordered_map<const jive::node*, std::unique_ptr<PointsToGraph::AllocaNode>>;
  using DeltaNodeMap = std::unordered_map<const delta::node*, std::unique_ptr<PointsToGraph::DeltaNode>>;
  using ImportNodeMap = std::unordered_map<const jive::argument*, std::unique_ptr<PointsToGraph::ImportNode>>;
  using LambdaNodeMap = std::unordered_map<const lambda::node*, std::unique_ptr<PointsToGraph::LambdaNode>>;
  using MallocNodeMap = std::unordered_map<const jive::node*, std::unique_ptr<PointsToGraph::MallocNode>>;
  using RegisterNodeMap = std::unordered_map<const jive::output*, std::unique_ptr<PointsToGraph::RegisterNode>>;

  using AllocaNodeIterator = NodeIterator<AllocaNode, AllocaNodeMap::iterator>;
  using AllocaNodeConstIterator = NodeConstIterator<AllocaNode, AllocaNodeMap::const_iterator>;
  using AllocaNodeRange = iterator_range<AllocaNodeIterator>;
  using AllocaNodeConstRange = iterator_range<AllocaNodeConstIterator>;

  using DeltaNodeIterator = NodeIterator<DeltaNode, DeltaNodeMap::iterator>;
  using DeltaNodeConstIterator = NodeConstIterator<DeltaNode, DeltaNodeMap::const_iterator>;
  using DeltaNodeRange = iterator_range<DeltaNodeIterator>;
  using DeltaNodeConstRange = iterator_range<DeltaNodeConstIterator>;

  using ImportNodeIterator = NodeIterator<ImportNode, ImportNodeMap::iterator>;
  using ImportNodeConstIterator = NodeConstIterator<ImportNode, ImportNodeMap::const_iterator>;
  using ImportNodeRange = iterator_range<ImportNodeIterator>;
  using ImportNodeConstRange = iterator_range<ImportNodeConstIterator>;

  using LambdaNodeIterator = NodeIterator<LambdaNode, LambdaNodeMap::iterator>;
  using LambdaNodeConstIterator = NodeConstIterator<LambdaNode, LambdaNodeMap::const_iterator>;
  using LambdaNodeRange = iterator_range<LambdaNodeIterator>;
  using LambdaNodeConstRange = iterator_range<LambdaNodeConstIterator>;

  using MallocNodeIterator = NodeIterator<MallocNode, MallocNodeMap::iterator>;
  using MallocNodeConstIterator = NodeConstIterator<MallocNode, MallocNodeMap::const_iterator>;
  using MallocNodeRange = iterator_range<MallocNodeIterator>;
  using MallocNodeConstRange = iterator_range<MallocNodeConstIterator>;

  using RegisterNodeIterator = NodeIterator<RegisterNode, RegisterNodeMap::iterator>;
  using RegisterNodeConstIterator = NodeConstIterator<RegisterNode, RegisterNodeMap::const_iterator>;
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

  DeltaNodeRange
  DeltaNodes();

  DeltaNodeConstRange
  DeltaNodes() const;

  ImportNodeRange
  ImportNodes();

  ImportNodeConstRange
  ImportNodes() const;

  LambdaNodeRange
  LambdaNodes();

  LambdaNodeConstRange
  LambdaNodes() const;

  MallocNodeRange
  MallocNodes();

  MallocNodeConstRange
  MallocNodes() const;

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
  NumDeltaNodes() const noexcept
  {
    return DeltaNodes_.size();
  }

  size_t
  NumImportNodes() const noexcept
  {
    return ImportNodes_.size();
  }

  size_t
  NumLambdaNodes() const noexcept
  {
    return LambdaNodes_.size();
  }

  size_t
  NumMallocNodes() const noexcept
  {
    return MallocNodes_.size();
  }

  size_t
  NumRegisterNodes() const noexcept
  {
    return RegisterNodes_.size();
  }

  size_t
  NumMemoryNodes() const noexcept
  {
    return NumAllocaNodes()
           + NumDeltaNodes()
           + NumImportNodes()
           + NumLambdaNodes()
           + NumMallocNodes()
           + 1; //External memory node
  }

  size_t
  NumNodes() const noexcept
  {
    return NumMemoryNodes() + NumRegisterNodes();
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

  const PointsToGraph::DeltaNode &
  GetDeltaNode(const delta::node & node) const
  {
    auto it = DeltaNodes_.find(&node);
    if (it == DeltaNodes_.end())
      throw error("Cannot find delta node in points-to graph.");

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

  const PointsToGraph::LambdaNode &
  GetLambdaNode(const lambda::node & node) const
  {
    auto it = LambdaNodes_.find(&node);
    if (it == LambdaNodes_.end())
      throw error("Cannot find lambda node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::MallocNode &
  GetMallocNode(const jive::node & node) const
  {
    auto it = MallocNodes_.find(&node);
    if (it == MallocNodes_.end())
      throw error("Cannot find malloc node in points-to graph.");

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

  /**
   * Returns all memory nodes that are marked as escaped from the module.
   *
   * @return A set with all escaped memory nodes.
   *
   * @see PointsToGraph::MemoryNode::MarkAsModuleEscaping()
   */
  const HashSet<const PointsToGraph::MemoryNode*> &
  GetEscapedMemoryNodes() const noexcept
  {
    return EscapedMemoryNodes_;
  }

  PointsToGraph::AllocaNode &
  AddAllocaNode(std::unique_ptr<PointsToGraph::AllocaNode> node);

  PointsToGraph::DeltaNode &
  AddDeltaNode(std::unique_ptr<PointsToGraph::DeltaNode> node);

  PointsToGraph::LambdaNode &
  AddLambdaNode(std::unique_ptr<PointsToGraph::LambdaNode> node);

  PointsToGraph::MallocNode &
  AddMallocNode(std::unique_ptr<PointsToGraph::MallocNode> node);

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
  void
  AddEscapedMemoryNode(PointsToGraph::MemoryNode & memoryNode);

  /**
   * All memory nodes that escape from the module.
   */
  HashSet<const PointsToGraph::MemoryNode*> EscapedMemoryNodes_;

  AllocaNodeMap AllocaNodes_;
  DeltaNodeMap DeltaNodes_;
  ImportNodeMap ImportNodes_;
  LambdaNodeMap LambdaNodes_;
  MallocNodeMap MallocNodes_;
  RegisterNodeMap RegisterNodes_;
  std::unique_ptr<PointsToGraph::UnknownMemoryNode> UnknownMemoryNode_;
  std::unique_ptr<ExternalMemoryNode> ExternalMemoryNode_;
};

/** \brief PointsTo graph node
*
*/
class PointsToGraph::Node {
  template<class NODETYPE> class ConstIterator;
  template<class NODETYPE> class Iterator;

  using SourceIterator = Iterator<PointsToGraph::Node>;
  using SourceConstIterator = ConstIterator<PointsToGraph::Node>;

  using TargetIterator = Iterator<PointsToGraph::MemoryNode>;
  using TargetConstIterator = ConstIterator<PointsToGraph::MemoryNode>;

  using SourceRange = iterator_range<SourceIterator>;
  using SourceConstRange = iterator_range<SourceConstIterator>;

  using TargetRange = iterator_range<TargetIterator>;
  using TargetConstRange = iterator_range<TargetConstIterator>;

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

  TargetRange
  Targets();

  TargetConstRange
  Targets() const;

  SourceRange
  Sources();

  SourceConstRange
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
  std::unordered_set<PointsToGraph::MemoryNode*> Targets_;
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

  /**
   * Marks this memory node as escaping the module.
   */
  void
  MarkAsModuleEscaping()
  {
    Graph().AddEscapedMemoryNode(*this);
  }

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

/** \brief PointsTo graph delta node
 *
 */
class PointsToGraph::DeltaNode final : public PointsToGraph::MemoryNode {
public:
  ~DeltaNode() noexcept override;

private:
  DeltaNode(
    PointsToGraph & pointsToGraph,
    const delta::node & deltaNode)
    : MemoryNode(pointsToGraph)
    , DeltaNode_(&deltaNode)
  {
    JLM_ASSERT(is<delta::operation>(&deltaNode));
  }

public:
  const delta::node &
  GetDeltaNode() const noexcept
  {
    return *DeltaNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::DeltaNode &
  Create(
    PointsToGraph & pointsToGraph,
    const delta::node & deltaNode)
  {
    auto n = std::unique_ptr<PointsToGraph::DeltaNode>(new DeltaNode(pointsToGraph, deltaNode));
    return pointsToGraph.AddDeltaNode(std::move(n));
  }

private:
  const delta::node * DeltaNode_;
};

/** \brief PointsTo graph malloc node
 *
 */
class PointsToGraph::MallocNode final : public PointsToGraph::MemoryNode {
public:
  ~MallocNode() noexcept override;

private:
  MallocNode(
    PointsToGraph & pointsToGraph,
    const jive::node & mallocNode)
    : MemoryNode(pointsToGraph)
    , MallocNode_(&mallocNode)
  {
    JLM_ASSERT(is<malloc_op>(&mallocNode));
  }

public:
  const jive::node &
  GetMallocNode() const noexcept
  {
    return *MallocNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::MallocNode &
  Create(
    PointsToGraph & pointsToGraph,
    const jive::node & node)
  {
    auto n = std::unique_ptr<PointsToGraph::MallocNode>(new MallocNode(pointsToGraph, node));
    return pointsToGraph.AddMallocNode(std::move(n));
  }

private:
  const jive::node * MallocNode_;
};

/** \brief PointsTo graph malloc node
 *
 */
class PointsToGraph::LambdaNode final : public PointsToGraph::MemoryNode {
public:
  ~LambdaNode() noexcept override;

private:
  LambdaNode(
    PointsToGraph & pointsToGraph,
    const lambda::node & lambdaNode)
    : MemoryNode(pointsToGraph)
    , LambdaNode_(&lambdaNode)
  {
    JLM_ASSERT(is<lambda::operation>(&lambdaNode));
  }

public:
  const lambda::node &
  GetLambdaNode() const noexcept
  {
    return *LambdaNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::LambdaNode &
  Create(
    PointsToGraph & pointsToGraph,
    const lambda::node & lambdaNode)
  {
    auto n = std::unique_ptr<PointsToGraph::LambdaNode>(new LambdaNode(pointsToGraph, lambdaNode));
    return pointsToGraph.AddLambdaNode(std::move(n));
  }

private:
  const lambda::node * LambdaNode_;
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

/** \brief Points-to graph node iterator
*/
template<class DATATYPE, class ITERATORTYPE>
class PointsToGraph::NodeIterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = DATATYPE*;
  using difference_type = std::ptrdiff_t;
  using pointer = DATATYPE**;
  using reference = DATATYPE*&;

private:
  friend PointsToGraph;

  explicit
  NodeIterator(const ITERATORTYPE & it)
    : it_(it)
  {}

public:
  [[nodiscard]] DATATYPE *
  Node() const noexcept
  {
    return it_->second.get();
  }

  DATATYPE &
  operator*() const
  {
    JLM_ASSERT(Node() != nullptr);
    return *Node();
  }

  DATATYPE *
  operator->() const
  {
    return Node();
  }

  NodeIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  NodeIterator
  operator++(int)
  {
    NodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const NodeIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const NodeIterator & other) const
  {
    return !operator==(other);
  }

private:
  ITERATORTYPE it_;
};

/** \brief Points-to graph node const iterator
*/
template<class DATATYPE, class ITERATORTYPE>
class PointsToGraph::NodeConstIterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = const DATATYPE*;
  using difference_type = std::ptrdiff_t;
  using pointer = const DATATYPE**;
  using reference = const DATATYPE*&;

private:
  friend PointsToGraph;

  explicit
  NodeConstIterator(const ITERATORTYPE & it)
    : it_(it)
  {}

public:
  [[nodiscard]] const DATATYPE *
  Node() const noexcept
  {
    return it_->second.get();
  }

  const DATATYPE &
  operator*() const
  {
    JLM_ASSERT(Node() != nullptr);
    return *Node();
  }

  const DATATYPE *
  operator->() const
  {
    return Node();
  }

  NodeConstIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  NodeConstIterator
  operator++(int)
  {
    NodeConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const NodeConstIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const NodeConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  ITERATORTYPE it_;
};

/** \brief Points-to graph edge iterator
*/
template <class NODETYPE>
class PointsToGraph::Node::Iterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = NODETYPE*;
  using difference_type = std::ptrdiff_t;
  using pointer = NODETYPE**;
  using reference = NODETYPE*&;

private:
  friend PointsToGraph::Node;

  explicit
  Iterator(const typename std::unordered_set<NODETYPE*>::iterator & it)
    : It_(it)
  {}

public:
  [[nodiscard]] NODETYPE *
  GetNode() const noexcept
  {
    return *It_;
  }

  NODETYPE &
  operator*() const
  {
    JLM_ASSERT(GetNode() != nullptr);
    return *GetNode();
  }

  NODETYPE *
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
  typename std::unordered_set<NODETYPE*>::iterator It_;
};

/** \brief Points-to graph edge const iterator
*/
template <class NODETYPE>
class PointsToGraph::Node::ConstIterator final
{
public:
  using itearator_category = std::forward_iterator_tag;
  using value_type = const NODETYPE*;
  using difference_type = std::ptrdiff_t;
  using pointer = const NODETYPE**;
  using reference = const NODETYPE*&;

private:
  friend PointsToGraph;

  explicit
  ConstIterator(const typename std::unordered_set<NODETYPE*>::const_iterator & it)
    : It_(it)
  {}

public:
  [[nodiscard]] const NODETYPE *
  GetNode() const noexcept
  {
    return *It_;
  }

  const NODETYPE &
  operator*() const
  {
    JLM_ASSERT(GetNode() != nullptr);
    return *GetNode();
  }

  const NODETYPE *
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
  typename std::unordered_set<NODETYPE*>::const_iterator It_;
};

}}

#endif