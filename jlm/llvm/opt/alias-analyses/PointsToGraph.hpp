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

namespace jlm::llvm
{

class RvsdgModule;

namespace aa
{

/** /brief PointsTo Graph
 *
 */
class PointsToGraph final
{
  template<class DATATYPE, class ITERATORTYPE, typename IteratorToPointer>
  class NodeIterator;

  template<class DATATYPE, class ITERATORTYPE, typename IteratorToPointer>
  class NodeConstIterator;

public:
  class AllocaNode;
  class DeltaNode;
  class ImportNode;
  class LambdaNode;
  class MallocNode;
  class MemoryNode;
  class Node;
  class RegisterSetNode;
  class UnknownMemoryNode;
  class ExternalMemoryNode;

  using AllocaNodeMap =
      std::unordered_map<const jlm::rvsdg::node *, std::unique_ptr<PointsToGraph::AllocaNode>>;
  using DeltaNodeMap =
      std::unordered_map<const delta::node *, std::unique_ptr<PointsToGraph::DeltaNode>>;
  using ImportNodeMap =
      std::unordered_map<const jlm::rvsdg::argument *, std::unique_ptr<PointsToGraph::ImportNode>>;
  using LambdaNodeMap =
      std::unordered_map<const lambda::node *, std::unique_ptr<PointsToGraph::LambdaNode>>;
  using MallocNodeMap =
      std::unordered_map<const jlm::rvsdg::node *, std::unique_ptr<PointsToGraph::MallocNode>>;
  using RegisterSetNodeMap =
      std::unordered_map<const rvsdg::output *, PointsToGraph::RegisterSetNode *>;
  using RegisterSetNodeVector = std::vector<std::unique_ptr<PointsToGraph::RegisterSetNode>>;

  template<class DataType, class IteratorType>
  struct IteratorToPointerFunctor
  {
    DataType *
    operator()(const IteratorType & it) const
    {
      return it->second.get();
    }
  };

  using AllocaNodeIterator = NodeIterator<
      AllocaNode,
      AllocaNodeMap::iterator,
      IteratorToPointerFunctor<AllocaNode, AllocaNodeMap::iterator>>;
  using AllocaNodeConstIterator = NodeConstIterator<
      AllocaNode,
      AllocaNodeMap::const_iterator,
      IteratorToPointerFunctor<AllocaNode, AllocaNodeMap::const_iterator>>;
  using AllocaNodeRange = jlm::util::iterator_range<AllocaNodeIterator>;
  using AllocaNodeConstRange = jlm::util::iterator_range<AllocaNodeConstIterator>;

  using DeltaNodeIterator = NodeIterator<
      DeltaNode,
      DeltaNodeMap::iterator,
      IteratorToPointerFunctor<DeltaNode, DeltaNodeMap::iterator>>;
  using DeltaNodeConstIterator = NodeConstIterator<
      DeltaNode,
      DeltaNodeMap::const_iterator,
      IteratorToPointerFunctor<DeltaNode, DeltaNodeMap::const_iterator>>;
  using DeltaNodeRange = jlm::util::iterator_range<DeltaNodeIterator>;
  using DeltaNodeConstRange = jlm::util::iterator_range<DeltaNodeConstIterator>;

  using ImportNodeIterator = NodeIterator<
      ImportNode,
      ImportNodeMap::iterator,
      IteratorToPointerFunctor<ImportNode, ImportNodeMap::iterator>>;
  using ImportNodeConstIterator = NodeConstIterator<
      ImportNode,
      ImportNodeMap::const_iterator,
      IteratorToPointerFunctor<ImportNode, ImportNodeMap::const_iterator>>;
  using ImportNodeRange = jlm::util::iterator_range<ImportNodeIterator>;
  using ImportNodeConstRange = jlm::util::iterator_range<ImportNodeConstIterator>;

  using LambdaNodeIterator = NodeIterator<
      LambdaNode,
      LambdaNodeMap::iterator,
      IteratorToPointerFunctor<LambdaNode, LambdaNodeMap::iterator>>;
  using LambdaNodeConstIterator = NodeConstIterator<
      LambdaNode,
      LambdaNodeMap::const_iterator,
      IteratorToPointerFunctor<LambdaNode, LambdaNodeMap::const_iterator>>;
  using LambdaNodeRange = jlm::util::iterator_range<LambdaNodeIterator>;
  using LambdaNodeConstRange = jlm::util::iterator_range<LambdaNodeConstIterator>;

  using MallocNodeIterator = NodeIterator<
      MallocNode,
      MallocNodeMap::iterator,
      IteratorToPointerFunctor<MallocNode, MallocNodeMap::iterator>>;
  using MallocNodeConstIterator = NodeConstIterator<
      MallocNode,
      MallocNodeMap::const_iterator,
      IteratorToPointerFunctor<MallocNode, MallocNodeMap::const_iterator>>;
  using MallocNodeRange = jlm::util::iterator_range<MallocNodeIterator>;
  using MallocNodeConstRange = jlm::util::iterator_range<MallocNodeConstIterator>;

  template<class IteratorType>
  struct RegisterSetNodeIteratorToPointerFunctor
  {
    RegisterSetNode *
    operator()(const IteratorType & it) const
    {
      return it->get();
    }
  };

  using RegisterSetNodeIterator = NodeIterator<
      RegisterSetNode,
      RegisterSetNodeVector::iterator,
      RegisterSetNodeIteratorToPointerFunctor<RegisterSetNodeVector::iterator>>;
  using RegisterSetNodeConstIterator = NodeConstIterator<
      RegisterSetNode,
      RegisterSetNodeVector::const_iterator,
      RegisterSetNodeIteratorToPointerFunctor<RegisterSetNodeVector::const_iterator>>;
  using RegisterSetNodeRange = util::iterator_range<RegisterSetNodeIterator>;
  using RegisterSetNodeConstRange = util::iterator_range<RegisterSetNodeConstIterator>;

private:
  PointsToGraph();

public:
  PointsToGraph(const PointsToGraph &) = delete;

  PointsToGraph(PointsToGraph &&) = delete;

  PointsToGraph &
  operator=(const PointsToGraph &) = delete;

  PointsToGraph &
  operator=(PointsToGraph &&) = delete;

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

  RegisterSetNodeRange
  RegisterSetNodes();

  RegisterSetNodeConstRange
  RegisterSetNodes() const;

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

  [[nodiscard]] size_t
  NumRegisterSetNodes() const noexcept
  {
    return RegisterSetNodes_.size();
  }

  size_t
  NumMemoryNodes() const noexcept
  {
    return NumAllocaNodes() + NumDeltaNodes() + NumImportNodes() + NumLambdaNodes()
         + NumMallocNodes() + 1; // External memory node
  }

  size_t
  NumNodes() const noexcept
  {
    return NumMemoryNodes() + NumRegisterSetNodes();
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
  GetAllocaNode(const jlm::rvsdg::node & node) const
  {
    auto it = AllocaNodes_.find(&node);
    if (it == AllocaNodes_.end())
      throw jlm::util::error("Cannot find alloca node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::DeltaNode &
  GetDeltaNode(const delta::node & node) const
  {
    auto it = DeltaNodes_.find(&node);
    if (it == DeltaNodes_.end())
      throw jlm::util::error("Cannot find delta node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::ImportNode &
  GetImportNode(const jlm::rvsdg::argument & argument) const
  {
    auto it = ImportNodes_.find(&argument);
    if (it == ImportNodes_.end())
      throw jlm::util::error("Cannot find import node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::LambdaNode &
  GetLambdaNode(const lambda::node & node) const
  {
    auto it = LambdaNodes_.find(&node);
    if (it == LambdaNodes_.end())
      throw jlm::util::error("Cannot find lambda node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::MallocNode &
  GetMallocNode(const jlm::rvsdg::node & node) const
  {
    auto it = MallocNodes_.find(&node);
    if (it == MallocNodes_.end())
      throw jlm::util::error("Cannot find malloc node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::RegisterSetNode &
  GetRegisterSetNode(const rvsdg::output & output) const
  {
    auto it = RegisterSetNodeMap_.find(&output);
    if (it == RegisterSetNodeMap_.end())
      throw util::error("Cannot find register set node in points-to graph.");

    return *it->second;
  }

  /**
   * Returns all memory nodes that are marked as escaped from the module.
   *
   * @return A set with all escaped memory nodes.
   *
   * @see PointsToGraph::MemoryNode::MarkAsModuleEscaping()
   */
  const jlm::util::HashSet<const PointsToGraph::MemoryNode *> &
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

  PointsToGraph::RegisterSetNode &
  AddRegisterSetNode(std::unique_ptr<PointsToGraph::RegisterSetNode> node);

  PointsToGraph::ImportNode &
  AddImportNode(std::unique_ptr<PointsToGraph::ImportNode> node);

  /**
   * Creates a GraphViz description of the given \p pointsToGraph,
   * including the names given to rvsdg::outputs by the \p outputMap,
   * for all RegisterNodes that correspond to names rvsdg::outputs.
   * @param pointsToGraph the graph to be drawn as a dot-file.
   * @param outputMap the mapping from rvsdg::output* to a unique name.
   * @return the text content of the resulting dot-file.
   */
  static std::string
  ToDot(
      const PointsToGraph & pointsToGraph,
      const std::unordered_map<const rvsdg::output *, std::string> & outputMap);

  /**
   * @brief Creates a GraphViz description of the given \p pointsToGraph.
   * @param pointsToGraph the graph to be drawn as a dot-file.
   * @return the text content of the resulting dot-file.
   */
  static std::string
  ToDot(const PointsToGraph & pointsToGraph)
  {
    const std::unordered_map<const rvsdg::output *, std::string> outputMap;
    return ToDot(pointsToGraph, outputMap);
  }

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
  jlm::util::HashSet<const PointsToGraph::MemoryNode *> EscapedMemoryNodes_;

  AllocaNodeMap AllocaNodes_;
  DeltaNodeMap DeltaNodes_;
  ImportNodeMap ImportNodes_;
  LambdaNodeMap LambdaNodes_;
  MallocNodeMap MallocNodes_;

  RegisterSetNodeMap RegisterSetNodeMap_;
  RegisterSetNodeVector RegisterSetNodes_;

  std::unique_ptr<PointsToGraph::UnknownMemoryNode> UnknownMemoryNode_;
  std::unique_ptr<ExternalMemoryNode> ExternalMemoryNode_;
};

/** \brief PointsTo graph node
 *
 */
class PointsToGraph::Node
{
  template<class NODETYPE>
  class ConstIterator;
  template<class NODETYPE>
  class Iterator;

  using SourceIterator = Iterator<PointsToGraph::Node>;
  using SourceConstIterator = ConstIterator<PointsToGraph::Node>;

  using TargetIterator = Iterator<PointsToGraph::MemoryNode>;
  using TargetConstIterator = ConstIterator<PointsToGraph::MemoryNode>;

  using SourceRange = jlm::util::iterator_range<SourceIterator>;
  using SourceConstRange = jlm::util::iterator_range<SourceConstIterator>;

  using TargetRange = jlm::util::iterator_range<TargetIterator>;
  using TargetConstRange = jlm::util::iterator_range<TargetConstIterator>;

public:
  virtual ~Node() noexcept;

  explicit Node(PointsToGraph & pointsToGraph)
      : PointsToGraph_(&pointsToGraph)
  {}

  Node(const Node &) = delete;

  Node(Node &&) = delete;

  Node &
  operator=(const Node &) = delete;

  Node &
  operator=(Node &&) = delete;

  TargetRange
  Targets();

  TargetConstRange
  Targets() const;

  SourceRange
  Sources();

  SourceConstRange
  Sources() const;

  PointsToGraph &
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

  template<class T>
  static bool
  Is(const Node & node)
  {
    static_assert(
        std::is_base_of<Node, T>::value,
        "Template parameter T must be derived from PointsToGraph::Node.");

    return dynamic_cast<const T *>(&node) != nullptr;
  }

private:
  PointsToGraph * PointsToGraph_;
  std::unordered_set<PointsToGraph::MemoryNode *> Targets_;
  std::unordered_set<PointsToGraph::Node *> Sources_;
};

/**
 * Represents a set of registers from the RVSDG that all point to the same
 * PointsToGraph::MemoryNode%s.
 */
class PointsToGraph::RegisterSetNode final : public PointsToGraph::Node
{
public:
  ~RegisterSetNode() noexcept override;

private:
  RegisterSetNode(PointsToGraph & pointsToGraph, util::HashSet<const rvsdg::output *> outputs)
      : Node(pointsToGraph),
        Outputs_(std::move(outputs))
  {}

public:
  const util::HashSet<const rvsdg::output *> &
  GetOutputs() const noexcept
  {
    return Outputs_;
  }

  std::string
  DebugString() const override;

  static std::string
  ToString(const rvsdg::output & output);

  static PointsToGraph::RegisterSetNode &
  Create(PointsToGraph & pointsToGraph, util::HashSet<const rvsdg::output *> outputs)
  {
    auto node = std::unique_ptr<PointsToGraph::RegisterSetNode>(
        new RegisterSetNode(pointsToGraph, std::move(outputs)));
    return pointsToGraph.AddRegisterSetNode(std::move(node));
  }

private:
  const util::HashSet<const rvsdg::output *> Outputs_;
};

/** \brief PointsTo graph memory node
 *
 */
class PointsToGraph::MemoryNode : public PointsToGraph::Node
{
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
  explicit MemoryNode(PointsToGraph & pointsToGraph)
      : Node(pointsToGraph)
  {}
};

/** \brief PointsTo graph alloca node
 *
 */
class PointsToGraph::AllocaNode final : public PointsToGraph::MemoryNode
{
public:
  ~AllocaNode() noexcept override;

private:
  AllocaNode(PointsToGraph & pointsToGraph, const jlm::rvsdg::node & allocaNode)
      : MemoryNode(pointsToGraph),
        AllocaNode_(&allocaNode)
  {
    JLM_ASSERT(is<alloca_op>(&allocaNode));
  }

public:
  const jlm::rvsdg::node &
  GetAllocaNode() const noexcept
  {
    return *AllocaNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::AllocaNode &
  Create(PointsToGraph & pointsToGraph, const jlm::rvsdg::node & node)
  {
    auto n = std::unique_ptr<PointsToGraph::AllocaNode>(new AllocaNode(pointsToGraph, node));
    return pointsToGraph.AddAllocaNode(std::move(n));
  }

private:
  const jlm::rvsdg::node * AllocaNode_;
};

/** \brief PointsTo graph delta node
 *
 */
class PointsToGraph::DeltaNode final : public PointsToGraph::MemoryNode
{
public:
  ~DeltaNode() noexcept override;

private:
  DeltaNode(PointsToGraph & pointsToGraph, const delta::node & deltaNode)
      : MemoryNode(pointsToGraph),
        DeltaNode_(&deltaNode)
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
  Create(PointsToGraph & pointsToGraph, const delta::node & deltaNode)
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
class PointsToGraph::MallocNode final : public PointsToGraph::MemoryNode
{
public:
  ~MallocNode() noexcept override;

private:
  MallocNode(PointsToGraph & pointsToGraph, const jlm::rvsdg::node & mallocNode)
      : MemoryNode(pointsToGraph),
        MallocNode_(&mallocNode)
  {
    JLM_ASSERT(is<malloc_op>(&mallocNode));
  }

public:
  const jlm::rvsdg::node &
  GetMallocNode() const noexcept
  {
    return *MallocNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::MallocNode &
  Create(PointsToGraph & pointsToGraph, const jlm::rvsdg::node & node)
  {
    auto n = std::unique_ptr<PointsToGraph::MallocNode>(new MallocNode(pointsToGraph, node));
    return pointsToGraph.AddMallocNode(std::move(n));
  }

private:
  const jlm::rvsdg::node * MallocNode_;
};

/** \brief PointsTo graph malloc node
 *
 */
class PointsToGraph::LambdaNode final : public PointsToGraph::MemoryNode
{
public:
  ~LambdaNode() noexcept override;

private:
  LambdaNode(PointsToGraph & pointsToGraph, const lambda::node & lambdaNode)
      : MemoryNode(pointsToGraph),
        LambdaNode_(&lambdaNode)
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
  Create(PointsToGraph & pointsToGraph, const lambda::node & lambdaNode)
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
class PointsToGraph::ImportNode final : public PointsToGraph::MemoryNode
{
public:
  ~ImportNode() noexcept override;

private:
  ImportNode(PointsToGraph & pointsToGraph, const jlm::rvsdg::argument & argument)
      : MemoryNode(pointsToGraph),
        Argument_(&argument)
  {
    JLM_ASSERT(dynamic_cast<const impport *>(&argument.port()));
  }

public:
  const jlm::rvsdg::argument &
  GetArgument() const noexcept
  {
    return *Argument_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::ImportNode &
  Create(PointsToGraph & pointsToGraph, const jlm::rvsdg::argument & argument)
  {
    auto n = std::unique_ptr<PointsToGraph::ImportNode>(new ImportNode(pointsToGraph, argument));
    return pointsToGraph.AddImportNode(std::move(n));
  }

private:
  const jlm::rvsdg::argument * Argument_;
};

/** \brief PointsTo graph unknown node
 *
 */
class PointsToGraph::UnknownMemoryNode final : public PointsToGraph::MemoryNode
{
  friend PointsToGraph;

public:
  ~UnknownMemoryNode() noexcept override;

private:
  explicit UnknownMemoryNode(PointsToGraph & pointsToGraph)
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
class PointsToGraph::ExternalMemoryNode final : public PointsToGraph::MemoryNode
{
  friend PointsToGraph;

public:
  ~ExternalMemoryNode() noexcept override;

private:
  explicit ExternalMemoryNode(PointsToGraph & pointsToGraph)
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
template<class DATATYPE, class ITERATORTYPE, typename IteratorToPointer>
class PointsToGraph::NodeIterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = DATATYPE *;
  using difference_type = std::ptrdiff_t;
  using pointer = DATATYPE **;
  using reference = DATATYPE *&;

private:
  friend PointsToGraph;

  explicit NodeIterator(const ITERATORTYPE & it)
      : it_(it)
  {}

public:
  [[nodiscard]] DATATYPE *
  Node() const noexcept
  {
    return IteratorToPointer_(it_);
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
  IteratorToPointer IteratorToPointer_;
};

/** \brief Points-to graph node const iterator
 */
template<class DATATYPE, class ITERATORTYPE, typename IteratorToPointer>
class PointsToGraph::NodeConstIterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = const DATATYPE *;
  using difference_type = std::ptrdiff_t;
  using pointer = const DATATYPE **;
  using reference = const DATATYPE *&;

private:
  friend PointsToGraph;

  explicit NodeConstIterator(const ITERATORTYPE & it)
      : it_(it)
  {}

public:
  [[nodiscard]] const DATATYPE *
  Node() const noexcept
  {
    return IteratorToPointer_(it_);
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
  IteratorToPointer IteratorToPointer_;
};

/** \brief Points-to graph edge iterator
 */
template<class NODETYPE>
class PointsToGraph::Node::Iterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = NODETYPE *;
  using difference_type = std::ptrdiff_t;
  using pointer = NODETYPE **;
  using reference = NODETYPE *&;

private:
  friend PointsToGraph::Node;

  explicit Iterator(const typename std::unordered_set<NODETYPE *>::iterator & it)
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
  typename std::unordered_set<NODETYPE *>::iterator It_;
};

/** \brief Points-to graph edge const iterator
 */
template<class NODETYPE>
class PointsToGraph::Node::ConstIterator final
{
public:
  using itearator_category = std::forward_iterator_tag;
  using value_type = const NODETYPE *;
  using difference_type = std::ptrdiff_t;
  using pointer = const NODETYPE **;
  using reference = const NODETYPE *&;

private:
  friend PointsToGraph;

  explicit ConstIterator(const typename std::unordered_set<NODETYPE *>::const_iterator & it)
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
  typename std::unordered_set<NODETYPE *>::const_iterator It_;
};

}
}

#endif
