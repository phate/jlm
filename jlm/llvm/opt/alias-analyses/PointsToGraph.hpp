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
  class RegisterNode;
  class UnknownMemoryNode;
  class ExternalMemoryNode;

  using AllocaNodeMap = std::unordered_map<const rvsdg::Node *, std::unique_ptr<AllocaNode>>;
  using DeltaNodeMap =
      std::unordered_map<const delta::node *, std::unique_ptr<PointsToGraph::DeltaNode>>;
  using ImportNodeMap =
      std::unordered_map<const rvsdg::RegionArgument *, std::unique_ptr<PointsToGraph::ImportNode>>;
  using LambdaNodeMap =
      std::unordered_map<const rvsdg::LambdaNode *, std::unique_ptr<PointsToGraph::LambdaNode>>;
  using MallocNodeMap = std::unordered_map<const rvsdg::Node *, std::unique_ptr<MallocNode>>;
  using RegisterNodeMap = std::unordered_map<const rvsdg::Output *, PointsToGraph::RegisterNode *>;
  using RegisterNodeVector = std::vector<std::unique_ptr<PointsToGraph::RegisterNode>>;

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
  using AllocaNodeRange = util::IteratorRange<AllocaNodeIterator>;
  using AllocaNodeConstRange = util::IteratorRange<AllocaNodeConstIterator>;

  using DeltaNodeIterator = NodeIterator<
      DeltaNode,
      DeltaNodeMap::iterator,
      IteratorToPointerFunctor<DeltaNode, DeltaNodeMap::iterator>>;
  using DeltaNodeConstIterator = NodeConstIterator<
      DeltaNode,
      DeltaNodeMap::const_iterator,
      IteratorToPointerFunctor<DeltaNode, DeltaNodeMap::const_iterator>>;
  using DeltaNodeRange = util::IteratorRange<DeltaNodeIterator>;
  using DeltaNodeConstRange = util::IteratorRange<DeltaNodeConstIterator>;

  using ImportNodeIterator = NodeIterator<
      ImportNode,
      ImportNodeMap::iterator,
      IteratorToPointerFunctor<ImportNode, ImportNodeMap::iterator>>;
  using ImportNodeConstIterator = NodeConstIterator<
      ImportNode,
      ImportNodeMap::const_iterator,
      IteratorToPointerFunctor<ImportNode, ImportNodeMap::const_iterator>>;
  using ImportNodeRange = jlm::util::IteratorRange<ImportNodeIterator>;
  using ImportNodeConstRange = jlm::util::IteratorRange<ImportNodeConstIterator>;

  using LambdaNodeIterator = NodeIterator<
      LambdaNode,
      LambdaNodeMap::iterator,
      IteratorToPointerFunctor<LambdaNode, LambdaNodeMap::iterator>>;
  using LambdaNodeConstIterator = NodeConstIterator<
      LambdaNode,
      LambdaNodeMap::const_iterator,
      IteratorToPointerFunctor<LambdaNode, LambdaNodeMap::const_iterator>>;
  using LambdaNodeRange = util::IteratorRange<LambdaNodeIterator>;
  using LambdaNodeConstRange = util::IteratorRange<LambdaNodeConstIterator>;

  using MallocNodeIterator = NodeIterator<
      MallocNode,
      MallocNodeMap::iterator,
      IteratorToPointerFunctor<MallocNode, MallocNodeMap::iterator>>;
  using MallocNodeConstIterator = NodeConstIterator<
      MallocNode,
      MallocNodeMap::const_iterator,
      IteratorToPointerFunctor<MallocNode, MallocNodeMap::const_iterator>>;
  using MallocNodeRange = util::IteratorRange<MallocNodeIterator>;
  using MallocNodeConstRange = util::IteratorRange<MallocNodeConstIterator>;

  template<class IteratorType>
  struct RegisterNodeIteratorToPointerFunctor
  {
    RegisterNode *
    operator()(const IteratorType & it) const
    {
      return it->get();
    }
  };

  using RegisterNodeIterator = NodeIterator<
      RegisterNode,
      RegisterNodeVector::iterator,
      RegisterNodeIteratorToPointerFunctor<RegisterNodeVector::iterator>>;
  using RegisterNodeConstIterator = NodeConstIterator<
      RegisterNode,
      RegisterNodeVector::const_iterator,
      RegisterNodeIteratorToPointerFunctor<RegisterNodeVector::const_iterator>>;
  using RegisterNodeRange = util::IteratorRange<RegisterNodeIterator>;
  using RegisterNodeConstRange = util::IteratorRange<RegisterNodeConstIterator>;

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

  [[nodiscard]] size_t
  NumRegisterNodes() const noexcept
  {
    return RegisterNodes_.size();
  }

  /**
   * @return the total number of registers that are represented by some RegisterNode
   */
  [[nodiscard]] size_t
  NumMappedRegisters() const noexcept
  {
    return RegisterNodeMap_.size();
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
  GetAllocaNode(const rvsdg::Node & node) const
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
  GetImportNode(const rvsdg::RegionArgument & argument) const
  {
    auto it = ImportNodes_.find(&argument);
    if (it == ImportNodes_.end())
      throw jlm::util::error("Cannot find import node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::LambdaNode &
  GetLambdaNode(const rvsdg::LambdaNode & node) const
  {
    auto it = LambdaNodes_.find(&node);
    if (it == LambdaNodes_.end())
      throw jlm::util::error("Cannot find lambda node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::MallocNode &
  GetMallocNode(const rvsdg::Node & node) const
  {
    auto it = MallocNodes_.find(&node);
    if (it == MallocNodes_.end())
      throw jlm::util::error("Cannot find malloc node in points-to graph.");

    return *it->second;
  }

  const PointsToGraph::RegisterNode &
  GetRegisterNode(const rvsdg::Output & output) const
  {
    auto it = RegisterNodeMap_.find(&output);
    if (it == RegisterNodeMap_.end())
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

  PointsToGraph::RegisterNode &
  AddRegisterNode(std::unique_ptr<PointsToGraph::RegisterNode> node);

  PointsToGraph::ImportNode &
  AddImportNode(std::unique_ptr<PointsToGraph::ImportNode> node);

  /**
   * Gets the total number of edges in the PointsToGraph.
   *
   * In addition, RegisterNodes can represent multiple registers,
   * in which case each outgoing edge represents multiple points-to relations.
   * The total number of points-to relations is also returned.
   *
   * @return a pair (number of edges, number of points-to relations)
   */
  [[nodiscard]] std::pair<size_t, size_t>
  NumEdges() const noexcept;

  /**
   * Checks if this PointsToGraph is a supergraph of \p subgraph.
   * Every node and every edge in the subgraph needs to have corresponding nodes and edges
   * present in this graph, defined by nodes representing the same registers and memory objects.
   * All nodes marked as escaping in the subgraph must also be marked as escaping in this graph.
   * @param subgraph the graph to compare against
   * @return true if this graph is a supergraph of the given subgraph, false otherwise
   */
  [[nodiscard]] bool
  IsSupergraphOf(const PointsToGraph & subgraph) const;

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
      const std::unordered_map<const rvsdg::Output *, std::string> & outputMap);

  /**
   * @brief Creates a GraphViz description of the given \p pointsToGraph.
   * @param pointsToGraph the graph to be drawn as a dot-file.
   * @return the text content of the resulting dot-file.
   */
  static std::string
  ToDot(const PointsToGraph & pointsToGraph)
  {
    const std::unordered_map<const rvsdg::Output *, std::string> outputMap;
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

  RegisterNodeMap RegisterNodeMap_;
  RegisterNodeVector RegisterNodes_;

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

  using SourceRange = util::IteratorRange<SourceIterator>;
  using SourceConstRange = util::IteratorRange<SourceConstIterator>;

  using TargetRange = util::IteratorRange<TargetIterator>;
  using TargetConstRange = util::IteratorRange<TargetConstIterator>;

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

  [[nodiscard]] bool
  HasTarget(const PointsToGraph::MemoryNode & target) const;

  SourceRange
  Sources();

  SourceConstRange
  Sources() const;

  [[nodiscard]] bool
  HasSource(const PointsToGraph::Node & source) const;

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
class PointsToGraph::RegisterNode final : public PointsToGraph::Node
{
public:
  ~RegisterNode() noexcept override;

private:
  RegisterNode(PointsToGraph & pointsToGraph, util::HashSet<const rvsdg::Output *> outputs)
      : Node(pointsToGraph),
        Outputs_(std::move(outputs))
  {}

public:
  const util::HashSet<const rvsdg::Output *> &
  GetOutputs() const noexcept
  {
    return Outputs_;
  }

  std::string
  DebugString() const override;

  static std::string
  ToString(const rvsdg::Output & output);

  static PointsToGraph::RegisterNode &
  Create(PointsToGraph & pointsToGraph, util::HashSet<const rvsdg::Output *> outputs)
  {
    auto node = std::unique_ptr<PointsToGraph::RegisterNode>(
        new RegisterNode(pointsToGraph, std::move(outputs)));
    return pointsToGraph.AddRegisterNode(std::move(node));
  }

private:
  const util::HashSet<const rvsdg::Output *> Outputs_;
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

  /**
   * @return true if this memory node is marked as escaping the module.
   */
  bool
  IsModuleEscaping() const
  {
    return Graph().GetEscapedMemoryNodes().Contains(this);
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
  AllocaNode(PointsToGraph & pointsToGraph, const rvsdg::Node & allocaNode)
      : MemoryNode(pointsToGraph),
        AllocaNode_(&allocaNode)
  {
    JLM_ASSERT(is<alloca_op>(&allocaNode));
  }

public:
  const rvsdg::Node &
  GetAllocaNode() const noexcept
  {
    return *AllocaNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::AllocaNode &
  Create(PointsToGraph & pointsToGraph, const rvsdg::Node & node)
  {
    auto n = std::unique_ptr<PointsToGraph::AllocaNode>(new AllocaNode(pointsToGraph, node));
    return pointsToGraph.AddAllocaNode(std::move(n));
  }

private:
  const rvsdg::Node * AllocaNode_;
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
  {}

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
  MallocNode(PointsToGraph & pointsToGraph, const rvsdg::Node & mallocNode)
      : MemoryNode(pointsToGraph),
        MallocNode_(&mallocNode)
  {
    JLM_ASSERT(is<malloc_op>(&mallocNode));
  }

public:
  const rvsdg::Node &
  GetMallocNode() const noexcept
  {
    return *MallocNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::MallocNode &
  Create(PointsToGraph & pointsToGraph, const rvsdg::Node & node)
  {
    auto n = std::unique_ptr<PointsToGraph::MallocNode>(new MallocNode(pointsToGraph, node));
    return pointsToGraph.AddMallocNode(std::move(n));
  }

private:
  const rvsdg::Node * MallocNode_;
};

/** \brief PointsTo graph malloc node
 *
 */
class PointsToGraph::LambdaNode final : public PointsToGraph::MemoryNode
{
public:
  ~LambdaNode() noexcept override;

private:
  LambdaNode(PointsToGraph & pointsToGraph, const rvsdg::LambdaNode & lambdaNode)
      : MemoryNode(pointsToGraph),
        LambdaNode_(&lambdaNode)
  {
    JLM_ASSERT(dynamic_cast<const llvm::LlvmLambdaOperation *>(&lambdaNode.GetOperation()));
  }

public:
  const rvsdg::LambdaNode &
  GetLambdaNode() const noexcept
  {
    return *LambdaNode_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::LambdaNode &
  Create(PointsToGraph & pointsToGraph, const rvsdg::LambdaNode & lambdaNode)
  {
    auto n = std::unique_ptr<PointsToGraph::LambdaNode>(new LambdaNode(pointsToGraph, lambdaNode));
    return pointsToGraph.AddLambdaNode(std::move(n));
  }

private:
  const rvsdg::LambdaNode * LambdaNode_;
};

/** \brief PointsTo graph import node
 *
 */
class PointsToGraph::ImportNode final : public PointsToGraph::MemoryNode
{
public:
  ~ImportNode() noexcept override;

private:
  ImportNode(PointsToGraph & pointsToGraph, const GraphImport & graphImport)
      : MemoryNode(pointsToGraph),
        GraphImport_(&graphImport)
  {}

public:
  const GraphImport &
  GetArgument() const noexcept
  {
    return *GraphImport_;
  }

  std::string
  DebugString() const override;

  static PointsToGraph::ImportNode &
  Create(PointsToGraph & pointsToGraph, const GraphImport & argument)
  {
    auto n = std::unique_ptr<PointsToGraph::ImportNode>(new ImportNode(pointsToGraph, argument));
    return pointsToGraph.AddImportNode(std::move(n));
  }

private:
  const GraphImport * GraphImport_;
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
  friend class PointsToGraph::Node;

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
