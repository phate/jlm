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
#include <jlm/util/Math.hpp>

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
 */
class PointsToGraph final
{
  using NodeIndex = uint32_t;

  enum class NodeKind : uint8_t
  {
    RegisterNode = 0,
    AllocaNode,
    DeltaNode,
    ImportNode,
    LambdaNode,
    MallocNode,
    COUNT
  };

  struct NodeData
  {
    NodeKind kind : util::BitWidthOfEnum(NodeKind::COUNT);

    // When set, the node is available from other modules in the program
    uint8_t isExternallyAvailable : 1;

    // When set, the PointsToGraph Node implicitly targets all externally available nodes
    uint8_t isTargetingAllExternallyAvailable : 1;
  };

  using AllocaNodeMap = std::unordered_map<const rvsdg::Node *, NodeIndex>;
  using DeltaNodeMap = std::unordered_map<const rvsdg::DeltaNode *, NodeIndex>;
  using ImportNodeMap = std::unordered_map<const rvsdg::RegionArgument *, NodeIndex>;
  using LambdaNodeMap = std::unordered_map<const rvsdg::LambdaNode *, NodeIndex>;
  using MallocNodeMap = std::unordered_map<const rvsdg::Node *, NodeIndex>;
  using RegisterNodeMap = std::unordered_map<const rvsdg::Output *, NodeIndex>;

  using AllocaNodeIterator =
      util::MapValuePtrIterator<const NodeIndex, AllocaNodeMap::const_iterator>;
  using AllocaNodeRange = util::IteratorRange<AllocaNodeIterator>;

  using DeltaNodeIterator =
      util::MapValuePtrIterator<const NodeIndex, DeltaNodeMap::const_iterator>;
  using DeltaNodeRange = util::IteratorRange<DeltaNodeIterator>;

  using ImportNodeIterator =
      util::MapValuePtrIterator<const NodeIndex, ImportNodeMap::const_iterator>;
  using ImportNodeRange = util::IteratorRange<ImportNodeIterator>;

  using LambdaNodeIterator =
      util::MapValuePtrIterator<const NodeIndex, LambdaNodeMap::const_iterator>;
  using LambdaNodeRange = util::IteratorRange<LambdaNodeIterator>;

  using MallocNodeIterator =
      util::MapValuePtrIterator<const NodeIndex, MallocNodeMap::const_iterator>;
  using MallocNodeRange = util::IteratorRange<MallocNodeIterator>;

  using RegisterNodeIterator = std::vector<NodeIndex>::const_iterator;
  using RegisterNodeRange = util::IteratorRange<RegisterNodeIterator>;

  using ExternallyAvailableIterator = std::vector<NodeIndex>::const_iterator;
  using ExternallyAvailableRange = util::IteratorRange<ExternallyAvailableIterator>;

  PointsToGraph();

public:
  PointsToGraph(const PointsToGraph &) = delete;

  PointsToGraph(PointsToGraph &&) = delete;

  PointsToGraph &
  operator=(const PointsToGraph &) = delete;

  PointsToGraph &
  operator=(PointsToGraph &&) = delete;

  AllocaNodeRange
  allocaNodes() const noexcept;

  DeltaNodeRange
  deltaNodes() const noexcept;

  ImportNodeRange
  importNodes() const noexcept;

  LambdaNodeRange
  lambdaNodes() const noexcept;

  MallocNodeRange
  mallocNodes() const noexcept;

  RegisterNodeRange
  registerNodes() const noexcept;

  size_t
  numAllocaNodes() const noexcept
  {
    return allocaMap_.size();
  }

  size_t
  numDeltaNodes() const noexcept
  {
    return deltaMap_.size();
  }

  size_t
  numImportNodes() const noexcept
  {
    return importMap_.size();
  }

  size_t
  numLambdaNodes() const noexcept
  {
    return lambdaMap_.size();
  }

  size_t
  numMallocNodes() const noexcept
  {
    return mallocMap_.size();
  }

  [[nodiscard]] size_t
  numRegisterNodes() const noexcept
  {
    return registerNodes_.size();
  }

  /**
   * @return the total number of registers that are represented by some RegisterNode
   */
  [[nodiscard]] size_t
  numMappedRegisters() const noexcept
  {
    return registerMap_.size();
  }

  size_t
  numMemoryNodes() const noexcept
  {
    return numAllocaNodes() + numDeltaNodes() + numImportNodes() + numLambdaNodes()
         + numMallocNodes();
  }

  size_t
  numNodes() const noexcept
  {
    JLM_ASSERT(nodeData_.size() == nodeTargets_.size());
    JLM_ASSERT(nodeData_.size() == nodeObjects_.size());
    return nodeData_.size();
  }

  NodeIndex
  getAllocaNode(const rvsdg::Node & node) const
  {
    if (const auto it = allocaMap_.find(&node); it != allocaMap_.end())
      return it->second;

    throw util::Error("Cannot find alloca node in points-to graph.");
  }

  NodeIndex
  getDeltaNode(const rvsdg::DeltaNode & node) const
  {
    if (const auto it = deltaMap_.find(&node); it != deltaMap_.end())
      return it->second;

    throw util::Error("Cannot find delta node in points-to graph.");
  }

  NodeIndex
  getImportNode(const rvsdg::RegionArgument & argument) const
  {
    if (const auto it = importMap_.find(&argument); it != importMap_.end())
      return it->second;

    throw util::Error("Cannot find import in points-to graph.");
  }

  NodeIndex
  getLambdaNode(const rvsdg::LambdaNode & node) const
  {
    if (const auto it = lambdaMap_.find(&node); it != lambdaMap_.end())
      return it->second;

    throw util::Error("Cannot find lambda node in points-to graph.");
  }

  NodeIndex
  getMallocNode(const rvsdg::Node & node) const
  {
    if (const auto it = mallocMap_.find(&node); it != mallocMap_.end())
      return it->second;

    throw util::Error("Cannot find malloc node in points-to graph.");
  }

  NodeIndex
  getRegisterNode(const rvsdg::Output & output) const
  {
    if (const auto it = registerMap_.find(&output); it != registerMap_.end())
      return it->second;

    throw util::Error("Cannot find node mapped to register in points-to graph.");
  }

  [[nodiscard]] const util::HashSet<NodeIndex> &
  getTargets(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeTargets_.size());
    return nodeTargets_[index];
  }

  [[nodiscard]] bool
  isExternallyAvailable(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeData_.size());
    return nodeData_[index].isExternallyAvailable;
  }

  [[nodiscard]] bool
  isTargetingAllExternallyAvailable(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeData_.size());
    return nodeData_[index].isTargetingAllExternallyAvailable;
  }

  [[nodiscard]] NodeKind
  getKind(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeData_.size());
    return nodeData_[index].kind;
  }

  /**
   * Returns all memory nodes that are marked as being externally available
   *
   * @return A set with all escaped memory nodes.
   *
   * @see PointsToGraph::MemoryNode::MarkAsModuleEscaping()
   */
  const std::vector<NodeIndex> &
  getExternallyAvailableNodes() const noexcept
  {
    return externallyAvailableNodes_;
  }

  const rvsdg::Node &
  getAllocaNodeObject(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    JLM_ASSERT(nodeData_[index].kind == NodeKind::AllocaNode);
    return *static_cast<const rvsdg::Node *>(nodeObjects_[index]);
  }

  const rvsdg::DeltaNode &
  getDeltaNodeObject(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    JLM_ASSERT(nodeData_[index].kind == NodeKind::DeltaNode);
    return *static_cast<const rvsdg::DeltaNode *>(nodeObjects_[index]);
  }

  const rvsdg::RegionArgument &
  getImportNodeObject(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    JLM_ASSERT(nodeData_[index].kind == NodeKind::ImportNode);
    return *static_cast<const rvsdg::RegionArgument *>(nodeObjects_[index]);
  }

  const rvsdg::LambdaNode &
  getLambdaNodeObject(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    JLM_ASSERT(nodeData_[index].kind == NodeKind::LambdaNode);
    return *static_cast<const rvsdg::LambdaNode *>(nodeObjects_[index]);
  }

  const rvsdg::Node &
  getMallocNodeObject(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    JLM_ASSERT(nodeData_[index].kind == NodeKind::MallocNode);
    return *static_cast<const rvsdg::Node *>(nodeObjects_[index]);
  }

  NodeIndex
  addAllocaNode(const rvsdg::Node & allocaNode, bool externallyAvailable);

  NodeIndex
  addDeltaNode(const rvsdg::DeltaNode & deltaNode, bool externallyAvailable);

  NodeIndex
  addImportNode(const rvsdg::RegionArgument & argument, bool externallyAvailable);

  NodeIndex
  addLambdaNode(const rvsdg::LambdaNode & allocaNode, bool externallyAvailable);

  NodeIndex
  addMallocNode(const rvsdg::Node & mallocNode, bool externallyAvailable);

  [[nodiscard]] NodeIndex
  addRegisterNode();

  void
  mapRegisterToNode(const rvsdg::Output & output, NodeIndex nodeIndex);

  /**
   * Marks the given node with targeting every node that is externally available.
   * Note that this method will not go over other targets and remove doubled up pointees.
   * @param index
   */
  void
  markAsTargetsAllExternallyAvailable(NodeIndex index);

  /**
   * Adds the given \p target to \p source's set of targets.
   * If source is marked as targeting all externally available nodes,
   * and the target is  marked as externally available, this is a no-op.
   * @param source the source node
   * @param target the target node, must be a memory node
   * @return true if the target was added
   */
  bool
  addTarget(NodeIndex source, NodeIndex target);

  [[nodiscard]] bool
  isNodeConstant(NodeIndex index) const noexcept;

  [[nodiscard]] std::optional<size_t>
  tryGetNodeSize(NodeIndex index) const noexcept;

  /**
   * Gets the total number of edges in the PointsToGraph.
   *
   * This can be counted in two different ways:
   *  - Only explicit edges
   *  - Both explicit and implicit edges
   *
   * In both cases, register nodes are only counted once, even if multiple registers map to them.
   *
   * @return a pair (number of explicit edges, total number of edges)
   */
  [[nodiscard]] std::pair<size_t, size_t>
  numEdges() const noexcept;

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

  NodeIndex addNode(NodeKind kind, bool externallyAvailable, const void * object);

  std::vector<NodeData> nodeData_;
  std::vector<util::HashSet<NodeIndex>> nodeTargets_;
  std::vector<const void *> nodeObjects_;

  AllocaNodeMap allocaMap_;
  DeltaNodeMap deltaMap_;
  ImportNodeMap importMap_;
  LambdaNodeMap lambdaMap_;
  MallocNodeMap mallocMap_;
  RegisterNodeMap registerMap_;

  // In-order lists of node indices, for specific node kinds or flags
  std::vector<NodeIndex> registerNodes_;
  std::vector<NodeIndex> externallyAvailableNodes_;
};

}
}

#endif
