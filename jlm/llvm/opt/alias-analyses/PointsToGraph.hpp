/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOGRAPH_HPP

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/HashSet.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/util/Math.hpp>

#include <memory>
#include <string>
#include <unordered_map>

namespace jlm::util::graph
{
class Writer;
}

namespace jlm::llvm
{

class RvsdgModule;

namespace aa
{

/**
 * The PointsToGraph is a graph where nodes represent virtual registers or locations in memory.
 * An edge, e.g., X -> Y, represents the possibility of node X containing a pointer to
 * memory represented by node Y.
 *
 * An edge X -> Y can be represented in two ways:
 *  - explicitly: node Y is a member of node X's set of explicit targets.
 *  - implicitly: node X is flagged as targeting all externally available memory,
 *                and node Y is flagged as being externally available.
 */
class PointsToGraph final
{
public:
  using NodeIndex = uint32_t;

  enum class NodeKind : uint8_t
  {
    // Register nodes can never be the target of pointers, and cannot be externally available
    RegisterNode = 0,
    // All other kinds of nodes are memory nodes, representing one or more memory locations
    AllocaNode,
    DeltaNode,
    ImportNode,
    LambdaNode,
    MallocNode,
    // Only one external node exists. It represents all memory not represented by any other node.
    ExternalNode,
    COUNT
  };

private:
  struct NodeData
  {
    // The kind of node
    NodeKind kind : util::BitWidthOfEnum(NodeKind::COUNT);

    // When set, the node is available from other modules in the program
    uint8_t isExternallyAvailable : 1;

    // When set, the points-to graph node implicitly targets all externally available nodes
    uint8_t isTargetingAllExternallyAvailable : 1;

    // When set, the node represents constant memory / a constant register
    uint8_t isConstant : 1;

    // The size of the memory allocation(s) represented by the node.
    // If the size is unknown, or too large to represent, it is set to UnknownMemorySize
    uint16_t memorySize : 10;
    static constexpr uint16_t UnknownMemorySize = (1 << 10) - 1;

    NodeData(
        NodeKind kind,
        bool externallyAvailable,
        bool targetsAllExternallyAvailable,
        bool isConstant,
        std::optional<size_t> memorySize)
        : kind(kind),
          isExternallyAvailable(externallyAvailable),
          isTargetingAllExternallyAvailable(targetsAllExternallyAvailable),
          isConstant(isConstant),
          memorySize(
              memorySize.has_value() && *memorySize < UnknownMemorySize ? *memorySize
                                                                        : UnknownMemorySize)
    {}
  };

  static_assert(sizeof(NodeData) == sizeof(uint16_t), "NodeData must fit in 16 bits");

  using AllocaNodeMap = std::unordered_map<const rvsdg::SimpleNode *, NodeIndex>;
  using DeltaNodeMap = std::unordered_map<const rvsdg::DeltaNode *, NodeIndex>;
  using ImportNodeMap = std::unordered_map<const rvsdg::GraphImport *, NodeIndex>;
  using LambdaNodeMap = std::unordered_map<const rvsdg::LambdaNode *, NodeIndex>;
  using MallocNodeMap = std::unordered_map<const rvsdg::SimpleNode *, NodeIndex>;
  using RegisterNodeMap = std::unordered_map<const rvsdg::Output *, NodeIndex>;

  using AllocaNodeIterator = util::MapValueIterator<const NodeIndex, AllocaNodeMap::const_iterator>;
  using AllocaNodeRange = util::IteratorRange<AllocaNodeIterator>;

  using DeltaNodeIterator = util::MapValueIterator<const NodeIndex, DeltaNodeMap::const_iterator>;
  using DeltaNodeRange = util::IteratorRange<DeltaNodeIterator>;

  using ImportNodeIterator = util::MapValueIterator<const NodeIndex, ImportNodeMap::const_iterator>;
  using ImportNodeRange = util::IteratorRange<ImportNodeIterator>;

  using LambdaNodeIterator = util::MapValueIterator<const NodeIndex, LambdaNodeMap::const_iterator>;
  using LambdaNodeRange = util::IteratorRange<LambdaNodeIterator>;

  using MallocNodeIterator = util::MapValueIterator<const NodeIndex, MallocNodeMap::const_iterator>;
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

  /**
   * @return iterator range of all nodes in the PointsToGraph that are of the AllocaNode kind.
   */
  AllocaNodeRange
  allocaNodes() const noexcept;

  /**
   * @return iterator range of all nodes in the PointsToGraph that are of the DeltaNode kind.
   */
  DeltaNodeRange
  deltaNodes() const noexcept;

  /**
   * @return iterator range of all nodes in the PointsToGraph that are of the ImportNode kind.
   */
  ImportNodeRange
  importNodes() const noexcept;

  /**
   * @return iterator range of all nodes in the PointsToGraph that are of the LambdaNode kind.
   */
  LambdaNodeRange
  lambdaNodes() const noexcept;

  /**
   * @return iterator range of all nodes in the PointsToGraph that are of the MallocNode kind.
   */
  MallocNodeRange
  mallocNodes() const noexcept;

  /**
   * @return the index of the node that represents all memory that is not represented by any
   * other memory node. It is impossible to target this node directly, so it must be targeted
   * through the flag for targeting all external memory.
   * It is also impossible to add explicit targets to this node, so all targets must be added
   * using the flag for marking memory as externally available.
   */
  NodeIndex
  getExternalMemoryNode() const noexcept;

  /**
   * @return iterator range of all nodes in the PointsToGraph that are of the RegisterNode kind.
   */
  RegisterNodeRange
  registerNodes() const noexcept;

  /**
   * @return the number of nodes in the PointsToGraph that are of the AllocaNode kind.
   */
  size_t
  numAllocaNodes() const noexcept
  {
    return allocaMap_.size();
  }

  /**
   * @return the number of nodes in the PointsToGraph that are of the DeltaNode kind.
   */
  size_t
  numDeltaNodes() const noexcept
  {
    return deltaMap_.size();
  }

  /**
   * @return the number of nodes in the PointsToGraph that are of the ImportNode kind.
   */
  size_t
  numImportNodes() const noexcept
  {
    return importMap_.size();
  }

  /**
   * @return the number of nodes in the PointsToGraph that are of the LambdaNode kind.
   */
  size_t
  numLambdaNodes() const noexcept
  {
    return lambdaMap_.size();
  }

  /**
   * @return the number of nodes in the PointsToGraph that are of the MallocNode kind.
   */
  size_t
  numMallocNodes() const noexcept
  {
    return mallocMap_.size();
  }

  /**
   * @return the number of nodes in the PointsToGraph that are of the RegisterNode kind.
   */
  [[nodiscard]] size_t
  numRegisterNodes() const noexcept
  {
    return registerNodes_.size();
  }

  /**
   * @return the total number of \ref rvsdg::Output's mapped to nodes in the PointsToGraph.
   */
  [[nodiscard]] size_t
  numMappedRegisters() const noexcept
  {
    return registerMap_.size();
  }

  /**
   * @return the number of nodes in the PointsToGraph that are classified as memory nodes,
   * i.e., all nodes that are not register nodes.
   */
  size_t
  numMemoryNodes() const noexcept
  {
    return numAllocaNodes() + numDeltaNodes() + numImportNodes() + numLambdaNodes()
         + numMallocNodes() + 1; // The external node
  }

  /**
   * @return the total number of nodes in the PointsToGraph.
   */
  size_t
  numNodes() const noexcept
  {
    JLM_ASSERT(nodeData_.size() == nodeExplicitTargets_.size());
    JLM_ASSERT(nodeData_.size() == nodeObjects_.size());
    return nodeData_.size();
  }

  /**
   * Checks whether a PointsToGraph AllocaNode has been created for the given RVSDG SimpleNode.
   * The SimpleNode must correspond to an AllocaOperation.
   * @param node the RVSDG node to look up.
   * @return true if an AllocaNode is mapped to this RVSDG node, false otherwise.
   */
  [[nodiscard]] bool
  hasNodeForAlloca(const rvsdg::SimpleNode & node) const
  {
    return allocaMap_.find(&node) != allocaMap_.end();
  }

  /**
   * Checks whether a PointsToGraph DeltaNode has been created for the given RVSDG DeltaNode.
   * @param node the RVSDG delta node to look up.
   * @return true if a DeltaNode is mapped to this node, false otherwise.
   */
  [[nodiscard]] bool
  hasNodeForDelta(const rvsdg::DeltaNode & node) const
  {
    return deltaMap_.find(&node) != deltaMap_.end();
  }

  /**
   * Checks whether a PointsToGraph ImportNode has been created for the given RVSDG GraphImport.
   * @param argument the RVSDG import to look up.
   * @return true if an ImportNode is mapped to this import, false otherwise.
   */
  [[nodiscard]] bool
  hasNodeForImport(const rvsdg::GraphImport & argument) const
  {
    return importMap_.find(&argument) != importMap_.end();
  }

  /**
   * Checks whether a PointsToGraph LambdaNode has been created for the given RVSDG LambdaNode.
   * @param node the RVSDG lambda node to look up.
   * @return true if a LambdaNode is mapped to this node, false otherwise.
   */
  [[nodiscard]] bool
  hasNodeForLambda(const rvsdg::LambdaNode & node) const
  {
    return lambdaMap_.find(&node) != lambdaMap_.end();
  }

  /**
   * Checks whether a PointsToGraph MallocNode has been created for the given RVSDG SimpleNode.
   * The SimpleNode must correspond to a MallocOperation.
   * @param node the RVSDG node to look up.
   * @return true if a MallocNode is mapped to this RVSDG node, false otherwise.
   */
  [[nodiscard]] bool
  hasNodeForMalloc(const rvsdg::SimpleNode & node) const
  {
    return mallocMap_.find(&node) != mallocMap_.end();
  }

  /**
   * Checks whether a PointsToGraph RegisterNode has been created for the given RVSDG Output.
   * @param output the RVSDG output to look up.
   * @return true if a RegisterNode is mapped to this output, false otherwise.
   */
  [[nodiscard]] bool
  hasNodeForRegister(const rvsdg::Output & output) const
  {
    return registerMap_.find(&output) != registerMap_.end();
  }

  /**
   * Retrieves the index of the PointsToGraph node mapped to the given \ref rvsdg::SimpleNode.
   * @param node the node being looked up, which must contain an \ref AllocaOperation.
   * @return the PointsToGraph node mapped to \p node, always of AllocaNode kind.
   * @throws std::out_of_range if the alloca node is not mapped to a PointsToGraph node.
   */
  [[nodiscard]] NodeIndex
  getNodeForAlloca(const rvsdg::SimpleNode & node) const
  {
    return allocaMap_.at(&node);
  }

  /**
   * Retrieves the index of the PointsToGraph node mapped to the given \ref DeltaNode.
   * @param node the delta node being looked up.
   * @return the PointsToGraph node mapped to \p node, always of DeltaNode kind.
   * @throws std::out_of_range if the delta node is not mapped to a PointsToGraph node.
   */
  NodeIndex
  getNodeForDelta(const rvsdg::DeltaNode & node) const
  {
    return deltaMap_.at(&node);
  }

  /**
   * Retrieves the index of the PointsToGraph node mapped to the given \ref GraphImport.
   * @param argument the import being looked up.
   * @return the PointsToGraph node mapped to \p argument, always of ImportNode kind.
   * @throws std::out_of_range if the argument is not mapped to a PointsToGraph node.
   */
  NodeIndex
  getNodeForImport(const rvsdg::GraphImport & argument) const
  {
    return importMap_.at(&argument);
  }

  /**
   * Retrieves the index of the PointsToGraph node mapped to the given \ref LambdaNode.
   * @param node the lambda node being looked up.
   * @return the PointsToGraph node mapped to \p node, always of LambdaNode kind.
   * @throws std::out_of_range if the lambda node is not mapped to a PointsToGraph node.
   */
  NodeIndex
  getNodeForLambda(const rvsdg::LambdaNode & node) const
  {
    return lambdaMap_.at(&node);
  }

  /**
   * Retrieves the index of the PointsToGraph node mapped to the given \ref rvsdg::SimpleNode.
   * @param node the node being looked up, which must contain a \ref MallocOperation.
   * @return the PointsToGraph node mapped to \p node, always of MallocNode kind.
   * @throws std::out_of_range if the malloc node is not mapped to a PointsToGraph node.
   */
  NodeIndex
  getNodeForMalloc(const rvsdg::SimpleNode & node) const
  {
    return mallocMap_.at(&node);
  }

  /**
   * Retrieves the index of the PointsToGraph node mapped to the given \p output.
   * @param output the RVSDG output being looked up.
   * @return the PointsToGraph node mapped to \p output, always of RegisterNode kind.
   * @throws std::out_of_range if the output is not mapped to a PointsToGraph node.
   */
  NodeIndex
  getNodeForRegister(const rvsdg::Output & output) const
  {
    return registerMap_.at(&output);
  }

  /**
   * Gets the NodeKind of the node with the given \p index.
   * @param index the index of the node in question.
   * @return the NodeKind of the given node.
   */
  [[nodiscard]] NodeKind
  getNodeKind(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeData_.size());
    return nodeData_[index].kind;
  }

  /**
   * Checks if the PointsToGraph node with the given \p index is of the RegisterNode kind.
   * @param index the index of the node in question.
   * @return true if node is a register node, otherwise false.
   */
  [[nodiscard]] bool
  isRegisterNode(NodeIndex index) const
  {
    return getNodeKind(index) == NodeKind::RegisterNode;
  }

  /**
   * Checks if the PointsToGraph node with the given \p index is a memory node.
   * This means all nodes that are not of the RegisterNode kind.
   * @param index the index of the node in question.
   * @return true if node is a memory node, otherwise false.
   */
  [[nodiscard]] bool
  isMemoryNode(NodeIndex index) const
  {
    return !isRegisterNode(index);
  }

  /**
   * Checks the the PointsToGraph node with the given \p index is flagged as externally available.
   * Only memory nodes can have this flag.
   * @param index the index of the PointsToGraph node.
   * @return true if the node is flagged, false otherwise.
   */
  [[nodiscard]] bool
  isExternallyAvailable(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeData_.size());
    return nodeData_[index].isExternallyAvailable;
  }

  /**
   * Checks if the PointsToGraph node with the given \p index is flagged as
   * targeting all externally available memory.
   * If it is, it implicitly targets every PointsToGraph node flagged as externally available.
   * @param index the index of the PointsToGraph node.
   * @return true if the node is flagged, false otherwise.
   */
  [[nodiscard]] bool
  isTargetingAllExternallyAvailable(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeData_.size());
    return nodeData_[index].isTargetingAllExternallyAvailable;
  }

  /**
   * Checks if the memory represented by the given node is representing constant memory.
   * Constant memory means the value stored in the memory never changes during execution.
   * @param index the index of the PointsToGraph node
   * @return true if the memory represented by the node is constant
   */
  [[nodiscard]] bool
  isNodeConstant(NodeIndex index) const noexcept
  {
    JLM_ASSERT(index < nodeData_.size());
    return nodeData_[index].isConstant;
  }

  /**
   * Gets the size of the memory location represented by the node with the given \p index.
   * The node can also represent many locations, as long as they all have the same size.
   * If the memory locations have varying or unknown size, nullopt is returned.
   * @param index the index of the PointsToGraph node
   * @return the size of the memory location(s) in bytes, or nullopt if unknown.
   */
  [[nodiscard]] std::optional<size_t>
  tryGetNodeSize(NodeIndex index) const noexcept
  {
    JLM_ASSERT(index < nodeData_.size());
    const auto size = nodeData_[index].memorySize;
    if (size == NodeData::UnknownMemorySize)
      return std::nullopt;
    return size;
  }

  /**
   * Gets the set of memory nodes that are targeted explicitly by the node with the given \p index.
   * When a node X explicitly targets a node Y, it means that the value(s) represented by X may
   * contain a pointer to memory represented by Y. Node Y must be a memory node.
   * Explicit targets is one way of representing pointer-pointee relations,
   * the other being implicit targets using flags.
   * @see isExternallyAvailable
   * @see isTargetingAllExternallyAvailable
   * @param index the index of the PointsToGraph node X.
   * @return the set of indices of PointsToGraph nodes targeted by X.
   */
  [[nodiscard]] const util::HashSet<NodeIndex> &
  getExplicitTargets(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeExplicitTargets_.size());
    return nodeExplicitTargets_[index];
  }

  /**
   * Checks if the node with index \p source targets the node with index \p target,
   * either explicitly or implicitly.
   * @param source the pointing node.
   * @param target the node that is possibly a pointee.
   * @return true if \p source targets \p target, otherwise false.
   */
  [[nodiscard]] bool
  isTargeting(NodeIndex source, NodeIndex target) const
  {
    if (isTargetingAllExternallyAvailable(source) && isExternallyAvailable(target))
      return true;
    return getExplicitTargets(source).Contains(target);
  }

  /**
   * @return the indices of all PointsToGraph nodes that are marked as being externally available.
   * Only memory nodes can have this flag.
   */
  const std::vector<NodeIndex> &
  getExternallyAvailableNodes() const noexcept
  {
    return externallyAvailableNodes_;
  }

  /**
   * @return the number of nodes in the PointsToGraph that are flagged as being
   * externally available. Only memory nodes can have this flag.
   */
  size_t
  numExternallyAvailableNodes() const noexcept
  {
    return externallyAvailableNodes_.size();
  }

  /**
   * @return the number of nodes in the PointsToGraph that are flagged as possibly
   * pointing to all memory nodes that are externally available.
   */
  size_t
  numNodesTargetingAllExternallyAvailable() const noexcept
  {
    return numNodesTargetingAllExternallyAvailable_;
  }

  /**
   * Gets the \ref rvsdg::SimpleNode mapped to the PointsToGraph node with the given \p index.
   * @param index the PointsToGraph node.
   * @return the associated node, which contains a \ref llvm::AllocaOperation.
   * @throws std::logic_error if the given node is not of the AllocaNode kind.
   */
  const rvsdg::SimpleNode &
  getAllocaForNode(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    if (nodeData_[index].kind != NodeKind::AllocaNode)
      throw std::logic_error("PointsToGraph node is not an AllocaNode");
    return *static_cast<const rvsdg::SimpleNode *>(nodeObjects_[index]);
  }

  /**
   * Gets the \ref rvsdg::DeltaNode mapped to the PointsToGraph node with the given \p index.
   * @param index the PointsToGraph node.
   * @return the associated delta node.
   * @throws std::logic_error if the given node is not of the DeltaNode kind.
   */
  const rvsdg::DeltaNode &
  getDeltaForNode(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    if (nodeData_[index].kind != NodeKind::DeltaNode)
      throw std::logic_error("PointsToGraph node is not a DeltaNode");
    return *static_cast<const rvsdg::DeltaNode *>(nodeObjects_[index]);
  }

  /**
   * Gets the \ref rvsdg::GraphImport mapped to the PointsToGraph node with the given \p index.
   * @param index the PointsToGraph node.
   * @return the associated graph import argument.
   * @throws std::logic_error if the given node is not of the ImportNode kind.
   */
  const rvsdg::GraphImport &
  getImportForNode(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    if (nodeData_[index].kind != NodeKind::ImportNode)
      throw std::logic_error("PointsToGraph node is not an ImportNode");
    return *static_cast<const rvsdg::GraphImport *>(nodeObjects_[index]);
  }

  /**
   * Gets the \ref rvsdg::LambdaNode mapped to the PointsToGraph node with the given \p index.
   * @param index the PointsToGraph node.
   * @return the associated lambda node.
   * @throws std::logic_error if the given node is not of the LambdaNode kind.
   */
  const rvsdg::LambdaNode &
  getLambdaForNode(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    if (nodeData_[index].kind != NodeKind::LambdaNode)
      throw std::logic_error("PointsToGraph node is not a LambdaNode");
    return *static_cast<const rvsdg::LambdaNode *>(nodeObjects_[index]);
  }

  /**
   * Gets the \ref rvsdg::SimpleNode mapped to the PointsToGraph node with the given \p index.
   * @param index the PointsToGraph node.
   * @return the associated node, which contains a \ref llvm::MallocOperation.
   * @throws std::logic_error if the given node is not of the MallocNode kind.
   */
  const rvsdg::SimpleNode &
  getMallocForNode(NodeIndex index) const
  {
    JLM_ASSERT(index < nodeObjects_.size());
    if (nodeData_[index].kind != NodeKind::MallocNode)
      throw std::logic_error("PointsToGraph node is not a MallocNode");
    return *static_cast<const rvsdg::SimpleNode *>(nodeObjects_[index]);
  }

  /**
   * Creates a new PointsToGraph node mapped to the given \p allocaNode,
   * which must have the \ref llvm::AllocaOperation.
   * @param allocaNode the alloca node. Must not already be mapped to any other node.
   * @param externallyAvailable if the created node should be flagged as externally available.
   * @return the index of the created PointsToGraph node.
   */
  NodeIndex
  addNodeForAlloca(const rvsdg::SimpleNode & allocaNode, bool externallyAvailable);

  /**
   * Creates a new PointsToGraph node of DeltaNode kind, mapped to the given \p deltaNode.
   * @param deltaNode the node.
   * @param externallyAvailable if the created node should be flagged as externally available.
   * @return the index of the created PointsToGraph node.
   */
  NodeIndex
  addNodeForDelta(const rvsdg::DeltaNode & deltaNode, bool externallyAvailable);

  /**
   * Creates a new PointsToGraph node of ImportNode kind, mapped to the given \p argument.
   * @param argument the graph import argument. Must not already be mapped to any other node.
   * @param externallyAvailable if the created node should be flagged as externally available.
   * @return the index of the created PointsToGraph node.
   */
  NodeIndex
  addNodeForImport(const rvsdg::GraphImport & argument, bool externallyAvailable);

  /**
   * Creates a new PointsToGraph node of LambdaNode kind, mapped to the given \p lambdaNode.
   * @param lambdaNode the lambda node. Must not already be mapped to any other node.
   * @param externallyAvailable if the created node should be flagged as externally available.
   * @return the index of the created PointsToGraph node.
   */
  NodeIndex
  addNodeForLambda(const rvsdg::LambdaNode & lambdaNode, bool externallyAvailable);

  /**
   * Creates a new PointsToGraph node mapped to the given \p mallocNode,
   * which must have the \ref llvm::MallocOperation.
   * @param mallocNode the malloc node. Must not already be mapped to any other node.
   * @param externallyAvailable if the created node should be flagged as externally available.
   * @return the index of the created PointsToGraph node.
   */
  NodeIndex
  addNodeForMalloc(const rvsdg::SimpleNode & mallocNode, bool externallyAvailable);

  /**
   * Creates a new PointsToGraph node of kind RegisterNode.
   * @see mapRegisterToNode to map RVSDG outputs to the PointsToGraph node.
   * @return the index of the new PointsToGraph node.
   */
  [[nodiscard]] NodeIndex
  addNodeForRegisters();

  /**
   * Maps the given \p output to the PointsToGraph node with the given \p nodeIndex.
   * The PointsToGraph node must be of the RegisterNode kind.
   * @param output the RVSDG output that is mapped
   * @param nodeIndex the index of the PointsToGraph node it is mapped to
   */
  void
  mapRegisterToNode(const rvsdg::Output & output, NodeIndex nodeIndex);

  /**
   * Marks the given node as targeting every node that is externally available.
   * Note that this method will not go over other targets and remove doubled up pointees.
   * @param index the index of the PointsToGraph node that should be marked
   */
  void
  markAsTargetsAllExternallyAvailable(NodeIndex index);

  /**
   * Adds the given \p target to \p source's set of targets.
   * If source is marked as targeting all externally available nodes,
   * and the target is marked as externally available, this is a no-op.
   *
   * Neither the target nor the source can be the external node.
   * That node can only be a target / targeted via flags.
   *
   * @param source the source node. Can not be the external node.
   * @param target the target node. Must be a memory node, and not the external node.
   * @return true if the target was added, or any flags were changed
   */
  bool
  addTarget(NodeIndex source, NodeIndex target);

  /**
   * Gets the total number of edges in the PointsToGraph.
   *
   * This can be counted in two different ways:
   *  - Only explicit edges
   *  - Both explicit and implicit edges (only counting doubled up pointees once)
   *
   * In both cases, register nodes are only counted once, even if multiple registers map to them.
   *
   * @return a pair (number of explicit edges, total number of edges)
   */
  [[nodiscard]] std::pair<size_t, size_t>
  numEdges() const noexcept;

  /**
   * Produces a debug string for the given node,
   * containing the index, flags, size, constness and info about the underlying object.
   * Does not include explicit targets.
   * @param index the index of the node to create a debug string for
   * @param separator the char to put between parts of the debug string
   * @return the debug string
   */
  [[nodiscard]] std::string
  getNodeDebugString(NodeIndex index, char separator = ' ') const;

  /**
   * Checks if this PointsToGraph is a supergraph of \p subgraph.
   * Every node in the subgraph needs to have corresponding nodes in the supergraph.
   * Any flags present in the subgraph must also be present in the supergraph.
   * All pointer-pointee relations in the subgraph must be represented in the supergraph,
   * either as explicit or implicit targets.
   * @param subgraph the graph to compare against
   * @return true if this graph is a supergraph of the given subgraph, false otherwise
   */
  [[nodiscard]] bool
  isSupergraphOf(const PointsToGraph & subgraph) const;

  /**
   * Writes the given \p pointsToGraph to the given \p graphWriter.
   * Each node is associated with the rvsdg object(s) it represents through attribute(s).
   */
  static void
  dumpGraph(util::graph::Writer & graphWriter, const PointsToGraph & pointsToGraph);

  /**
   * Shorthand for dumping the given \p pointsToGraph as a string in the GraphViz dot format.
   * @return the points to graph in dot format.
   */
  static std::string
  dumpDot(const PointsToGraph & pointsToGraph);

  static std::unique_ptr<PointsToGraph>
  create()
  {
    return std::unique_ptr<PointsToGraph>(new PointsToGraph());
  }

private:
  NodeIndex
  addNode(
      NodeKind kind,
      bool externallyAvailable,
      bool isConstant,
      std::optional<size_t> memorySize,
      const void * object);

  // Vectors containing information per node
  std::vector<NodeData> nodeData_;
  // For each node, the set of memory nodes that it explicitly targets
  std::vector<util::HashSet<NodeIndex>> nodeExplicitTargets_;
  // For each memory node, this vector contains a pointer to the associated RVSDG object.
  std::vector<const void *> nodeObjects_;

  // Mappings from RVSDG objects to their corresponding PointsToGraph NodeIndex
  AllocaNodeMap allocaMap_;
  DeltaNodeMap deltaMap_;
  ImportNodeMap importMap_;
  LambdaNodeMap lambdaMap_;
  MallocNodeMap mallocMap_;

  // The external node, representing all memory not represented by any other memory node
  NodeIndex externalMemoryNode_;

  // RegisterNode is the only kind of PointsToGraph nodes where multiple
  // \ref rvsdg::Output* can be mapped to the same PointsToGraph node.
  RegisterNodeMap registerMap_;

  // In-order list of node indices containing all nodes of RegisterNode kind
  std::vector<NodeIndex> registerNodes_;
  // In-order list of node indices containing all nodes that are flagged isExternallyAvailable
  std::vector<NodeIndex> externallyAvailableNodes_;
  // The number of nodes that have the isTargetingAllExternallyAvailable flag
  size_t numNodesTargetingAllExternallyAvailable_ = 0;
};

}
}

#endif
