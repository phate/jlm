/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/GraphWriter.hpp>

namespace jlm::llvm::aa
{

PointsToGraph::PointsToGraph()
{
  // Create the single external node, representing all memory not represented by any other node
  externalMemoryNode_ = addNode(NodeKind::ExternalNode, true, false, std::nullopt, nullptr);
  // The external node can never be explicitly targeted and never explicitly target any other node
  markAsTargetsAllExternallyAvailable(externalMemoryNode_);
}

PointsToGraph::AllocaNodeRange
PointsToGraph::allocaNodes() const noexcept
{
  return { AllocaNodeIterator(allocaMap_.cbegin()), AllocaNodeIterator(allocaMap_.cend()) };
}

PointsToGraph::DeltaNodeRange
PointsToGraph::deltaNodes() const noexcept
{
  return { DeltaNodeIterator(deltaMap_.cbegin()), DeltaNodeIterator(deltaMap_.cend()) };
}

PointsToGraph::ImportNodeRange
PointsToGraph::importNodes() const noexcept
{
  return { ImportNodeIterator(importMap_.cbegin()), ImportNodeIterator(importMap_.cend()) };
}

PointsToGraph::LambdaNodeRange
PointsToGraph::lambdaNodes() const noexcept
{
  return { LambdaNodeIterator(lambdaMap_.cbegin()), LambdaNodeIterator(lambdaMap_.cend()) };
}

PointsToGraph::MallocNodeRange
PointsToGraph::mallocNodes() const noexcept
{
  return { MallocNodeIterator(mallocMap_.cbegin()), MallocNodeIterator(mallocMap_.cend()) };
}

PointsToGraph::NodeIndex
PointsToGraph::getExternalMemoryNode() const noexcept
{
  return externalMemoryNode_;
}

PointsToGraph::RegisterNodeRange
PointsToGraph::registerNodes() const noexcept
{
  return { registerNodes_.cbegin(), registerNodes_.cend() };
}

PointsToGraph::NodeIndex
PointsToGraph::addNodeForAlloca(const rvsdg::SimpleNode & allocaNode, bool externallyAvailable)
{
  if (!is<AllocaOperation>(allocaNode.GetOperation()))
    throw std::logic_error("Node is not an alloca operation");

  auto [it, added] = allocaMap_.try_emplace(&allocaNode, 0);
  if (!added)
    throw std::logic_error("Alloca node already exists in the graph.");

  // Try to include the size of the allocation in the created node
  const auto getMemorySize = [](const rvsdg::Node & allocaNode) -> std::optional<size_t>
  {
    const auto allocaOp = util::assertedCast<const AllocaOperation>(
        &static_cast<const rvsdg::SimpleNode &>(allocaNode).GetOperation());

    // An alloca has a count parameter, which on rare occasions is not just the constant 1.
    const auto elementCount = tryGetConstantSignedInteger(*allocaNode.input(0)->origin());
    if (elementCount.has_value() && *elementCount >= 0)
      return *elementCount * GetTypeAllocSize(*allocaOp->ValueType());
    return std::nullopt;
  };

  return it->second = addNode(
             NodeKind::AllocaNode,
             externallyAvailable,
             false,
             getMemorySize(allocaNode),
             &allocaNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addNodeForDelta(const rvsdg::DeltaNode & deltaNode, bool externallyAvailable)
{
  auto [it, added] = deltaMap_.try_emplace(&deltaNode, 0);
  if (!added)
    throw std::logic_error("Delta node already exists in the graph.");

  const auto isConstant = deltaNode.GetOperation().constant();
  const auto memorySize = GetTypeAllocSize(*deltaNode.GetOperation().Type());

  return it->second =
             addNode(NodeKind::DeltaNode, externallyAvailable, isConstant, memorySize, &deltaNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addNodeForImport(const rvsdg::GraphImport & import, bool externallyAvailable)
{
  auto [it, added] = importMap_.try_emplace(&import, 0);
  if (!added)
    throw std::logic_error("Import node already exists in the graph.");

  const auto isConstant = [](const rvsdg::GraphImport & import) -> bool
  {
    if (const auto graphImport = dynamic_cast<const GraphImport *>(&import))
      return graphImport->isConstant();
    return false;
  };

  const auto getMemorySize = [](const rvsdg::GraphImport & import) -> std::optional<size_t>
  {
    if (const auto graphImport = dynamic_cast<const GraphImport *>(&import))
    {
      auto size = GetTypeAllocSize(*graphImport->ValueType());

      // C code can contain declarations like this:
      //     extern char myArray[];
      // which means there is an array of unknown size defined in a different module.
      // In the LLVM IR the import gets an array length of 0, but that is not correct.
      if (size != 0)
        return size;
    }
    return std::nullopt;
  };

  return it->second = addNode(
             NodeKind::ImportNode,
             externallyAvailable,
             isConstant(import),
             getMemorySize(import),
             &import);
}

PointsToGraph::NodeIndex
PointsToGraph::addNodeForLambda(const rvsdg::LambdaNode & lambdaNode, bool externallyAvailable)
{
  auto [it, added] = lambdaMap_.try_emplace(&lambdaNode, 0);
  if (!added)
    throw std::logic_error("Lambda node already exists in the graph.");

  // A function can never be written to, so it is regarded constant
  // It should never be read from either, so its size is 0
  return it->second = addNode(NodeKind::LambdaNode, externallyAvailable, true, 0, &lambdaNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addNodeForMalloc(const rvsdg::SimpleNode & mallocNode, bool externallyAvailable)
{
  if (!is<MallocOperation>(mallocNode.GetOperation()))
    throw std::logic_error("Node is not an alloca operation");

  auto [it, added] = mallocMap_.try_emplace(&mallocNode, 0);
  if (!added)
    throw std::logic_error("Malloc node already exists in the graph.");

  const auto tryGetMemorySize = [](const rvsdg::Node & mallocNode) -> std::optional<size_t>
  {
    // If the size parameter of the malloc node is a constant, that is our size
    auto size = tryGetConstantSignedInteger(*MallocOperation::sizeInput(mallocNode).origin());

    // Only return the size if it is a positive integer, to avoid unsigned underflow
    if (size.has_value() && *size >= 0)
      return *size;

    return std::nullopt;
  };

  return it->second = addNode(
             NodeKind::MallocNode,
             externallyAvailable,
             false,
             tryGetMemorySize(mallocNode),
             &mallocNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addNodeForRegisters()
{
  // Registers need to be static single assignment, so they are regarded as constants
  // Their size in memory is not known, as most registers never even live in memory.
  auto index = addNode(NodeKind::RegisterNode, false, true, std::nullopt, nullptr);
  registerNodes_.emplace_back(index);
  return index;
}

void
PointsToGraph::mapRegisterToNode(const rvsdg::Output & output, NodeIndex nodeIndex)
{
  if (!isRegisterNode(nodeIndex))
    throw std::logic_error("Node is not a register node");

  const auto [_, added] = registerMap_.emplace(&output, nodeIndex);
  if (!added)
    throw std::logic_error("Register is already mapped in the graph.");
}

void
PointsToGraph::markAsTargetsAllExternallyAvailable(NodeIndex index)
{
  JLM_ASSERT(index < nodeData_.size());
  if (!nodeData_[index].isTargetingAllExternallyAvailable)
  {
    numNodesTargetingAllExternallyAvailable_++;
    nodeData_[index].isTargetingAllExternallyAvailable = true;
  }
}

bool
PointsToGraph::addTarget(NodeIndex source, NodeIndex target)
{
  JLM_ASSERT(source < nodeExplicitTargets_.size());

  // The external memory node should never be an explicit target
  JLM_ASSERT(target != externalMemoryNode_);
  // The external memory node should never have explicit targets
  JLM_ASSERT(source != externalMemoryNode_);

  // Register nodes can never be targeted
  JLM_ASSERT(getNodeKind(target) != NodeKind::RegisterNode);

  // Skip adding the target if it is already being targeted implicitly
  if (isTargetingAllExternallyAvailable(source) && isExternallyAvailable(target))
    return false;
  return nodeExplicitTargets_[source].insert(target);
}

std::pair<size_t, size_t>
PointsToGraph::numEdges() const noexcept
{
  size_t numExplicitEdges = 0;
  size_t numImplicitEdges = 0;
  size_t numDoubledUpEdges = 0;

  for (NodeIndex i = 0; i < numNodes(); i++)
  {
    for (auto target : getExplicitTargets(i).Items())
    {
      numExplicitEdges++;
      if (isTargetingAllExternallyAvailable(i) && isExternallyAvailable(target))
        numDoubledUpEdges++;
    }

    if (isTargetingAllExternallyAvailable(i))
      numImplicitEdges += externallyAvailableNodes_.size();
  }

  return std::make_pair(numExplicitEdges, numExplicitEdges + numImplicitEdges - numDoubledUpEdges);
}

std::string
PointsToGraph::getNodeDebugString(NodeIndex index, char seperator) const
{
  std::ostringstream ss;
  ss << "(PtG#" << std::setfill('0') << std::setw(3) << index << std::setw(0) << " ";

  switch (getNodeKind(index))
  {
  case NodeKind::AllocaNode:
    ss << getAllocaForNode(index).DebugString();
    break;
  case NodeKind::DeltaNode:
    ss << getDeltaForNode(index).DebugString();
    break;
  case NodeKind::ImportNode:
    ss << getImportForNode(index).debug_string();
    break;
  case NodeKind::LambdaNode:
    ss << getLambdaForNode(index).DebugString();
    break;
  case NodeKind::MallocNode:
    ss << getMallocForNode(index).DebugString();
    break;
  case NodeKind::ExternalNode:
    ss << "external";
    break;
  case NodeKind::RegisterNode:
    ss << "register";
    break;
  default:
    throw std::logic_error("Unknown PtG node kind");
  }

  if (isExternallyAvailable(index))
  {
    ss << seperator << "(ExtAv)";
  }
  if (isTargetingAllExternallyAvailable(index))
  {
    ss << seperator << "(TgtAllExtAv)";
  }

  if (const auto size = tryGetNodeSize(index))
  {
    ss << seperator << "(" << size.value() << " bytes)";
  }
  if (isNodeConstant(index))
  {
    ss << seperator << "(const)";
  }
  ss << " )";

  return ss.str();
}

bool
PointsToGraph::isSupergraphOf(const PointsToGraph & subgraph) const
{
  // Given a memory node in the PointsToGraph called "original",
  // finds the corresponding memory node in the PointsToGraph called "other".
  // If no corresponding memory node exists in the graph, nullopt is returned.
  auto getCorrespondingMemoryNode = [](NodeIndex node,
                                       const PointsToGraph & original,
                                       const PointsToGraph & other) -> std::optional<NodeIndex>
  {
    const auto kind = original.getNodeKind(node);
    if (kind == NodeKind::AllocaNode)
    {
      const auto & alloca = original.getAllocaForNode(node);
      if (!other.hasNodeForAlloca(alloca))
        return std::nullopt;
      return other.getNodeForAlloca(alloca);
    }
    if (kind == NodeKind::DeltaNode)
    {
      const auto & delta = original.getDeltaForNode(node);
      if (!other.hasNodeForDelta(delta))
        return std::nullopt;
      return other.getNodeForDelta(delta);
    }
    if (kind == NodeKind::ImportNode)
    {
      const auto & argument = original.getImportForNode(node);
      if (!other.hasNodeForImport(argument))
        return std::nullopt;
      return other.getNodeForImport(argument);
    }
    if (kind == NodeKind::LambdaNode)
    {
      const auto & lambda = original.getLambdaForNode(node);
      if (!other.hasNodeForLambda(lambda))
        return std::nullopt;
      return other.getNodeForLambda(lambda);
    }
    if (kind == NodeKind::MallocNode)
    {
      const auto & malloc = original.getMallocForNode(node);
      if (!other.hasNodeForMalloc(malloc))
        return std::nullopt;
      return other.getNodeForMalloc(malloc);
    }
    if (kind == NodeKind::ExternalNode)
    {
      return other.getExternalMemoryNode();
    }

    throw std::logic_error("Unknown type of memory node");
  };

  // Given two nodes, checks if the superset node points to everything the subset node points to,
  // and that the superset node has all flags the subset node has
  auto isNodeSuperset = [&](const PointsToGraph & supersetGraph,
                            NodeIndex supersetNode,
                            const PointsToGraph & subsetGraph,
                            NodeIndex subsetNode)
  {
    // The superset node must have any flags set on the subset node
    if (subsetGraph.isExternallyAvailable(subsetNode)
        && !supersetGraph.isExternallyAvailable(supersetNode))
      return false;

    if (subsetGraph.isTargetingAllExternallyAvailable(subsetNode)
        && !supersetGraph.isTargetingAllExternallyAvailable(supersetNode))
      return false;

    // Make sure all targets of the subset node are also targets of the superset node
    for (auto subsetTarget : subsetGraph.getExplicitTargets(subsetNode).Items())
    {
      // There must be a corresponding memory node representing the target in the superset graph
      auto correspondingTarget =
          getCorrespondingMemoryNode(subsetTarget, subsetGraph, supersetGraph);
      if (!correspondingTarget.has_value())
        return false;

      // If the target is a member of the explicit targets of the superset node, it is OK
      if (supersetGraph.getExplicitTargets(supersetNode).Contains(*correspondingTarget))
        continue;

      // The target can also be an implicit target of the superset node
      if (supersetGraph.isTargetingAllExternallyAvailable(supersetNode)
          && supersetGraph.isExternallyAvailable(*correspondingTarget))
        continue;

      // If we get here, the subset node has a target that the superset node does not
      return false;
    }

    return true;
  };

  // Given a memory node from the subgraph, check that a corresponding node exists in this graph.
  // All edges the other node has, must have corresponding edges in this graph.
  // If the other node is marked as escaping, the corresponding node must be as well.
  auto hasSuperOfMemoryNode = [&](NodeIndex subsetNode)
  {
    auto thisNode = getCorrespondingMemoryNode(subsetNode, subgraph, *this);
    if (!thisNode.has_value())
      return false;

    return isNodeSuperset(*this, *thisNode, subgraph, subsetNode);
  };

  // Iterate through all memory nodes in the subgraph, and make sure we have corresponding nodes
  for (const auto node : subgraph.allocaNodes())
  {
    if (!hasSuperOfMemoryNode(node))
      return false;
  }
  for (const auto node : subgraph.deltaNodes())
  {
    if (!hasSuperOfMemoryNode(node))
      return false;
  }
  for (const auto node : subgraph.importNodes())
  {
    if (!hasSuperOfMemoryNode(node))
      return false;
  }
  for (const auto node : subgraph.lambdaNodes())
  {
    if (!hasSuperOfMemoryNode(node))
      return false;
  }
  for (const auto node : subgraph.mallocNodes())
  {
    if (!hasSuperOfMemoryNode(node))
      return false;
  }

  // For each register mapped to a RegisterNode in the subgraph, this graph must also have mapped
  // the same register to be a supergraph. The RegisterNode must point to a superset of what
  // the subgraph's RegisterNode points to.
  for (auto [rvsdgOutput, subNode] : subgraph.registerMap_)
  {
    if (!hasNodeForRegister(*rvsdgOutput))
      return false;
    const auto supersetRegisterNode = getNodeForRegister(*rvsdgOutput);

    if (!isNodeSuperset(*this, supersetRegisterNode, subgraph, subNode))
      return false;
  }

  return true;
}

PointsToGraph::NodeIndex
PointsToGraph::addNode(
    NodeKind kind,
    bool externallyAvailable,
    bool isConstant,
    std::optional<size_t> memorySize,
    const void * object)
{
  const auto index = nodeData_.size();

  nodeData_.emplace_back(NodeData(kind, externallyAvailable, false, isConstant, memorySize));
  nodeExplicitTargets_.emplace_back();
  nodeObjects_.push_back(object);

  if (externallyAvailable)
    externallyAvailableNodes_.push_back(index);

  return index;
}

void
PointsToGraph::dumpGraph(util::graph::Writer & graphWriter, const PointsToGraph & pointsToGraph)
{
  const auto [explicitEdges, totalEdges] = pointsToGraph.numEdges();

  auto & graph = graphWriter.CreateGraph();
  graph.SetLabel("Points-to graph");
  graph.AppendToLabel(
      util::strfmt("Explicit edges: ", explicitEdges, ", total edges: ", totalEdges));

  std::vector<util::graph::Node *> nodes;
  nodes.resize(pointsToGraph.numNodes());

  for (NodeIndex ptgNode = 0; ptgNode < pointsToGraph.numNodes(); ptgNode++)
  {
    auto & node = graph.CreateNode();
    node.SetLabel(pointsToGraph.getNodeDebugString(ptgNode, '\n'));
    if (pointsToGraph.isExternallyAvailable(ptgNode))
      node.SetFillColor(util::graph::Colors::Yellow);

    if (pointsToGraph.isMemoryNode(ptgNode))
    {
      // Memory nodes are boxes, and have an associated object
      node.SetShape("box");
      node.SetAttributeObject("rvsdgObject", pointsToGraph.nodeObjects_[ptgNode]);
    }
    else
    {
      // Register nodes are oval
      node.SetShape("oval");
    }

    nodes[ptgNode] = &node;
  }

  // Attach all rvsdg outputs to their RegisterNode, using incrementing attribute names
  std::unordered_map<NodeIndex, size_t> outputCount;
  for (const auto & [rvsdgOutput, ptgNode] : pointsToGraph.registerMap_)
  {
    const auto count = outputCount[ptgNode]++;
    nodes[ptgNode]->SetAttributeObject(util::strfmt("output", count), rvsdgOutput);
  }

  // Add all explicit edges
  for (NodeIndex ptgNode = 0; ptgNode < pointsToGraph.numNodes(); ptgNode++)
  {
    for (auto target : pointsToGraph.getExplicitTargets(ptgNode).Items())
    {
      graph.CreateEdge(*nodes[ptgNode], *nodes[target], true);
    }
  }
}

std::string
PointsToGraph::dumpDot(const PointsToGraph & pointsToGraph)
{
  util::graph::Writer writer;
  dumpGraph(writer, pointsToGraph);
  std::ostringstream ss;
  writer.outputAllGraphs(ss, util::graph::OutputFormat::Dot);
  return ss.str();
}

}
