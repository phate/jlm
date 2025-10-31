/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/trace.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

#include <RVSDG/Ops.h.inc>
#include <typeindex>
#include <unordered_map>

namespace jlm::llvm::aa
{

PointsToGraph::PointsToGraph()
{}

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

PointsToGraph::RegisterNodeRange
PointsToGraph::registerNodes() const noexcept
{
  return { registerNodes_.cbegin(), registerNodes_.cend() };
}

PointsToGraph::NodeIndex
PointsToGraph::addAllocaNode(const rvsdg::Node & allocaNode, bool externallyAvailable)
{
  if (!is<AllocaOperation>(&allocaNode))
    throw std::logic_error("Node is not an alloca node");

  auto [it, added] = allocaMap_.try_emplace(&allocaNode, 0);
  if (!added)
    throw std::logic_error("Alloca node already exists in the graph.");

  return it->second = addNode(NodeKind::AllocaNode, externallyAvailable, &allocaNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addDeltaNode(const rvsdg::DeltaNode & deltaNode, bool externallyAvailable)
{
  auto [it, added] = deltaMap_.try_emplace(&deltaNode, 0);
  if (!added)
    throw std::logic_error("Delta node already exists in the graph.");

  return it->second = addNode(NodeKind::DeltaNode, externallyAvailable, &deltaNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addImportNode(const rvsdg::RegionArgument & argument, bool externallyAvailable)
{
  auto [it, added] = importMap_.try_emplace(&argument, 0);
  if (!added)
    throw std::logic_error("Import node already exists in the graph.");

  return it->second = addNode(NodeKind::ImportNode, externallyAvailable, &argument);
}

PointsToGraph::NodeIndex
PointsToGraph::addLambdaNode(const rvsdg::LambdaNode & lambdaNode, bool externallyAvailable)
{
  auto [it, added] = lambdaMap_.try_emplace(&lambdaNode, 0);
  if (!added)
    throw std::logic_error("Lambda node already exists in the graph.");

  return it->second = addNode(NodeKind::LambdaNode, externallyAvailable, &lambdaNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addMallocNode(const rvsdg::Node & mallocNode, bool externallyAvailable)
{
  if (!is<MallocOperation>(&mallocNode))
    throw std::logic_error("Node is not a malloc node");

  auto [it, added] = mallocMap_.try_emplace(&mallocNode, 0);
  if (!added)
    throw std::logic_error("Malloc node already exists in the graph.");

  return it->second = addNode(NodeKind::MallocNode, externallyAvailable, &mallocNode);
}

PointsToGraph::NodeIndex
PointsToGraph::addRegisterNode()
{
  auto index = addNode(NodeKind::RegisterNode, false, nullptr);
  registerNodes_.emplace_back(index);
  return index;
}

void
PointsToGraph::mapRegisterToNode(const rvsdg::Output & output, NodeIndex nodeIndex)
{
  if (getKind(nodeIndex) != NodeKind::RegisterNode)
    throw std::logic_error("Node is not a register node");

  const auto [_, added] = registerMap_.emplace(&output, nodeIndex);
  if (!added)
    throw std::logic_error("Register is already mapped in the graph.");
}

void
PointsToGraph::markAsTargetsAllExternallyAvailable(NodeIndex index)
{
  JLM_ASSERT(index < nodeData_.size());
  nodeData_[index].isTargetingAllExternallyAvailable = true;
}

bool
PointsToGraph::addTarget(NodeIndex source, NodeIndex target)
{
  JLM_ASSERT(source < nodeTargets_.size());

  // A register is the only type of node that can not be targeted
  JLM_ASSERT(getKind(target) != NodeKind::RegisterNode);

  if (isTargetingAllExternallyAvailable(source) && isExternallyAvailable(target))
    return false;
  return nodeTargets_[source].insert(target);
}

bool
PointsToGraph::isNodeConstant(NodeIndex index) const noexcept
{
  const auto kind = getKind(index);
  switch (kind)
  {
  case NodeKind::AllocaNode:
    return false;
  case NodeKind::DeltaNode:
  {
    const auto & deltaNode = getDeltaNodeObject(index);
    return deltaNode.constant();
  }
  case NodeKind::ImportNode:
  {
    const auto & import = getImportNodeObject(index);
    if (const auto graphImport = dynamic_cast<const GraphImport *>(&import))
      return graphImport->isConstant();
    return false;
  }
  case NodeKind::LambdaNode:
    return true;
  case NodeKind::MallocNode:
    return false;
  case NodeKind::RegisterNode:
    // Registers are not memory, but are by all means constant
    return true;
  default:
    JLM_UNREACHABLE("Unknown PtG node kind");
  }
}

std::optional<size_t>
PointsToGraph::tryGetNodeSize(NodeIndex index) const noexcept
{
  const auto kind = getKind(index);
  switch (kind)
  {
  case NodeKind::AllocaNode:
  {
    const auto & allocaNode = getAllocaNodeObject(index);
    const auto allocaOp = util::assertedCast<const AllocaOperation>(&allocaNode.GetOperation());

    // An alloca has a count parameter, which on rare occasions is not just the constant 1.
    const auto elementCount = tryGetConstantSignedInteger(*allocaNode.input(0)->origin());
    if (elementCount.has_value())
      return *elementCount * GetTypeSize(*allocaOp->ValueType());

    return std::nullopt;
  }
  case NodeKind::DeltaNode:
  {
    const auto & deltaNode = getDeltaNodeObject(index);
    return GetTypeSize(*deltaNode.GetOperation().Type());
  }
  case NodeKind::ImportNode:
  {
    const auto & import = getImportNodeObject(index);
    if (const auto graphImport = dynamic_cast<const GraphImport *>(&import))
    {
      auto size = GetTypeSize(*graphImport->ValueType());

      // C code can contain declarations like this:
      //     extern char myArray[];
      // which means there is an array of unknown size defined in a different module.
      // In the LLVM IR the import gets an array length of 0, but that is not correct.
      if (size != 0)
        return size;
    }
    return std::nullopt;
  }
  case NodeKind::LambdaNode:
  {
    // Functions should never be read from or written to, so use size 0
    return 0;
  }
  case NodeKind::MallocNode:
  {
    const auto & mallocNode = getMallocNodeObject(index);
    // If the size parameter of the malloc node is a constant, that is our size
    auto size = tryGetConstantSignedInteger(*mallocNode.input(0)->origin());

    // Only return the size if it is a positive integer, to avoid unsigned underflow
    if (size.has_value() && *size >= 0)
      return *size;

    return std::nullopt;
  }
  case NodeKind::RegisterNode:
    // Registers are not memory
    return std::nullopt;
  default:
    JLM_UNREACHABLE("Unknown PtG node kind");
  }
}

std::pair<size_t, size_t>
PointsToGraph::numEdges() const noexcept
{
  size_t numExplicitEdges = 0;
  size_t numImplicitEdges = 0;

  for (NodeIndex i = 0; i < numNodes(); i++)
  {
    numExplicitEdges += nodeTargets_[i].Size();
    if (isTargetingAllExternallyAvailable(i))
      numImplicitEdges += externallyAvailableNodes_.size();
  }

  return std::make_pair(numExplicitEdges, numExplicitEdges + numImplicitEdges);
}

bool
PointsToGraph::IsSupergraphOf([[maybe_unused]] const jlm::llvm::aa::PointsToGraph & subgraph) const
{
  /*
  TODO: Translate to new PointsToGraph
  // Given a memory node representing a memory object in an RVSDG module, this function finds
  // a memory node representing the same memory object in a different PointsToGraph.
  // If no corresponding memory node exists in the graph, nullptr is returned
  auto GetCorrespondingMemoryNode =
      [](const PointsToGraph::MemoryNode & node,
         const PointsToGraph & graph) -> const PointsToGraph::MemoryNode *
  {
    if (auto allocaNode = dynamic_cast<const AllocaNode *>(&node))
    {
      if (auto it = graph.AllocaNodes_.find(&allocaNode->GetAllocaNode());
          it != graph.AllocaNodes_.end())
        return it->second.get();
    }
    else if (auto deltaNode = dynamic_cast<const DeltaNode *>(&node))
    {
      if (auto it = graph.deltaMap_.find(&deltaNode->GetDeltaNode());
          it != graph.deltaMap_.end())
        return it->second.get();
    }
    else if (auto importNode = dynamic_cast<const ImportNode *>(&node))
    {
      if (auto it = graph.ImportNodes_.find(&importNode->GetArgument());
          it != graph.ImportNodes_.end())
        return it->second.get();
    }
    else if (auto lambdaNode = dynamic_cast<const LambdaNode *>(&node))
    {
      if (auto it = graph.LambdaNodes_.find(&lambdaNode->GetLambdaNode());
          it != graph.LambdaNodes_.end())
        return it->second.get();
    }
    else if (auto mallocNode = dynamic_cast<const MallocNode *>(&node))
    {
      if (auto it = graph.MallocNodes_.find(&mallocNode->GetMallocNode());
          it != graph.MallocNodes_.end())
        return it->second.get();
    }
    else if (MemoryNode::Is<UnknownMemoryNode>(node))
    {
      return &graph.GetUnknownMemoryNode();
    }
    else if (MemoryNode::Is<ExternalMemoryNode>(node))
    {
      return &graph.GetExternalMemoryNode();
    }
    else
      JLM_UNREACHABLE("Unknown type of MemoryNode");

    return nullptr;
  };

  // Given two nodes, checks if the first node points to everything the second points to.
  // The nodes can belong to different PointsToGraphs.
  auto HasSupersetOfPointees = [&GetCorrespondingMemoryNode](
                                   const PointsToGraph::Node & superset,
                                   const PointsToGraph::Node & subset)
  {
    // Early return if the subset is larger
    if (subset.NumTargets() > superset.NumTargets())
      return false;

    for (auto & subsetTarget : subset.Targets())
    {
      auto correspondingTarget = GetCorrespondingMemoryNode(subsetTarget, superset.Graph());

      // Check if the superset is pointing to the target
      if (correspondingTarget == nullptr || !superset.HasTarget(*correspondingTarget))
        return false;
    }
    return true;
  };

  // Given a memory node from the subgraph, check that a corresponding node exists in this graph.
  // All edges the other node has, must have corresponding edges in this graph.
  // If the other node is marked as escaping, the corresponding node must be as well.
  auto HasSuperOfMemoryNode = [&](const PointsToGraph::MemoryNode & subNode)
  {
    auto thisNode = GetCorrespondingMemoryNode(subNode, *this);
    if (thisNode == nullptr)
      return false;

    if (subNode.IsModuleEscaping() && !thisNode->IsModuleEscaping())
      return false;

    return HasSupersetOfPointees(*thisNode, subNode);
  };

  // Early return if the subgraph is representing more memory objects or registers than us
  if (subgraph.NumMemoryNodes() > NumMemoryNodes())
    return false;
  if (subgraph.RegisterNodeMap_.size() > RegisterNodeMap_.size())
    return false;

  // Iterate through all memory nodes in the subgraph, and make sure we have corresponding nodes
  for (auto & node : subgraph.AllocaNodes())
  {
    if (!HasSuperOfMemoryNode(node))
      return false;
  }
  for (auto & node : subgraph.DeltaNodes())
  {
    if (!HasSuperOfMemoryNode(node))
      return false;
  }
  for (auto & node : subgraph.ImportNodes())
  {
    if (!HasSuperOfMemoryNode(node))
      return false;
  }
  for (auto & node : subgraph.LambdaNodes())
  {
    if (!HasSuperOfMemoryNode(node))
      return false;
  }
  for (auto & node : subgraph.MallocNodes())
  {
    if (!HasSuperOfMemoryNode(node))
      return false;
  }

  // For each register mapped to a RegisterNode in the subgraph, this graph must also have mapped
  // the same register to be a supergraph. The RegisterNode must point to a superset of what
  // the subgraph's RegisterNode points to.
  for (auto [rvsdgOutput, subNode] : subgraph.RegisterNodeMap_)
  {
    auto correspondingRegisterNode = RegisterNodeMap_.find(rvsdgOutput);
    if (correspondingRegisterNode == RegisterNodeMap_.end())
      return false;

    if (!HasSupersetOfPointees(*correspondingRegisterNode->second, *subNode))
      return false;
  }*/

  return true;
}

std::string
PointsToGraph::ToDot(
    const PointsToGraph & pointsToGraph,
    const std::unordered_map<const rvsdg::Output *, std::string> & outputMap)
{
  auto nodeFill = [&](NodeIndex node)
  {
    // Nodes that are marked as having escaped the module get a background color
    if (pointsToGraph.isExternallyAvailable(node))
      return "style=filled, fillcolor=\"yellow\", ";
    return "";
  };

  auto nodeShape = [&](NodeIndex node)
  {
    if (pointsToGraph.getKind(node) == NodeKind::RegisterNode)
      return "oval";
    return "box";
  };

  auto nodeLabel = [&](NodeIndex node) -> std::string
  {
    const auto kind = pointsToGraph.getKind(node);
    switch (kind)
    {
    case NodeKind::AllocaNode:
      return pointsToGraph.getAllocaNodeObject(node).DebugString();
    case NodeKind::DeltaNode:
      return pointsToGraph.getDeltaNodeObject(node).DebugString();
    case NodeKind::ImportNode:
      return pointsToGraph.getImportNodeObject(node).debug_string();
    case NodeKind::LambdaNode:
      return pointsToGraph.getLambdaNodeObject(node).DebugString();
    case NodeKind::MallocNode:
      return pointsToGraph.getMallocNodeObject(node).DebugString();
    case NodeKind::RegisterNode:
      return "register"; // TODO: Include which outputs it maps to
    default:
      JLM_UNREACHABLE("Unknown PtG node kind");
    }
  };

  auto nodeTargetsAllExternalLabel = [&](NodeIndex node) -> std::string_view
  {
    if (pointsToGraph.isTargetingAllExternallyAvailable(node))
      return "\\n->*";
    return "";
  };

  auto nodeString = [&](NodeIndex node)
  {
    return util::strfmt(
        "{ ",
        node,
        " [",
        nodeFill(node),
        "label = \"",
        nodeLabel(node),
        nodeTargetsAllExternalLabel(node),
        "\" ",
        "shape = \"",
        nodeShape(node),
        "\"]; }\n");
  };

  auto edgeString = [](NodeIndex source, NodeIndex target)
  {
    return util::strfmt(source, " -> ", target, "\n");
  };

  auto printNodeAndEdges = [&](NodeIndex node)
  {
    std::string dot;
    dot += nodeString(node);
    for (auto & target : pointsToGraph.getTargets(node).Items())
      dot += edgeString(node, target);

    return dot;
  };

  std::string dot("digraph PointsToGraph {\n");
  for (size_t i = 0; i < pointsToGraph.numNodes(); i++)
  {
    dot += printNodeAndEdges(i);
  }
  dot += "label=\"Yellow = Escaping memory node\"\n";
  dot += "}\n";

  return dot;
}

PointsToGraph::UnknownMemoryNode::~UnknownMemoryNode() noexcept = default;

std::string
PointsToGraph::UnknownMemoryNode::DebugString() const
{
  return "UnknownMemory";
}

std::optional<size_t>
PointsToGraph::UnknownMemoryNode::tryGetSize() const noexcept
{
  return std::nullopt;
}

bool
PointsToGraph::UnknownMemoryNode::isConstant() const noexcept
{
  return false;
}

PointsToGraph::ExternalMemoryNode::~ExternalMemoryNode() noexcept = default;

std::string
PointsToGraph::ExternalMemoryNode::DebugString() const
{
  return "ExternalMemory";
}

std::optional<size_t>
PointsToGraph::ExternalMemoryNode::tryGetSize() const noexcept
{
  return std::nullopt;
}

bool
PointsToGraph::ExternalMemoryNode::isConstant() const noexcept
{
  return false;
}

}
