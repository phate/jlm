/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/trace.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

#include <typeindex>
#include <unordered_map>

namespace jlm::llvm::aa
{

PointsToGraph::PointsToGraph()
    : NextMemoryNodeId_(0)
{
  UnknownMemoryNode_ = UnknownMemoryNode::Create(*this);
  ExternalMemoryNode_ = ExternalMemoryNode::Create(*this);

  // The external memory node has by definition always escaped
  EscapedMemoryNodes_.insert(ExternalMemoryNode_.get());
}

void
PointsToGraph::AddEscapedMemoryNode(PointsToGraph::MemoryNode & memoryNode)
{
  JLM_ASSERT(&memoryNode.Graph() == this);
  EscapedMemoryNodes_.insert(&memoryNode);
}

PointsToGraph::AllocaNodeRange
PointsToGraph::AllocaNodes()
{
  return { AllocaNodeIterator(AllocaNodes_.begin()), AllocaNodeIterator(AllocaNodes_.end()) };
}

PointsToGraph::AllocaNodeConstRange
PointsToGraph::AllocaNodes() const
{
  return { AllocaNodeConstIterator(AllocaNodes_.begin()),
           AllocaNodeConstIterator(AllocaNodes_.end()) };
}

PointsToGraph::DeltaNodeRange
PointsToGraph::DeltaNodes()
{
  return { DeltaNodeIterator(DeltaNodes_.begin()), DeltaNodeIterator(DeltaNodes_.end()) };
}

PointsToGraph::DeltaNodeConstRange
PointsToGraph::DeltaNodes() const
{
  return { DeltaNodeConstIterator(DeltaNodes_.begin()), DeltaNodeConstIterator(DeltaNodes_.end()) };
}

PointsToGraph::LambdaNodeRange
PointsToGraph::LambdaNodes()
{
  return { LambdaNodeIterator(LambdaNodes_.begin()), LambdaNodeIterator(LambdaNodes_.end()) };
}

PointsToGraph::LambdaNodeConstRange
PointsToGraph::LambdaNodes() const
{
  return { LambdaNodeConstIterator(LambdaNodes_.begin()),
           LambdaNodeConstIterator(LambdaNodes_.end()) };
}

PointsToGraph::MallocNodeRange
PointsToGraph::MallocNodes()
{
  return { MallocNodeIterator(MallocNodes_.begin()), MallocNodeIterator(MallocNodes_.end()) };
}

PointsToGraph::MallocNodeConstRange
PointsToGraph::MallocNodes() const
{
  return { MallocNodeConstIterator(MallocNodes_.begin()),
           MallocNodeConstIterator(MallocNodes_.end()) };
}

PointsToGraph::ImportNodeRange
PointsToGraph::ImportNodes()
{
  return { ImportNodeIterator(ImportNodes_.begin()), ImportNodeIterator(ImportNodes_.end()) };
}

PointsToGraph::ImportNodeConstRange
PointsToGraph::ImportNodes() const
{
  return { ImportNodeConstIterator(ImportNodes_.begin()),
           ImportNodeConstIterator(ImportNodes_.end()) };
}

PointsToGraph::RegisterNodeRange
PointsToGraph::RegisterNodes()
{
  return { RegisterNodeIterator(RegisterNodes_.begin()),
           RegisterNodeIterator(RegisterNodes_.end()) };
}

PointsToGraph::RegisterNodeConstRange
PointsToGraph::RegisterNodes() const
{
  return { RegisterNodeConstIterator(RegisterNodes_.begin()),
           RegisterNodeConstIterator(RegisterNodes_.end()) };
}

PointsToGraph::AllocaNode &
PointsToGraph::AddAllocaNode(std::unique_ptr<PointsToGraph::AllocaNode> node)
{
  auto tmp = node.get();
  AllocaNodes_[&node->GetAllocaNode()] = std::move(node);

  return *tmp;
}

PointsToGraph::DeltaNode &
PointsToGraph::AddDeltaNode(std::unique_ptr<PointsToGraph::DeltaNode> node)
{
  auto tmp = node.get();
  DeltaNodes_[&node->GetDeltaNode()] = std::move(node);

  return *tmp;
}

PointsToGraph::LambdaNode &
PointsToGraph::AddLambdaNode(std::unique_ptr<PointsToGraph::LambdaNode> node)
{
  auto tmp = node.get();
  LambdaNodes_[&node->GetLambdaNode()] = std::move(node);

  return *tmp;
}

PointsToGraph::MallocNode &
PointsToGraph::AddMallocNode(std::unique_ptr<PointsToGraph::MallocNode> node)
{
  auto tmp = node.get();
  MallocNodes_[&node->GetMallocNode()] = std::move(node);

  return *tmp;
}

PointsToGraph::RegisterNode &
PointsToGraph::AddRegisterNode(std::unique_ptr<PointsToGraph::RegisterNode> node)
{
  auto tmp = node.get();
  for (auto output : node->GetOutputs().Items())
    RegisterNodeMap_[output] = tmp;

  RegisterNodes_.emplace_back(std::move(node));

  return *tmp;
}

PointsToGraph::ImportNode &
PointsToGraph::AddImportNode(std::unique_ptr<PointsToGraph::ImportNode> node)
{
  auto tmp = node.get();
  ImportNodes_[&node->GetArgument()] = std::move(node);

  return *tmp;
}

std::pair<size_t, size_t>
PointsToGraph::NumEdges() const noexcept
{
  size_t numEdges = 0;

  auto countMemoryNodes = [&](auto iterable)
  {
    for (const MemoryNode & node : iterable)
    {
      numEdges += node.NumTargets();
    }
  };

  countMemoryNodes(AllocaNodes());
  countMemoryNodes(DeltaNodes());
  countMemoryNodes(ImportNodes());
  countMemoryNodes(LambdaNodes());
  countMemoryNodes(MallocNodes());

  numEdges += GetExternalMemoryNode().NumTargets();

  // For register nodes, the number of edges and number of points-to relations is different
  size_t numPointsToRelations = numEdges;
  for (auto & registerNode : RegisterNodes())
  {
    numEdges += registerNode.NumTargets();
    numPointsToRelations += registerNode.NumTargets() * registerNode.GetOutputs().Size();
  }

  return std::make_pair(numEdges, numPointsToRelations);
}

bool
PointsToGraph::IsSupergraphOf(const jlm::llvm::aa::PointsToGraph & subgraph) const
{
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
      if (auto it = graph.DeltaNodes_.find(&deltaNode->GetDeltaNode());
          it != graph.DeltaNodes_.end())
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
  }

  return true;
}

std::string
PointsToGraph::ToDot(
    const PointsToGraph & pointsToGraph,
    const std::unordered_map<const rvsdg::Output *, std::string> & outputMap)
{
  auto nodeFill = [&](const PointsToGraph::Node & node)
  {
    // Nodes that are marked as having escaped the module get a background color
    if (const auto memoryNode = dynamic_cast<const MemoryNode *>(&node))
      if (pointsToGraph.GetEscapedMemoryNodes().Contains(memoryNode))
        return "style=filled, fillcolor=\"yellow\", ";
    return "";
  };

  auto nodeShape = [](const PointsToGraph::Node & node)
  {
    static std::unordered_map<std::type_index, std::string> shapes(
        { { typeid(AllocaNode), "box" },
          { typeid(DeltaNode), "box" },
          { typeid(ImportNode), "box" },
          { typeid(LambdaNode), "box" },
          { typeid(MallocNode), "box" },
          { typeid(RegisterNode), "oval" },
          { typeid(UnknownMemoryNode), "box" },
          { typeid(ExternalMemoryNode), "box" } });

    if (shapes.find(typeid(node)) != shapes.end())
      return shapes[typeid(node)];

    JLM_UNREACHABLE("Unknown points-to graph Node type.");
  };

  auto nodeLabel = [&](const PointsToGraph::Node & node)
  {
    // If the node is NOT a register node, then the label is just the DebugString
    auto registerNode = dynamic_cast<const RegisterNode *>(&node);
    if (registerNode == nullptr)
    {
      return node.DebugString();
    }

    // Otherwise, include the mapped name (if any) to its rvsdg::outputs.
    std::string label;
    auto outputs = registerNode->GetOutputs();
    for (auto output : outputs.Items())
    {
      label += RegisterNode::ToString(*output);
      if (auto it = outputMap.find(output); it != outputMap.end())
      {
        label += util::strfmt(" (", it->second, ")");
      }
      label += "\\n";
    }

    return label;
  };

  auto nodeString = [&](const PointsToGraph::Node & node)
  {
    return util::strfmt(
        "{ ",
        reinterpret_cast<uintptr_t>(&node),
        " [",
        nodeFill(node),
        "label = \"",
        nodeLabel(node),
        "\" ",
        "shape = \"",
        nodeShape(node),
        "\"]; }\n");
  };

  auto edgeString = [](const PointsToGraph::Node & node, const PointsToGraph::Node & target)
  {
    return util::strfmt(
        reinterpret_cast<uintptr_t>(&node),
        " -> ",
        reinterpret_cast<uintptr_t>(&target),
        "\n");
  };

  auto printNodeAndEdges = [&](const PointsToGraph::Node & node)
  {
    std::string dot;
    dot += nodeString(node);
    for (auto & target : node.Targets())
      dot += edgeString(node, target);

    return dot;
  };

  std::string dot("digraph PointsToGraph {\n");
  for (auto & allocaNode : pointsToGraph.AllocaNodes())
    dot += printNodeAndEdges(allocaNode);

  for (auto & deltaNode : pointsToGraph.DeltaNodes())
    dot += printNodeAndEdges(deltaNode);

  for (auto & importNode : pointsToGraph.ImportNodes())
    dot += printNodeAndEdges(importNode);

  for (auto & lambdaNode : pointsToGraph.LambdaNodes())
    dot += printNodeAndEdges(lambdaNode);

  for (auto & mallocNode : pointsToGraph.MallocNodes())
    dot += printNodeAndEdges(mallocNode);

  for (auto & registerNode : pointsToGraph.RegisterNodes())
    dot += printNodeAndEdges(registerNode);

  dot += nodeString(pointsToGraph.GetUnknownMemoryNode());
  dot += nodeString(pointsToGraph.GetExternalMemoryNode());
  dot += "label=\"Yellow = Escaping memory node\"\n";
  dot += "}\n";

  return dot;
}

PointsToGraph::Node::~Node() noexcept = default;

PointsToGraph::Node::TargetRange
PointsToGraph::Node::Targets()
{
  return { TargetIterator(Targets_.begin()), TargetIterator(Targets_.end()) };
}

PointsToGraph::Node::TargetConstRange
PointsToGraph::Node::Targets() const
{
  return { TargetConstIterator(Targets_.begin()), TargetConstIterator(Targets_.end()) };
}

bool
PointsToGraph::Node::HasTarget(const PointsToGraph::MemoryNode & target) const
{
  return Targets_.find(const_cast<MemoryNode *>(&target)) != Targets_.end();
}

PointsToGraph::Node::SourceRange
PointsToGraph::Node::Sources()
{
  return { SourceIterator(Sources_.begin()), SourceIterator(Sources_.end()) };
}

PointsToGraph::Node::SourceConstRange
PointsToGraph::Node::Sources() const
{
  return { SourceConstIterator(Sources_.begin()), SourceConstIterator(Sources_.end()) };
}

bool
PointsToGraph::Node::HasSource(const PointsToGraph::Node & source) const
{
  return Sources_.find(const_cast<Node *>(&source)) != Sources_.end();
}

void
PointsToGraph::Node::AddEdge(PointsToGraph::MemoryNode & target)
{
  if (&Graph() != &target.Graph())
    throw util::Error("Points-to graph nodes are not in the same graph.");

  Targets_.insert(&target);
  target.Sources_.insert(this);
}

void
PointsToGraph::Node::RemoveEdge(PointsToGraph::MemoryNode & target)
{
  if (&Graph() != &target.Graph())
    throw util::Error("Points-to graph nodes are not in the same graph.");

  target.Sources_.erase(this);
  Targets_.erase(&target);
}

PointsToGraph::RegisterNode::~RegisterNode() noexcept = default;

std::string
PointsToGraph::RegisterNode::ToString(const rvsdg::Output & output)
{
  auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(output);

  if (node != nullptr)
    return util::strfmt(node->DebugString(), ":o", output.index());

  node = output.region()->node();
  if (node != nullptr)
    return util::strfmt(node->DebugString(), ":a", output.index());

  if (auto graphImport = dynamic_cast<const GraphImport *>(&output))
  {
    return util::strfmt("import:", graphImport->Name());
  }

  return "RegisterNode";
}

std::string
PointsToGraph::RegisterNode::DebugString() const
{
  auto & outputs = GetOutputs();

  size_t n = 0;
  std::string debugString;
  for (auto output : outputs.Items())
  {
    debugString += ToString(*output);
    debugString += n != (outputs.Size() - 1) ? "\n" : "";
    n++;
  }

  return debugString;
}

PointsToGraph::MemoryNode::~MemoryNode() noexcept = default;

PointsToGraph::AllocaNode::~AllocaNode() noexcept = default;

std::string
PointsToGraph::AllocaNode::DebugString() const
{
  return GetAllocaNode().DebugString();
}

PointsToGraph::DeltaNode::~DeltaNode() noexcept = default;

std::string
PointsToGraph::DeltaNode::DebugString() const
{
  return GetDeltaNode().DebugString();
}

PointsToGraph::LambdaNode::~LambdaNode() noexcept = default;

std::string
PointsToGraph::LambdaNode::DebugString() const
{
  return GetLambdaNode().DebugString();
}

PointsToGraph::MallocNode::~MallocNode() noexcept = default;

std::string
PointsToGraph::MallocNode::DebugString() const
{
  return GetMallocNode().DebugString();
}

PointsToGraph::ImportNode::~ImportNode() noexcept = default;

std::string
PointsToGraph::ImportNode::DebugString() const
{
  return GetArgument().Name();
}

PointsToGraph::UnknownMemoryNode::~UnknownMemoryNode() noexcept = default;

std::string
PointsToGraph::UnknownMemoryNode::DebugString() const
{
  return "UnknownMemory";
}

PointsToGraph::ExternalMemoryNode::~ExternalMemoryNode() noexcept = default;

std::string
PointsToGraph::ExternalMemoryNode::DebugString() const
{
  return "ExternalMemory";
}

std::optional<size_t>
getMemoryNodeSize(const PointsToGraph::MemoryNode & memoryNode)
{
  if (dynamic_cast<const PointsToGraph::LambdaNode *>(&memoryNode))
  {
    // Functions should never be read from or written to, so they have no size
    return 0;
  }
  if (auto delta = dynamic_cast<const PointsToGraph::DeltaNode *>(&memoryNode))
  {
    return GetTypeSize(*delta->GetDeltaNode().GetOperation().Type());
  }
  if (auto import = dynamic_cast<const PointsToGraph::ImportNode *>(&memoryNode))
  {
    auto size = GetTypeSize(*import->GetArgument().ValueType());

    // C code can contain declarations like this:
    //     extern char myArray[];
    // which means there is an array of unknown size defined in a different module.
    // In the LLVM IR the import gets an array length of 0, but that is not correct.
    if (size == 0)
      return std::nullopt;

    return size;
  }
  if (auto alloca = dynamic_cast<const PointsToGraph::AllocaNode *>(&memoryNode))
  {
    const auto & allocaNode = alloca->GetAllocaNode();
    const auto allocaOp = util::assertedCast<const AllocaOperation>(&allocaNode.GetOperation());

    // An alloca has a count parameter, which on rare occasions is not just the constant 1.
    const auto elementCount = tryGetConstantSignedInteger(*allocaNode.input(0)->origin());
    if (elementCount.has_value())
      return *elementCount * GetTypeSize(*allocaOp->ValueType());

    return std::nullopt;
  }
  if (auto malloc = dynamic_cast<const PointsToGraph::MallocNode *>(&memoryNode))
  {
    const auto & mallocNode = malloc->GetMallocNode();

    return tryGetConstantSignedInteger(*mallocNode.input(0)->origin());
  }
  if (dynamic_cast<const PointsToGraph::ExternalMemoryNode *>(&memoryNode))
  {
    return std::nullopt;
  }

  throw std::logic_error("Unknown memory node type.");
}

bool
isMemoryNodeConstant(const PointsToGraph::MemoryNode & memoryNode)
{
  if (dynamic_cast<const PointsToGraph::LambdaNode *>(&memoryNode))
  {
    // Functions are always constant memory
    return true;
  }
  if (auto delta = dynamic_cast<const PointsToGraph::DeltaNode *>(&memoryNode))
  {
    return delta->GetDeltaNode().constant();
  }
  if (auto import = dynamic_cast<const PointsToGraph::ImportNode *>(&memoryNode))
  {
    return import->GetArgument().isConstant();
  }
  return false;
}

}
