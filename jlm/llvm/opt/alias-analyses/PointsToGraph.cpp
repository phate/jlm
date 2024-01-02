/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

#include <typeindex>
#include <unordered_map>

namespace jlm::llvm::aa
{

PointsToGraph::PointsToGraph()
{
  UnknownMemoryNode_ = UnknownMemoryNode::Create(*this);
  ExternalMemoryNode_ = ExternalMemoryNode::Create(*this);
}

void
PointsToGraph::AddEscapedMemoryNode(PointsToGraph::MemoryNode & memoryNode)
{
  JLM_ASSERT(&memoryNode.Graph() == this);
  EscapedMemoryNodes_.Insert(&memoryNode);
}

PointsToGraph::AllocaNodeRange
PointsToGraph::AllocaNodes()
{
  auto mapToPointer = [](const AllocaNodeMap::iterator & it)
  {
    return it->second.get();
  };

  return { AllocaNodeIterator(AllocaNodes_.begin(), mapToPointer),
           AllocaNodeIterator(AllocaNodes_.end(), mapToPointer) };
}

PointsToGraph::AllocaNodeConstRange
PointsToGraph::AllocaNodes() const
{
  auto mapToPointer = [](const AllocaNodeMap::const_iterator & it)
  {
    return it->second.get();
  };

  return { AllocaNodeConstIterator(AllocaNodes_.begin(), mapToPointer),
           AllocaNodeConstIterator(AllocaNodes_.end(), mapToPointer) };
}

PointsToGraph::DeltaNodeRange
PointsToGraph::DeltaNodes()
{
  auto mapToPointer = [](const DeltaNodeMap::iterator & it)
  {
    return it->second.get();
  };

  return { DeltaNodeIterator(DeltaNodes_.begin(), mapToPointer),
           DeltaNodeIterator(DeltaNodes_.end(), mapToPointer) };
}

PointsToGraph::DeltaNodeConstRange
PointsToGraph::DeltaNodes() const
{
  auto mapToPointer = [](const DeltaNodeMap::const_iterator & it)
  {
    return it->second.get();
  };

  return { DeltaNodeConstIterator(DeltaNodes_.begin(), mapToPointer),
           DeltaNodeConstIterator(DeltaNodes_.end(), mapToPointer) };
}

PointsToGraph::LambdaNodeRange
PointsToGraph::LambdaNodes()
{
  auto mapToPointer = [](const LambdaNodeMap::iterator & it)
  {
    return it->second.get();
  };

  return { LambdaNodeIterator(LambdaNodes_.begin(), mapToPointer),
           LambdaNodeIterator(LambdaNodes_.end(), mapToPointer) };
}

PointsToGraph::LambdaNodeConstRange
PointsToGraph::LambdaNodes() const
{
  auto mapToPointer = [](const LambdaNodeMap::const_iterator & it)
  {
    return it->second.get();
  };

  return { LambdaNodeConstIterator(LambdaNodes_.begin(), mapToPointer),
           LambdaNodeConstIterator(LambdaNodes_.end(), mapToPointer) };
}

PointsToGraph::MallocNodeRange
PointsToGraph::MallocNodes()
{
  auto mapToPointer = [](const MallocNodeMap::iterator & it)
  {
    return it->second.get();
  };

  return { MallocNodeIterator(MallocNodes_.begin(), mapToPointer),
           MallocNodeIterator(MallocNodes_.end(), mapToPointer) };
}

PointsToGraph::MallocNodeConstRange
PointsToGraph::MallocNodes() const
{
  auto mapToPointer = [](const MallocNodeMap::const_iterator & it)
  {
    return it->second.get();
  };

  return { MallocNodeConstIterator(MallocNodes_.begin(), mapToPointer),
           MallocNodeConstIterator(MallocNodes_.end(), mapToPointer) };
}

PointsToGraph::ImportNodeRange
PointsToGraph::ImportNodes()
{
  auto mapToPointer = [](const ImportNodeMap::iterator & it)
  {
    return it->second.get();
  };

  return { ImportNodeIterator(ImportNodes_.begin(), mapToPointer),
           ImportNodeIterator(ImportNodes_.end(), mapToPointer) };
}

PointsToGraph::ImportNodeConstRange
PointsToGraph::ImportNodes() const
{
  auto mapToPointer = [](const ImportNodeMap::const_iterator & it)
  {
    return it->second.get();
  };

  return { ImportNodeConstIterator(ImportNodes_.begin(), mapToPointer),
           ImportNodeConstIterator(ImportNodes_.end(), mapToPointer) };
}

PointsToGraph::RegisterNodeRange
PointsToGraph::RegisterNodes()
{
  auto mapToPointer = [](const RegisterNodeMap::iterator & it)
  {
    return it->second.get();
  };

  return { RegisterNodeIterator(RegisterNodes_.begin(), mapToPointer),
           RegisterNodeIterator(RegisterNodes_.end(), mapToPointer) };
}

PointsToGraph::RegisterNodeConstRange
PointsToGraph::RegisterNodes() const
{
  auto mapToPointer = [](const RegisterNodeMap::const_iterator & it)
  {
    return it->second.get();
  };

  return { RegisterNodeConstIterator(RegisterNodes_.begin(), mapToPointer),
           RegisterNodeConstIterator(RegisterNodes_.end(), mapToPointer) };
}

PointsToGraph::RegisterSetNodeRange
PointsToGraph::RegisterSetNodes()
{
  auto mapToPointer = [](const RegisterSetNodeMap::iterator & it)
  {
    return it->second;
  };

  return { RegisterSetNodeIterator(RegisterSetNodeMap_.begin(), mapToPointer),
           RegisterSetNodeIterator(RegisterSetNodeMap_.end(), mapToPointer) };
}

PointsToGraph::RegisterSetNodeConstRange
PointsToGraph::RegisterSetNodes() const
{
  auto mapToPointer = [](const RegisterSetNodeMap::const_iterator & it)
  {
    return it->second;
  };

  return { RegisterSetNodeConstIterator(RegisterSetNodeMap_.begin(), mapToPointer),
           RegisterSetNodeConstIterator(RegisterSetNodeMap_.end(), mapToPointer) };
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
  RegisterNodes_[&node->GetOutput()] = std::move(node);

  return *tmp;
}

PointsToGraph::RegisterSetNode &
PointsToGraph::AddRegisterSetNode(std::unique_ptr<PointsToGraph::RegisterSetNode> node)
{
  auto tmp = node.get();
  for (auto output : node->GetOutputs().Items())
    RegisterSetNodeMap_[output] = tmp;

  RegisterSetNodes_.emplace_back(std::move(node));

  return *tmp;
}

PointsToGraph::ImportNode &
PointsToGraph::AddImportNode(std::unique_ptr<PointsToGraph::ImportNode> node)
{
  auto tmp = node.get();
  ImportNodes_[&node->GetArgument()] = std::move(node);

  return *tmp;
}

std::string
PointsToGraph::ToDot(
    const PointsToGraph & pointsToGraph,
    const std::unordered_map<const rvsdg::output *, std::string> & outputMap)
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
          { typeid(RegisterSetNode), "oval" },
          { typeid(UnknownMemoryNode), "box" },
          { typeid(ExternalMemoryNode), "box" } });

    if (shapes.find(typeid(node)) != shapes.end())
      return shapes[typeid(node)];

    JLM_UNREACHABLE("Unknown points-to graph Node type.");
  };

  auto nodeLabel = [&](const PointsToGraph::Node & node)
  {
    // If the node is a RegisterNode, and has a name mapped to its rvsdg::output, include that name
    if (const auto registerNode = dynamic_cast<const RegisterNode *>(&node))
      if (const auto it = outputMap.find(&registerNode->GetOutput()); it != outputMap.end())
        return util::strfmt(node.DebugString(), " (", it->second, ")");

    // Otherwise the label is just the DebugString.
    return node.DebugString();
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

  for (auto & registerSetNode : pointsToGraph.RegisterSetNodes())
    dot += printNodeAndEdges(registerSetNode);

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

void
PointsToGraph::Node::AddEdge(PointsToGraph::MemoryNode & target)
{
  if (&Graph() != &target.Graph())
    throw util::error("Points-to graph nodes are not in the same graph.");

  Targets_.insert(&target);
  target.Sources_.insert(this);
}

void
PointsToGraph::Node::RemoveEdge(PointsToGraph::MemoryNode & target)
{
  if (&Graph() != &target.Graph())
    throw util::error("Points-to graph nodes are not in the same graph.");

  target.Sources_.erase(this);
  Targets_.erase(&target);
}

PointsToGraph::RegisterNode::~RegisterNode() noexcept = default;

static std::string
CreateDotString(const rvsdg::output & output)
{
  auto node = jlm::rvsdg::node_output::node(&output);

  if (node != nullptr)
    return util::strfmt(node->operation().debug_string(), ":o", output.index());

  node = output.region()->node();
  if (node != nullptr)
    return util::strfmt(node->operation().debug_string(), ":a", output.index());

  if (is_import(&output))
  {
    auto port = util::AssertedCast<const impport>(&output.port());
    return util::strfmt("import:", port->name());
  }

  return "RegisterNode";
}

std::string
PointsToGraph::RegisterNode::DebugString() const
{
  return CreateDotString(GetOutput());
}

PointsToGraph::RegisterSetNode::~RegisterSetNode() noexcept = default;

std::string
PointsToGraph::RegisterSetNode::DebugString() const
{
  auto & outputs = GetOutputs();

  size_t n = 0;
  std::string debugString("{");
  for (auto output : outputs.Items())
  {
    debugString += CreateDotString(*output);
    debugString += n != (outputs.Size() - 1) ? ", " : "";
    n++;
  }
  debugString += "}";

  return debugString;
}

PointsToGraph::MemoryNode::~MemoryNode() noexcept = default;

PointsToGraph::AllocaNode::~AllocaNode() noexcept = default;

std::string
PointsToGraph::AllocaNode::DebugString() const
{
  return GetAllocaNode().operation().debug_string();
}

PointsToGraph::DeltaNode::~DeltaNode() noexcept = default;

std::string
PointsToGraph::DeltaNode::DebugString() const
{
  return GetDeltaNode().operation().debug_string();
}

PointsToGraph::LambdaNode::~LambdaNode() noexcept = default;

std::string
PointsToGraph::LambdaNode::DebugString() const
{
  return GetLambdaNode().operation().debug_string();
}

PointsToGraph::MallocNode::~MallocNode() noexcept = default;

std::string
PointsToGraph::MallocNode::DebugString() const
{
  return GetMallocNode().operation().debug_string();
}

PointsToGraph::ImportNode::~ImportNode() noexcept = default;

std::string
PointsToGraph::ImportNode::DebugString() const
{
  auto port = util::AssertedCast<const impport>(&GetArgument().port());
  return port->name();
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

}
