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

PointsToGraph::RegisterSetNodeRange
PointsToGraph::RegisterSetNodes()
{
  return { RegisterSetNodeIterator(RegisterSetNodes_.begin()),
           RegisterSetNodeIterator(RegisterSetNodes_.end()) };
}

PointsToGraph::RegisterSetNodeConstRange
PointsToGraph::RegisterSetNodes() const
{
  return { RegisterSetNodeConstIterator(RegisterSetNodes_.begin()),
           RegisterSetNodeConstIterator(RegisterSetNodes_.end()) };
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
  std::string debugString;
  for (auto output : outputs.Items())
  {
    debugString += CreateDotString(*output);
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
