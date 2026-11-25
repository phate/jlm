/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSetFolding.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/util/iterator_range.hpp>

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/rvsdg/MatchVariant.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace jlm::llvm::aa
{

MemoryStateGraph::MemoryStateGraph(const rvsdg::Region & region, const ModRefSet & modRefSet)
    : region_(&region)
{
  entryNode_ = &addNode(*region.node(), modRefSet);
  exitNode_ = &addNode(*region.node(), modRefSet);
}

const MemoryStateGraph::Node &
MemoryStateGraph::getEntryNode() const noexcept
{
  return *entryNode_;
}

const MemoryStateGraph::Node &
MemoryStateGraph::getExitNode() const noexcept
{
  return *exitNode_;
}

const rvsdg::Region &
MemoryStateGraph::getRegion() const noexcept
{
  return *region_;
}

MemoryStateGraph::NodeRange
MemoryStateGraph::nodes() const noexcept
{
  return { NodeIterator(nodes_.begin()), NodeIterator(nodes_.end()) };
}

MemoryStateGraph::EdgeRange
MemoryStateGraph::edges() const noexcept
{
  return { edges_.begin(), edges_.end() };
}

MemoryStateGraph::Node &
MemoryStateGraph::addNode(const rvsdg::Node & rvsdgNode, const ModRefSet & modRefSet)
{
  auto node = std::make_unique<Node>(rvsdgNode, modRefSet);
  nodes_.push_back(std::move(node));
  return *nodes_.back().get();
}

void
MemoryStateGraph::addEdge(const Node & from, const Node & to)
{
  edges_[&from] = &to;
}

std::unique_ptr<MemoryStateGraph>
MemoryStateGraph::create(const rvsdg::Region & region, const ModRefSet & modRefSet)
{
  return std::make_unique<MemoryStateGraph>(region, modRefSet);
}

static bool
hasMemoryState(const rvsdg::Node & node)
{
  for (const auto & input : node.Inputs())
  {
    if (rvsdg::is<MemoryStateType>(input.Type()))
      return true;
  }
  for (const auto & output : node.Outputs())
  {
    if (rvsdg::is<MemoryStateType>(output.Type()))
      return true;
  }
  return false;
}

static const ModRefSet &
getModRefSet(const rvsdg::StructuralNode & structuralNode, const ModRefSummary & modRefSummary)
{
  return rvsdg::MatchTypeOrFail(
      structuralNode,
      [&modRefSummary](const rvsdg::GammaNode & gammaNode) -> const ModRefSet &
      {
        auto & entryModRefSet = modRefSummary.GetGammaEntryModRef(gammaNode);
        auto & exitModRefSet = modRefSummary.GetGammaExitModRef(gammaNode);
        JLM_ASSERT(entryModRefSet == exitModRefSet);
        return entryModRefSet;
      },
      [&modRefSummary](const rvsdg::ThetaNode & thetaNode) -> const ModRefSet &
      {
        return modRefSummary.GetThetaModRef(thetaNode);
      },
      [&modRefSummary](const rvsdg::LambdaNode & lambdaNode) -> const ModRefSet &
      {
        auto & entryModRefSet = modRefSummary.GetLambdaEntryModRef(lambdaNode);
        auto & exitModRefSet = modRefSummary.GetLambdaExitModRef(lambdaNode);
        JLM_ASSERT(entryModRefSet == exitModRefSet);
        return entryModRefSet;
      });
}

static std::unordered_map<const rvsdg::Node *, MemoryStateGraph::Node *>
createNodes(
    const rvsdg::Region & region,
    const ModRefSummary & modRefSummary,
    MemoryStateGraph & memoryStateGraph)
{
  std::unordered_map<const rvsdg::Node *, MemoryStateGraph::Node *> nodeMapping;

  for (const auto & rvsdgNode : region.Nodes())
  {
    MemoryStateGraph::Node * memoryStateNode = rvsdg::MatchTypeWithDefault(
        rvsdgNode,
        [&](const rvsdg::LambdaNode & lambdaNode)
        {
          auto & modRefSet = getModRefSet(lambdaNode, modRefSummary);
          return &memoryStateGraph.addNode(lambdaNode, modRefSet);
        },
        [&](const rvsdg::GammaNode & gammaNode)
        {
          auto & modRefSet = getModRefSet(gammaNode, modRefSummary);
          return &memoryStateGraph.addNode(gammaNode, modRefSet);
        },
        [&](const rvsdg::ThetaNode & thetaNode)
        {
          return &memoryStateGraph.addNode(thetaNode, modRefSummary.GetThetaModRef(thetaNode));
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          return rvsdg::MatchTypeWithDefault(
              rvsdgNode.GetOperation(),
              [&](const LoadOperation &)
              {
                return &memoryStateGraph.addNode(
                    simpleNode,
                    modRefSummary.GetSimpleNodeModRef(simpleNode));
              },
              [&](const StoreOperation &)
              {
                return &memoryStateGraph.addNode(
                    simpleNode,
                    modRefSummary.GetSimpleNodeModRef(simpleNode));
              },
              [&](const MemCpyOperation &)
              {
                return &memoryStateGraph.addNode(
                    simpleNode,
                    modRefSummary.GetSimpleNodeModRef(simpleNode));
              },
              [&](const MemMoveOperation &)
              {
                return &memoryStateGraph.addNode(
                    simpleNode,
                    modRefSummary.GetSimpleNodeModRef(simpleNode));
              },
              [&](const MemSetOperation &)
              {
                return &memoryStateGraph.addNode(
                    simpleNode,
                    modRefSummary.GetSimpleNodeModRef(simpleNode));
              },
              [&](const AllocaOperation &)
              {
                return &memoryStateGraph.addNode(
                    simpleNode,
                    modRefSummary.GetSimpleNodeModRef(simpleNode));
              },
              [&](const MallocOperation &)
              {
                return &memoryStateGraph.addNode(
                    simpleNode,
                    modRefSummary.GetSimpleNodeModRef(simpleNode));
              },
              [&simpleNode]()
              {
                JLM_ASSERT(!hasMemoryState(simpleNode));
                return nullptr;
              });
        },
        []()
        {
          return nullptr;
        });

    if (memoryStateNode)
    {
      nodeMapping[&rvsdgNode] = memoryStateNode;
    }
  }

  return nodeMapping;
}

static void
createEdges(
    MemoryStateGraph & memoryStateGraph,
    const std::unordered_map<const rvsdg::Node *, MemoryStateGraph::Node *> & nodeMapping)
{
  auto addEdge = [&memoryStateGraph,
                  &nodeMapping](const rvsdg::Output & origin, const MemoryStateGraph::Node & toNode)
  {
    auto fromNode = rvsdg::MatchVariant(
        origin.GetOwner(),
        [&memoryStateGraph](const rvsdg::Region *) -> const MemoryStateGraph::Node *
        {
          return &memoryStateGraph.getEntryNode();
        },
        [&nodeMapping](const rvsdg::Node * node) -> const MemoryStateGraph::Node *
        {
          return nodeMapping.at(node);
        });

    memoryStateGraph.addEdge(*fromNode, toNode);
  };

  for (auto [rvsdgNode, memoryStateNode] : nodeMapping)
  {
    for (auto & input : rvsdgNode->Inputs())
    {
      if (rvsdg::is<MemoryStateType>(input.Type()))
      {
        addEdge(*input.origin(), *memoryStateNode);
      }
    }
  }

  for (auto & result : memoryStateGraph.getRegion().Results())
  {
    if (rvsdg::is<MemoryStateType>(result->Type()))
    {
      addEdge(*result->origin(), memoryStateGraph.getExitNode());
    }
  }
}

std::unique_ptr<MemoryStateGraph>
constructMemoryStateGraph(const rvsdg::Region & region, const ModRefSummary & modRefSummary)
{
  auto & modRefSet = getModRefSet(*region.node(), modRefSummary);
  auto memoryStateGraph = MemoryStateGraph::create(region, modRefSet);

  const auto nodeMapping = createNodes(region, modRefSummary, *memoryStateGraph);
  createEdges(*memoryStateGraph, nodeMapping);

  return memoryStateGraph;
}

static std::string
formatModRefSet(const ModRefSet & modRefSet)
{
  std::vector<PointsToGraph::NodeIndex> indices;
  indices.reserve(modRefSet.getModRefNodes().size());
  for (const auto & [idx, _] : modRefSet.getModRefNodes())
  {
    indices.push_back(idx);
  }
  std::sort(indices.begin(), indices.end());

  std::string result;
  for (size_t i = 0; i < indices.size(); ++i)
  {
    if (i > 0)
      result += ",";
    result += std::to_string(indices[i]);
  }

  return result;
}

static std::string
escapeDotString(const std::string & s)
{
  std::string result;
  for (char c : s)
  {
    if (c == '"')
      result += "\\\"";
    else if (c == '\\')
      result += "\\\\";
    else if (c == '\n')
      result += "\\n";
    else
      result += c;
  }
  return result;
}

static std::string
convertNodeIdToDot(const MemoryStateGraph & memoryStateGraph, const MemoryStateGraph::Node & node)
{
  std::string prefix;
  if (&node == &memoryStateGraph.getEntryNode())
    prefix = "entry_";
  else if (&node == &memoryStateGraph.getExitNode())
    prefix = "exit_";

  return "\"" + prefix + "r" + std::to_string(node.rvsdgNode.region()->getRegionId()) + "_n"
       + std::to_string(node.rvsdgNode.GetNodeId()) + "\"";
}

static std::string
convertEdgeToDot(
    const MemoryStateGraph & memoryStateGraph,
    const MemoryStateGraph::Node & from,
    const MemoryStateGraph::Node & to)
{
  return convertNodeIdToDot(memoryStateGraph, from) + " -> "
       + convertNodeIdToDot(memoryStateGraph, to) + ";";
}

static std::string
convertNodeToDot(const MemoryStateGraph & memoryStateGraph, const MemoryStateGraph::Node & node)
{
  auto modRefSetStr = formatModRefSet(node.modRefSet);

  std::string escapedLabel = escapeDotString(node.rvsdgNode.DebugString());
  std::string nodeStr = convertNodeIdToDot(memoryStateGraph, node) + " [members=\"" + modRefSetStr
                      + "\", label=\"" + escapedLabel + "\\n{" + modRefSetStr + "}\"];";
  return nodeStr;
}

std::string
toDot(const MemoryStateGraph & memoryStateGraph)
{
  std::stringstream sstr;

  std::string graphName = "REGION_" + std::to_string(memoryStateGraph.getRegion().getRegionId());
  sstr << "digraph \"" << escapeDotString(graphName) << "\" {\n";
  for (auto & node : memoryStateGraph.nodes())
  {
    sstr << "  " << convertNodeToDot(memoryStateGraph, node) << "\n";
  }
  for (const auto & [from, to] : memoryStateGraph.edges())
  {
    sstr << "  " << convertEdgeToDot(memoryStateGraph, *from, *to) << "\n";
  }
  sstr << "}\n";

  return sstr.str();
}

}
