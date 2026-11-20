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

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace jlm::llvm::aa
{

class MemoryStateGraph
{
public:
  struct Node
  {
    Node(const rvsdg::Node & rvsdgNode, const ModRefSet & modRefSet)
        : rvsdgNode(rvsdgNode),
          modRefSet(modRefSet)
    {}

    const rvsdg::Node & rvsdgNode;
    const ModRefSet & modRefSet;
  };

  Node &
  addNode(const rvsdg::Node & rvsdgNode, const ModRefSet & modRefSet)
  {
    auto node = std::make_unique<Node>(rvsdgNode, modRefSet);
    nodes_.push_back(std::move(node));
    return *nodes_.back().get();
  }

  void
  addEdge(Node & from, Node & to)
  {
    edges_[&from] = &to;
  }

  static std::unique_ptr<MemoryStateGraph>
  create()
  {
    return std::make_unique<MemoryStateGraph>();
  }

private:
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<Node *, Node *> edges_;
};

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

static std::unordered_map<const rvsdg::Node *, MemoryStateGraph::Node *>
createMemoryStateNodes(
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
          return &memoryStateGraph.addNode(
              lambdaNode,
              modRefSummary.GetLambdaExitModRef(lambdaNode));
        },
        [&](const rvsdg::GammaNode & gammaNode)
        {
          auto & entryModRefSet = modRefSummary.GetGammaEntryModRef(gammaNode);
          auto & exitModRefSet = modRefSummary.GetGammaExitModRef(gammaNode);
          JLM_ASSERT(entryModRefSet == exitModRefSet);

          return &memoryStateGraph.addNode(gammaNode, entryModRefSet);
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
addEdges(
    MemoryStateGraph & memoryStateGraph,
    const std::unordered_map<const rvsdg::Node *, MemoryStateGraph::Node *> & nodeMapping)
{
  for (auto [rvsdgNode, memoryStateNode] : nodeMapping)
  {
    for (auto & input : rvsdgNode->Inputs())
    {
      if (rvsdg::is<MemoryStateType>(input.Type()))
      {
        auto fromNode = nodeMapping.at(input.origin());
      }
    }
  }
}

std::unique_ptr<MemoryStateGraph>
constructMemoryStateGraph(const rvsdg::Region & region, const ModRefSummary & modRefSummary)
{
  auto memoryStateGraph = MemoryStateGraph::create();

  auto entryNode = memoryStateGraph->addNode(nullptr, );
  auto exitNode = memoryStateGraph->addNode();

  const auto nodeMapping = createMemoryStateNodes(region, modRefSummary, *memoryStateGraph);
  addEdges(*memoryStateGraph, nodeMapping);

  return memoryStateGraph;
}

}
