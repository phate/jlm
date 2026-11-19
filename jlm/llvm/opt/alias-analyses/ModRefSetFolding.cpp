/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/ModRefSetFolding.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/rvsdg/node.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

namespace jlm::llvm::aa
{

class DAGGraph
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

  Node *
  addNode(const rvsdg::Node & rvsdgNode, const ModRefSet & modRefSet)
  {
    auto node = std::make_unique<Node>(rvsdgNode, modRefSet);
    nodes_.push_back(std::move(node));
    return nodes_.back().get();
  }

  void
  addEdge(Node & from, Node & to)
  {
    edges_[&from] = &to;
  }

private:
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<Node *, Node *> edges_;
};

}
