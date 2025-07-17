/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/ipgraph.hpp>

#include <algorithm>

/* Tarjan's SCC algorithm */

static void
strongconnect(
    const jlm::llvm::InterProceduralGraphNode * node,
    std::unordered_map<const jlm::llvm::InterProceduralGraphNode *, std::pair<size_t, size_t>> &
        map,
    std::vector<const jlm::llvm::InterProceduralGraphNode *> & node_stack,
    size_t & index,
    std::vector<std::unordered_set<const jlm::llvm::InterProceduralGraphNode *>> & sccs)
{
  map.emplace(node, std::make_pair(index, index));
  node_stack.push_back(node);
  index++;

  for (auto callee : *node)
  {
    if (map.find(callee) == map.end())
    {
      /* successor has not been visited yet; recurse on it */
      strongconnect(callee, map, node_stack, index, sccs);
      map[node].second = std::min(map[node].second, map[callee].second);
    }
    else if (std::find(node_stack.begin(), node_stack.end(), callee) != node_stack.end())
    {
      /* successor is in stack and hence in the current SCC */
      map[node].second = std::min(map[node].second, map[callee].first);
    }
  }

  if (map[node].second == map[node].first)
  {
    std::unordered_set<const jlm::llvm::InterProceduralGraphNode *> scc;
    const jlm::llvm::InterProceduralGraphNode * w = nullptr;
    do
    {
      w = node_stack.back();
      node_stack.pop_back();
      scc.insert(w);
    } while (w != node);

    sccs.push_back(scc);
  }
}

namespace jlm::llvm
{

void
InterProceduralGraph::add_node(std::unique_ptr<InterProceduralGraphNode> node)
{
  nodes_.push_back(std::move(node));
}

std::vector<std::unordered_set<const InterProceduralGraphNode *>>
InterProceduralGraph::find_sccs() const
{
  std::vector<std::unordered_set<const InterProceduralGraphNode *>> sccs;

  std::unordered_map<const InterProceduralGraphNode *, std::pair<size_t, size_t>> map;
  std::vector<const InterProceduralGraphNode *> node_stack;
  size_t index = 0;

  for (auto & node : *this)
  {
    if (map.find(&node) == map.end())
      strongconnect(&node, map, node_stack, index, sccs);
  }

  return sccs;
}

const InterProceduralGraphNode *
InterProceduralGraph::find(const std::string & name) const noexcept
{
  for (auto & node : nodes_)
  {
    if (node->name() == name)
      return node.get();
  }

  return nullptr;
}

InterProceduralGraphNode::~InterProceduralGraphNode() noexcept = default;

FunctionNode::~FunctionNode() noexcept = default;

const std::string &
FunctionNode::name() const noexcept
{
  return name_;
}

const jlm::rvsdg::Type &
FunctionNode::type() const noexcept
{
  return *FunctionType_;
}

std::shared_ptr<const jlm::rvsdg::Type>
FunctionNode::Type() const
{
  return FunctionType_;
}

const llvm::linkage &
FunctionNode::linkage() const noexcept
{
  return linkage_;
}

bool
FunctionNode::hasBody() const noexcept
{
  return cfg() != nullptr;
}

void
FunctionNode::add_cfg(std::unique_ptr<ControlFlowGraph> cfg)
{
  if (cfg->fcttype() != fcttype())
    throw util::error("CFG does not match the function node's type.");

  cfg_ = std::move(cfg);
}

fctvariable::~fctvariable() noexcept = default;

DataNode::~DataNode() noexcept = default;

const std::string &
DataNode::name() const noexcept
{
  return name_;
}

const PointerType &
DataNode::type() const noexcept
{
  static PointerType pointerType;
  return pointerType;
}

std::shared_ptr<const rvsdg::Type>
DataNode::Type() const
{
  return PointerType::Create();
}

const llvm::linkage &
DataNode::linkage() const noexcept
{
  return linkage_;
}

bool
DataNode::hasBody() const noexcept
{
  return initialization() != nullptr;
}

}
