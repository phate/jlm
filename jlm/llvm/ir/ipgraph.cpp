/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/ipgraph.hpp>

#include <algorithm>

/* Tarjan's SCC algorithm */

static void
strongconnect(
    const jlm::llvm::ipgraph_node * node,
    std::unordered_map<const jlm::llvm::ipgraph_node *, std::pair<size_t, size_t>> & map,
    std::vector<const jlm::llvm::ipgraph_node *> & node_stack,
    size_t & index,
    std::vector<std::unordered_set<const jlm::llvm::ipgraph_node *>> & sccs)
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
    std::unordered_set<const jlm::llvm::ipgraph_node *> scc;
    const jlm::llvm::ipgraph_node * w;
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
InterProceduralGraph::add_node(std::unique_ptr<ipgraph_node> node)
{
  nodes_.push_back(std::move(node));
}

std::vector<std::unordered_set<const ipgraph_node *>>
InterProceduralGraph::find_sccs() const
{
  std::vector<std::unordered_set<const ipgraph_node *>> sccs;

  std::unordered_map<const ipgraph_node *, std::pair<size_t, size_t>> map;
  std::vector<const ipgraph_node *> node_stack;
  size_t index = 0;

  for (auto & node : *this)
  {
    if (map.find(&node) == map.end())
      strongconnect(&node, map, node_stack, index, sccs);
  }

  return sccs;
}

const ipgraph_node *
InterProceduralGraph::find(const std::string & name) const noexcept
{
  for (auto & node : nodes_)
  {
    if (node->name() == name)
      return node.get();
  }

  return nullptr;
}

/* ipgraph node */

ipgraph_node::~ipgraph_node()
{}

/* function node */

function_node::~function_node()
{}

const std::string &
function_node::name() const noexcept
{
  return name_;
}

const jlm::rvsdg::Type &
function_node::type() const noexcept
{
  return *FunctionType_;
}

std::shared_ptr<const jlm::rvsdg::Type>
function_node::Type() const
{
  return FunctionType_;
}

const llvm::linkage &
function_node::linkage() const noexcept
{
  return linkage_;
}

bool
function_node::hasBody() const noexcept
{
  return cfg() != nullptr;
}

void
function_node::add_cfg(std::unique_ptr<llvm::ControlFlowGraph> cfg)
{
  if (cfg->fcttype() != fcttype())
    throw util::error("CFG does not match the function node's type.");

  cfg_ = std::move(cfg);
}

/* function variable */

fctvariable::~fctvariable()
{}

/* data node */

data_node::~data_node()
{}

const std::string &
data_node::name() const noexcept
{
  return name_;
}

const PointerType &
data_node::type() const noexcept
{
  static PointerType pointerType;
  return pointerType;
}

std::shared_ptr<const rvsdg::Type>
data_node::Type() const
{
  return PointerType::Create();
}

const llvm::linkage &
data_node::linkage() const noexcept
{
  return linkage_;
}

bool
data_node::hasBody() const noexcept
{
  return initialization() != nullptr;
}

}
