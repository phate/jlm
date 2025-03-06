/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <algorithm>
#include <deque>
#include <unordered_map>

namespace jlm::llvm
{

/* argument */

argument::~argument()
{}

/* cfg entry node */

entry_node::~entry_node()
{}

/* cfg exit node */

exit_node::~exit_node()
{}

/* cfg */

cfg::cfg(ipgraph_module & im)
    : module_(im)
{
  entry_ = std::unique_ptr<entry_node>(new entry_node(*this));
  exit_ = std::unique_ptr<exit_node>(new exit_node(*this));
  entry_->add_outedge(exit_.get());
}

cfg::iterator
cfg::remove_node(cfg::iterator & nodeit)
{
  auto & cfg = nodeit->cfg();

  for (auto & inedge : nodeit->InEdges())
  {
    if (inedge.source() != nodeit.node())
      throw util::error("cannot remove node. It has still incoming edges.");
  }

  nodeit->remove_outedges();
  std::unique_ptr<basic_block> tmp(nodeit.node());
  auto rit = iterator(std::next(cfg.nodes_.find(tmp)));
  cfg.nodes_.erase(tmp);
  tmp.release();
  return rit;
}

cfg::iterator
cfg::remove_node(basic_block * bb)
{
  auto & cfg = bb->cfg();

  auto it = cfg.find_node(bb);
  return remove_node(it);
}

std::string
cfg::ToAscii(const cfg & controlFlowGraph)
{
  std::string str;
  auto nodes = breadth_first(controlFlowGraph);
  auto labels = CreateLabels(nodes);
  for (const auto & node : nodes)
  {
    str += labels.at(node) + ":";
    str += (is<basic_block>(node) ? "\n" : " ");

    if (auto entryNode = dynamic_cast<const entry_node *>(node))
    {
      str += ToAscii(*entryNode);
    }
    else if (auto exitNode = dynamic_cast<const exit_node *>(node))
    {
      str += ToAscii(*exitNode);
    }
    else if (auto basicBlock = dynamic_cast<const basic_block *>(node))
    {
      str += ToAscii(*basicBlock, labels);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled control flow graph node type!");
    }
  }

  return str;
}

std::string
cfg::ToAscii(const entry_node & entryNode)
{
  std::string str;
  for (size_t n = 0; n < entryNode.narguments(); n++)
  {
    str += entryNode.argument(n)->debug_string() + " ";
  }

  return str + "\n";
}

std::string
cfg::ToAscii(const exit_node & exitNode)
{
  std::string str;
  for (size_t n = 0; n < exitNode.nresults(); n++)
  {
    str += exitNode.result(n)->debug_string() + " ";
  }

  return str;
}

std::string
cfg::ToAscii(
    const basic_block & basicBlock,
    const std::unordered_map<cfg_node *, std::string> & labels)
{
  auto & threeAddressCodes = basicBlock.tacs();

  std::string str;
  for (const auto & tac : threeAddressCodes)
  {
    str += "\t" + tac::ToAscii(*tac);
    if (tac != threeAddressCodes.last())
      str += "\n";
  }

  if (threeAddressCodes.last())
  {
    if (is<BranchOperation>(threeAddressCodes.last()->operation()))
      str += " " + CreateTargets(basicBlock, labels);
    else
      str += "\n\t" + CreateTargets(basicBlock, labels);
  }
  else
  {
    str += "\t" + CreateTargets(basicBlock, labels);
  }

  return str + "\n";
}

std::string
cfg::CreateTargets(
    const cfg_node & node,
    const std::unordered_map<cfg_node *, std::string> & labels)
{
  size_t n = 0;
  std::string str("[");
  for (auto & outedge : node.OutEdges())
  {
    str += labels.at(outedge.sink());
    if (n != node.NumOutEdges() - 1)
      str += ", ";
  }
  str += "]";

  return str;
}

std::unordered_map<cfg_node *, std::string>
cfg::CreateLabels(const std::vector<cfg_node *> & nodes)
{
  std::unordered_map<cfg_node *, std::string> map;
  for (size_t n = 0; n < nodes.size(); n++)
  {
    auto node = nodes[n];
    if (is<entry_node>(node))
    {
      map[node] = "entry";
    }
    else if (is<exit_node>(node))
    {
      map[node] = "exit";
    }
    else if (is<basic_block>(node))
    {
      map[node] = util::strfmt("bb", n);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled control flow graph node type!");
    }
  }

  return map;
}

/* supporting functions */

std::vector<cfg_node *>
postorder(const llvm::cfg & cfg)
{
  JLM_ASSERT(is_closed(cfg));

  std::function<void(cfg_node *, std::unordered_set<cfg_node *> &, std::vector<cfg_node *> &)>
      traverse = [&](cfg_node * node,
                     std::unordered_set<cfg_node *> & visited,
                     std::vector<cfg_node *> & nodes)
  {
    visited.insert(node);
    for (size_t n = 0; n < node->NumOutEdges(); n++)
    {
      auto edge = node->OutEdge(n);
      if (visited.find(edge->sink()) == visited.end())
        traverse(edge->sink(), visited, nodes);
    }

    nodes.push_back(node);
  };

  std::vector<cfg_node *> nodes;
  std::unordered_set<cfg_node *> visited;
  traverse(cfg.entry(), visited, nodes);

  return nodes;
}

std::vector<cfg_node *>
reverse_postorder(const llvm::cfg & cfg)
{
  auto nodes = postorder(cfg);
  std::reverse(nodes.begin(), nodes.end());
  return nodes;
}

std::vector<cfg_node *>
breadth_first(const llvm::cfg & cfg)
{
  std::deque<cfg_node *> next({ cfg.entry() });
  std::vector<cfg_node *> nodes({ cfg.entry() });
  std::unordered_set<cfg_node *> visited({ cfg.entry() });
  while (!next.empty())
  {
    auto node = next.front();
    next.pop_front();

    for (auto & outedge : node->OutEdges())
    {
      if (visited.find(outedge.sink()) == visited.end())
      {
        visited.insert(outedge.sink());
        next.push_back(outedge.sink());
        nodes.push_back(outedge.sink());
      }
    }
  }

  return nodes;
}

size_t
ntacs(const llvm::cfg & cfg)
{
  size_t ntacs = 0;
  for (auto & node : cfg)
  {
    if (auto bb = dynamic_cast<const basic_block *>(&node))
      ntacs += bb->ntacs();
  }

  return ntacs;
}

}
