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

  for (auto & inedge : nodeit->inedges())
  {
    if (inedge->source() != nodeit.node())
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
  for (const auto & node : nodes)
  {
    str += CreateLabel(*node) + ":";
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
      str += ToAscii(*basicBlock);
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
cfg::ToAscii(const basic_block & basicBlock)
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
    if (is<branch_op>(threeAddressCodes.last()->operation()))
      str += " " + CreateTargets(basicBlock);
    else
      str += "\n\t" + CreateTargets(basicBlock);
  }
  else
  {
    str += "\t" + CreateTargets(basicBlock);
  }

  return str + "\n";
}

std::string
cfg::CreateTargets(const cfg_node & node)
{
  size_t n = 0;
  std::string str("[");
  for (auto it = node.begin_outedges(); it != node.end_outedges(); it++, n++)
  {
    str += CreateLabel(*it->sink());
    if (n != node.noutedges() - 1)
      str += ", ";
  }
  str += "]";

  return str;
}

std::string
cfg::CreateLabel(const cfg_node & node)
{
  return util::strfmt(&node);
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
    for (size_t n = 0; n < node->noutedges(); n++)
    {
      auto edge = node->outedge(n);
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

    for (auto it = node->begin_outedges(); it != node->end_outedges(); it++)
    {
      if (visited.find(it->sink()) == visited.end())
      {
        visited.insert(it->sink());
        next.push_back(it->sink());
        nodes.push_back(it->sink());
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
