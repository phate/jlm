/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/ipgraph.hpp>

namespace jlm::llvm
{

/* edge */

cfg_edge::cfg_edge(cfg_node * source, cfg_node * sink, size_t index) noexcept
    : source_(source),
      sink_(sink),
      index_(index)
{}

void
cfg_edge::divert(cfg_node * new_sink)
{
  if (sink_ == new_sink)
    return;

  sink_->inedges_.erase(this);
  sink_ = new_sink;
  new_sink->inedges_.insert(this);
}

BasicBlock *
cfg_edge::split()
{
  auto sink = sink_;
  auto bb = BasicBlock::create(source_->cfg());
  divert(bb);
  bb->add_outedge(sink);
  return bb;
}

/* node */

cfg_node::~cfg_node()
{}

size_t
cfg_node::NumOutEdges() const noexcept
{
  return outedges_.size();
}

void
cfg_node::remove_inedges()
{
  while (inedges_.size() != 0)
  {
    cfg_edge * edge = *inedges_.begin();
    JLM_ASSERT(edge->sink() == this);
    edge->source()->remove_outedge(edge->index());
  }
}

size_t
cfg_node::NumInEdges() const noexcept
{
  return inedges_.size();
}

bool
cfg_node::no_predecessor() const noexcept
{
  return NumInEdges() == 0;
}

bool
cfg_node::single_predecessor() const noexcept
{
  if (NumInEdges() == 0)
    return false;

  for (auto i = inedges_.begin(); i != inedges_.end(); i++)
  {
    JLM_ASSERT((*i)->sink() == this);
    if ((*i)->source() != (*inedges_.begin())->source())
      return false;
  }

  return true;
}

bool
cfg_node::no_successor() const noexcept
{
  return NumOutEdges() == 0;
}

bool
cfg_node::single_successor() const noexcept
{
  if (NumOutEdges() == 0)
    return false;

  for (auto & edge : OutEdges())
  {
    JLM_ASSERT(edge.source() == this);
    if (edge.sink() != OutEdge(0)->sink())
      return false;
  }

  return true;
}

bool
cfg_node::has_selfloop_edge() const noexcept
{
  for (auto & edge : OutEdges())
  {
    if (edge.is_selfloop())
      return true;
  }

  return false;
}

}
