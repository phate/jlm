/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/traverser.hpp>

using namespace std::placeholders;

/* top down traverser */

namespace jlm::rvsdg
{

TopDownTraverser::~TopDownTraverser() noexcept = default;

TopDownTraverser::TopDownTraverser(Region * region)
    : region_(*region)
{
  for (auto & node : region->TopNodes())
    tracker_.set_nodestate(&node, traversal_nodestate::frontier);

  for (size_t n = 0; n < region->narguments(); n++)
  {
    auto argument = region->argument(n);
    for (const auto & user : argument->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
      {
        if (!predecessors_visited(node))
          continue;

        tracker_.set_nodestate(node, traversal_nodestate::frontier);
      }
    }
  }

  callbacks_.push_back(on_node_create.connect(std::bind(&TopDownTraverser::node_create, this, _1)));
  callbacks_.push_back(
      on_input_change.connect(std::bind(&TopDownTraverser::input_change, this, _1, _2, _3)));
}

bool
TopDownTraverser::predecessors_visited(const Node * node) noexcept
{
  for (size_t n = 0; n < node->ninputs(); n++)
  {
    auto predecessor = TryGetOwnerNode<Node>(*node->input(n)->origin());
    if (!predecessor)
      continue;

    if (tracker_.get_nodestate(predecessor) != traversal_nodestate::behind)
      return false;
  }

  return true;
}

Node *
TopDownTraverser::next()
{
  Node * node = tracker_.peek();
  if (!node)
    return nullptr;

  tracker_.set_nodestate(node, traversal_nodestate::behind);
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    for (const auto & user : node->output(n)->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
      {
        if (!predecessors_visited(node))
        {
          continue;
        }
        if (tracker_.get_nodestate(node) == traversal_nodestate::ahead)
        {
          tracker_.set_nodestate(node, traversal_nodestate::frontier);
        }
      }
    }
  }

  return node;
}

void
TopDownTraverser::node_create(Node * node)
{
  if (node->region() != &region_)
    return;

  if (predecessors_visited(node))
    tracker_.set_nodestate(node, traversal_nodestate::behind);
  else
    tracker_.set_nodestate(node, traversal_nodestate::frontier);
}

void
TopDownTraverser::input_change(Input * in, Output *, Output *)
{
  if (in->region() != &region_)
    return;

  auto node = TryGetOwnerNode<Node>(*in);
  if (!node)
    return;

  auto state = tracker_.get_nodestate(node);

  // ignore nodes that have been traversed already, or that are already
  // marked for later traversal
  if (state != traversal_nodestate::ahead)
    return;

  // node may just have become eligible for visiting, check
  if (predecessors_visited(node))
    tracker_.set_nodestate(node, traversal_nodestate::frontier);
}

TopDownConstTraverser::~TopDownConstTraverser() noexcept = default;

TopDownConstTraverser::TopDownConstTraverser(const Region & region)
{
  for (auto & node : region.TopNodes())
    tracker_.set_nodestate(&node, traversal_nodestate::frontier);

  for (size_t n = 0; n < region.narguments(); n++)
  {
    auto argument = region.argument(n);
    for (const auto & user : argument->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
      {
        if (!predecessors_visited(node))
          continue;

        tracker_.set_nodestate(node, traversal_nodestate::frontier);
      }
    }
  }
}

const Node *
TopDownConstTraverser::next()
{
  const Node * node = tracker_.peek();
  if (!node)
    return nullptr;

  tracker_.set_nodestate(node, traversal_nodestate::behind);
  for (size_t n = 0; n < node->noutputs(); n++)
  {
    for (const auto & user : node->output(n)->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
      {
        if (!predecessors_visited(node))
        {
          continue;
        }
        if (tracker_.get_nodestate(node) == traversal_nodestate::ahead)
        {
          tracker_.set_nodestate(node, traversal_nodestate::frontier);
        }
      }
    }
  }

  return node;
}

bool
TopDownConstTraverser::predecessors_visited(const Node * node) noexcept
{
  for (size_t n = 0; n < node->ninputs(); n++)
  {
    auto predecessor = TryGetOwnerNode<Node>(*node->input(n)->origin());
    if (!predecessor)
      continue;

    if (tracker_.get_nodestate(predecessor) != traversal_nodestate::behind)
      return false;
  }

  return true;
}

static bool
HasSuccessors(const Node & node)
{
  for (size_t n = 0; n < node.noutputs(); n++)
  {
    const auto output = node.output(n);
    for (const auto & user : output->Users())
    {
      if (TryGetOwnerNode<Node>(user))
        return true;
    }
  }

  return false;
}

BottomUpTraverser::~BottomUpTraverser() noexcept = default;

BottomUpTraverser::BottomUpTraverser(Region * region, bool revisit)
    : region_(*region),
      new_node_state_(revisit ? traversal_nodestate::frontier : traversal_nodestate::behind)
{
  for (auto & bottomNode : region->BottomNodes())
  {
    tracker_.set_nodestate(&bottomNode, traversal_nodestate::frontier);
  }

  for (size_t n = 0; n < region->nresults(); n++)
  {
    const auto node = TryGetOwnerNode<Node>(*region->result(n)->origin());
    if (node && !HasSuccessors(*node))
      tracker_.set_nodestate(node, traversal_nodestate::frontier);
  }

  callbacks_.push_back(
      on_node_create.connect(std::bind(&BottomUpTraverser::node_create, this, _1)));
  callbacks_.push_back(
      on_node_destroy.connect(std::bind(&BottomUpTraverser::node_destroy, this, _1)));
  callbacks_.push_back(
      on_input_change.connect(std::bind(&BottomUpTraverser::input_change, this, _1, _2, _3)));
}

Node *
BottomUpTraverser::next()
{
  auto node = tracker_.peek();
  if (!node)
    return nullptr;

  tracker_.set_nodestate(node, traversal_nodestate::behind);
  for (size_t n = 0; n < node->ninputs(); n++)
  {
    auto producer = TryGetOwnerNode<Node>(*node->input(n)->origin());
    if (producer && tracker_.get_nodestate(producer) == traversal_nodestate::ahead)
      tracker_.set_nodestate(producer, traversal_nodestate::frontier);
  }
  return node;
}

void
BottomUpTraverser::node_create(Node * node)
{
  if (node->region() != &region_)
    return;

  tracker_.set_nodestate(node, new_node_state_);
}

void
BottomUpTraverser::node_destroy(Node * node)
{
  if (node->region() != &region_)
    return;

  for (size_t n = 0; n < node->ninputs(); n++)
  {
    auto producer = TryGetOwnerNode<Node>(*node->input(n)->origin());
    if (!producer)
    {
      continue;
    }
    bool successors_visited = true;
    for (const auto & output : producer->Outputs())
    {
      for (auto & user : output.Users())
      {
        auto u_node = TryGetOwnerNode<Node>(user);
        successors_visited =
            successors_visited && tracker_.get_nodestate(u_node) == traversal_nodestate::behind;
      }
    }
    if (!successors_visited)
    {
      continue;
    }
    if (producer && tracker_.get_nodestate(producer) == traversal_nodestate::ahead)
      tracker_.set_nodestate(producer, traversal_nodestate::frontier);
  }
}

void
BottomUpTraverser::input_change(Input * in, Output * old_origin, Output *)
{
  if (in->region() != &region_)
    return;

  if (!TryGetOwnerNode<Node>(*in))
    return;

  auto node = TryGetOwnerNode<Node>(*old_origin);
  if (!node)
    return;

  traversal_nodestate state = tracker_.get_nodestate(node);

  /* ignore nodes that have been traversed already, or that are already
  marked for later traversal */
  if (state != traversal_nodestate::ahead)
    return;

  /* make sure node is visited eventually, might now be visited earlier
  as there (potentially) is one less obstructing node below */
  tracker_.set_nodestate(node, traversal_nodestate::frontier);
}

traversal_nodestate
TraversalTracker::get_nodestate(Node * node)
{
  auto i = states_.find(node);
  return i == states_.end() ? traversal_nodestate::ahead : i->second.state;
}

void
TraversalTracker::set_nodestate(Node * node, traversal_nodestate state)
{
  auto i = states_.find(node);
  if (i == states_.end())
  {
    FrontierList::iterator j = frontier_.end();
    if (state == traversal_nodestate::frontier)
    {
      frontier_.push_back(node);
      j = std::prev(frontier_.end());
    }
    states_.emplace(node, State{ state, j });
  }
  else
  {
    auto old_state = i->second.state;
    if (old_state != state)
    {
      if (old_state == traversal_nodestate::frontier)
      {
        frontier_.erase(i->second.pos);
        i->second.pos = frontier_.end();
      }
      i->second.state = state;
      if (state == traversal_nodestate::frontier)
      {
        frontier_.push_back(node);
        i->second.pos = std::prev(frontier_.end());
      }
    }
  }
}

Node *
TraversalTracker::peek()
{
  return frontier_.empty() ? nullptr : frontier_.front();
}

traversal_nodestate
TraversalConstTracker::get_nodestate(const Node * node)
{
  auto i = states_.find(node);
  return i == states_.end() ? traversal_nodestate::ahead : i->second.state;
}

void
TraversalConstTracker::set_nodestate(const Node * node, traversal_nodestate state)
{
  auto i = states_.find(node);
  if (i == states_.end())
  {
    FrontierList::iterator j = frontier_.end();
    if (state == traversal_nodestate::frontier)
    {
      frontier_.push_back(node);
      j = std::prev(frontier_.end());
    }
    states_.emplace(node, State{ state, j });
  }
  else
  {
    auto old_state = i->second.state;
    if (old_state != state)
    {
      if (old_state == traversal_nodestate::frontier)
      {
        frontier_.erase(i->second.pos);
        i->second.pos = frontier_.end();
      }
      i->second.state = state;
      if (state == traversal_nodestate::frontier)
      {
        frontier_.push_back(node);
        i->second.pos = std::prev(frontier_.end());
      }
    }
  }
}

const Node *
TraversalConstTracker::peek()
{
  return frontier_.empty() ? nullptr : frontier_.front();
}

}
