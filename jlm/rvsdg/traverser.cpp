/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/traverser.hpp>

/*
  How traversers operate:

  Each node needs a number of "activations" depending on edges
  connected to the node (is inputs or outputs, dependent on direction
  of traversal).

  Each edge coming from the region boundaries will provide one activation,
  and each edge outgoing from a node that has been visited already will
  provide one activation. So effectively, both bottom up and top down
  traversers are just a little bit of beancounting edges and activations.
  This is done in such a way that each node and each edge is visited exactly
  once. Thus, the overall time complexity of either traversal is O(#N + #E).
*/
namespace jlm::rvsdg
{

TopDownTraverser::Observer::~Observer() noexcept = default;

TopDownTraverser::Observer::Observer(Region & region, TopDownTraverser & traverser)
    : RegionObserver(region),
      traverser_(traverser)
{}

void
TopDownTraverser::Observer::onNodeCreate(Node * node)
{
  traverser_.onNodeCreate(node);
}

void
TopDownTraverser::Observer::onNodeDestroy(Node * node)
{
  traverser_.onNodeDestroy(node);
}

void
TopDownTraverser::Observer::onInputCreate(Input * input)
{
  traverser_.onInputCreate(input);
}

void
TopDownTraverser::Observer::onInputChange(Input * input, Output * old_origin, Output * new_origin)
{
  traverser_.onInputChange(input, old_origin, new_origin);
}

void
TopDownTraverser::Observer::onInputDestroy(Input * input)
{
  traverser_.onInputDestroy(input);
}

TopDownTraverser::~TopDownTraverser() noexcept = default;

TopDownTraverser::TopDownTraverser(Region * region)
    : observer_(*region, *this)
{
  for (auto & node : region->TopNodes())
  {
    tracker_.checkNodeActivation(&node, node.ninputs());
  }

  for (size_t n = 0; n < region->narguments(); n++)
  {
    auto argument = region->argument(n);
    for (const auto & user : argument->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
      {
        tracker_.incActivationCount(node, node->ninputs());
      }
    }
  }
}

bool
TopDownTraverser::isOutputActivated(const Output * output) const
{
  if (auto pred = TryGetOwnerNode<Node>(*output))
  {
    return tracker_.isNodeVisited(pred);
  }

  return true;
}

void
TopDownTraverser::markVisited(Node * node)
{
  tracker_.checkMarkNodeVisited(node);
  for (const auto & output : node->Outputs())
  {
    for (const auto & user : output.Users())
    {
      if (auto next = TryGetOwnerNode<Node>(user))
      {
        tracker_.incActivationCount(next, next->ninputs());
      }
    }
  }
}

Node *
TopDownTraverser::next()
{
  Node * node = tracker_.peek();
  if (!node)
    return nullptr;

  markVisited(node);

  return node;
}

void
TopDownTraverser::onNodeCreate(Node * node)
{
  for (const auto & input : node->Inputs())
  {
    if (isOutputActivated(input.origin()))
    {
      tracker_.incActivationCount(node, node->ninputs());
    }
  }

  if (node->ninputs() == 0)
    tracker_.checkNodeActivation(node, node->ninputs());

  // If node would end up on frontier (because all predecessors
  // have been visited), mark it as visited instead (we do not
  // want to revisit newly created nodes during topdown traversal).
  tracker_.checkMarkNodeVisited(node);
}

void TopDownTraverser::onNodeDestroy(Node * node)
{
  tracker_.removeNode(node);
}

void
TopDownTraverser::onInputCreate(Input * input)
{
  auto node = TryGetOwnerNode<Node>(*input);
  if (!node)
    return;

  if (isOutputActivated(input->origin()))
  {
    tracker_.incActivationCount(node, node->ninputs());
  }
  else
  {
    tracker_.checkNodeDeactivation(node, node->ninputs());
  }
}

void
TopDownTraverser::onInputChange(Input * in, Output * old_output, Output * new_output)
{
  auto node = TryGetOwnerNode<Node>(*in);
  if (!node)
    return;

  int change = 0;
  if (isOutputActivated(new_output))
  {
    change += 1;
  }
  if (isOutputActivated(old_output))
  {
    change -= 1;
  }

  if (change == 1)
  {
    tracker_.incActivationCount(node, node->ninputs());
  }
  else if (change == -1)
  {
    tracker_.decActivationCount(node, node->ninputs());
  }
}

void
TopDownTraverser::onInputDestroy(Input * input)
{
  auto node = TryGetOwnerNode<Node>(*input);
  if (!node)
    return;

  if (isOutputActivated(input->origin()))
  {
    tracker_.decActivationCount(node, 0);
  }
  else
  {
    tracker_.checkNodeActivation(node, node->ninputs() - 1);
  }
}

BottomUpTraverser::Observer::~Observer() noexcept = default;

BottomUpTraverser::Observer::Observer(Region & region, BottomUpTraverser & traverser)
    : RegionObserver(region),
      traverser_(traverser)
{}

void
BottomUpTraverser::Observer::onNodeCreate(Node * node)
{
  traverser_.onNodeCreate(node);
}

void
BottomUpTraverser::Observer::onNodeDestroy(Node * node)
{
  traverser_.onNodeDestroy(node);
}

void
BottomUpTraverser::Observer::onInputCreate(Input * input)
{
  traverser_.onInputCreate(input);
}

void
BottomUpTraverser::Observer::onInputChange(Input * input, Output * old_origin, Output * new_origin)
{
  traverser_.onInputChange(input, old_origin, new_origin);
}

void
BottomUpTraverser::Observer::onInputDestroy(Input * input)
{
  traverser_.onInputDestroy(input);
}

BottomUpTraverser::~BottomUpTraverser() noexcept = default;

BottomUpTraverser::BottomUpTraverser(Region * region)
    : observer_(*region, *this)
{
  for (auto & node : region->BottomNodes())
    tracker_.checkNodeActivation(&node, node.NumSuccessors());

  for (size_t n = 0; n < region->nresults(); n++)
  {
    if (auto node = TryGetOwnerNode<Node>(*region->result(n)->origin()))
      tracker_.incActivationCount(node, node->NumSuccessors());
  }
}

Node *
BottomUpTraverser::next()
{
  auto node = tracker_.peek();
  if (!node)
    return nullptr;

  markVisited(node);
  return node;
}

bool
BottomUpTraverser::isInputActivated(const Input * input) const
{
  if (auto node = TryGetOwnerNode<Node>(*input))
  {
    return tracker_.isNodeVisited(node);
  }
  else
  {
    return true;
  }
}

void
BottomUpTraverser::markVisited(Node * node)
{
  tracker_.checkMarkNodeVisited(node);
  for (const auto & input : node->Inputs())
  {
    if (auto pred = TryGetOwnerNode<Node>(*input.origin()))
    {
      tracker_.incActivationCount(pred, pred->NumSuccessors());
    }
  }
}

void
BottomUpTraverser::onNodeCreate(Node * node)
{
  markVisited(node);
}

void
BottomUpTraverser::onNodeDestroy(Node * node)
{
  for (const auto & input : node->Inputs())
  {
    if (auto pred = TryGetOwnerNode<Node>(*input.origin()))
    {
      // Set threshold to 0 here: The predecessor node is
      // still connected, so its successor count is not correct.
      // However, if the node has been activated before, then
      // it will remain activated after this removal. The only
      // thing that we need to ensure here is that the total
      // count is correct.
      tracker_.decActivationCount(pred, 0);
    }
  }
}

void
BottomUpTraverser::onInputCreate(Input * input)
{
  if (auto pred = TryGetOwnerNode<Node>(*input->origin()))
  {
    if (isInputActivated(input))
    {
      tracker_.incActivationCount(pred, pred->NumSuccessors());
    }
    else
    {
      tracker_.checkNodeDeactivation(pred, pred->NumSuccessors());
    }
  }
}

void
BottomUpTraverser::onInputChange(Input * in, Output * old_origin, Output * new_origin)
{
  if (isInputActivated(in))
  {
    if (auto pred = TryGetOwnerNode<Node>(*old_origin))
    {
      tracker_.decActivationCount(pred, pred->NumSuccessors());
    }
    if (auto pred = TryGetOwnerNode<Node>(*new_origin))
    {
      tracker_.incActivationCount(pred, pred->NumSuccessors());
    }
  }
}

void
BottomUpTraverser::onInputDestroy(Input * input)
{
  if (auto pred = TryGetOwnerNode<Node>(*input->origin()))
  {
    if (isInputActivated(input))
    {
      tracker_.decActivationCount(pred, 0);
    }
    else
    {
      tracker_.checkNodeActivation(pred, pred->NumSuccessors() - 1);
    }
  }
}

bool
TraversalTracker::isNodeVisited(Node * node) const
{
  auto i = states_.find(node);
  return i == states_.end() ? false : i->second.state == traversal_nodestate::behind;
}

void
TraversalTracker::checkNodeActivation(Node * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ traversal_nodestate::ahead }).first;
  if (i->second.activation_count >= threshold && i->second.state == traversal_nodestate::ahead)
  {
    frontier_.push_back(node);
    i->second.pos = std::prev(frontier_.end());
    i->second.state = traversal_nodestate::frontier;
  }
}

void
TraversalTracker::checkNodeDeactivation(Node * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ traversal_nodestate::ahead }).first;
  if (i->second.activation_count < threshold && i->second.state == traversal_nodestate::frontier)
  {
    frontier_.erase(i->second.pos);
    i->second.pos = frontier_.end();
    i->second.state = traversal_nodestate::ahead;
  }
}

void
TraversalTracker::checkMarkNodeVisited(Node * node)
{
  auto i = states_.emplace(node, State{ traversal_nodestate::ahead }).first;
  if (i->second.state == traversal_nodestate::frontier)
  {
    frontier_.erase(i->second.pos);
    i->second.pos = frontier_.end();
    i->second.state = traversal_nodestate::behind;
  }
}

void
TraversalTracker::incActivationCount(Node * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ traversal_nodestate::ahead }).first;
  i->second.activation_count += 1;
  checkNodeActivation(node, threshold);
}

void
TraversalTracker::decActivationCount(Node * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ traversal_nodestate::ahead }).first;
  i->second.activation_count -= 1;
  checkNodeDeactivation(node, threshold);
}

void
TraversalTracker::removeNode(Node * node)
{
  if (const auto it = states_.find(node); it != states_.end())
  {
    if (it->second.state == traversal_nodestate::frontier)
      frontier_.erase(it->second.pos);
    states_.erase(it);
  }
}

Node *
TraversalTracker::peek()
{
  return frontier_.empty() ? nullptr : frontier_.front();
}

}
