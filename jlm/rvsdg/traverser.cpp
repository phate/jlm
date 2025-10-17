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

namespace detail
{

template<typename Traverser>
ForwardingObserver<Traverser>::~ForwardingObserver() noexcept = default;

template<typename Traverser>
ForwardingObserver<Traverser>::ForwardingObserver(Region & region, Traverser & traverser)
    : RegionObserver(region),
      traverser_(traverser)
{}

template<typename Traverser>
void
ForwardingObserver<Traverser>::onNodeCreate(Node * node)
{
  traverser_.onNodeCreate(node);
}

template<typename Traverser>
void
ForwardingObserver<Traverser>::onNodeDestroy(Node * node)
{
  traverser_.onNodeDestroy(node);
}

template<typename Traverser>
void
ForwardingObserver<Traverser>::onInputCreate(Input * input)
{
  traverser_.onInputCreate(input);
}

template<typename Traverser>
void
ForwardingObserver<Traverser>::onInputChange(Input * input, Output * oldOrigin, Output * newOrigin)
{
  traverser_.onInputChange(input, oldOrigin, newOrigin);
}

template<typename Traverser>
void
ForwardingObserver<Traverser>::onInputDestroy(Input * input)
{
  traverser_.onInputDestroy(input);
}

template<typename Traverser>
DummyObserver<Traverser>::DummyObserver(
    [[maybe_unused]] const Region & region,
    [[maybe_unused]] Traverser & traverser)
{}

template<bool IsConst>
TopDownTraverserGeneric<IsConst>::~TopDownTraverserGeneric() noexcept = default;

template<bool IsConst>
TopDownTraverserGeneric<IsConst>::TopDownTraverserGeneric(RegionType * region)
    : observer_(*region, *this)
{
  for (auto & node : region->TopNodes())
  {
    tracker_.checkNodeActivation(&node, node.ninputs());
  }

  for (auto argument : region->Arguments())
  {
    for (const auto & user : argument->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
      {
        tracker_.incActivationCount(node, node->ninputs());
      }
    }
  }
}

template<bool IsConst>
bool
TopDownTraverserGeneric<IsConst>::isOutputActivated(const Output & output)
{
  if (auto pred = TryGetOwnerNode<Node>(output))
  {
    return tracker_.isNodeVisited(pred);
  }

  // The output is a region argument, always considered activated
  return true;
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::markAsVisited(NodeType & node)
{
  tracker_.checkMarkNodeVisitedIfFrontier(&node);
  for (const auto & output : node.Outputs())
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

template<bool IsConst>
typename TopDownTraverserGeneric<IsConst>::NodeType *
TopDownTraverserGeneric<IsConst>::next()
{
  const auto node = tracker_.peek();
  if (!node)
    return nullptr;

  markAsVisited(*node);

  return node;
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onNodeCreate(NodeType * node)
{
  for (const auto & input : node->Inputs())
  {
    if (isOutputActivated(*input.origin()))
    {
      tracker_.incActivationCount(node, node->ninputs());
    }
  }

  if (node->ninputs() == 0)
    tracker_.checkNodeActivation(node, node->ninputs());

  // If node would end up on frontier (because all predecessors
  // have been visited), mark it as visited instead (we do not
  // want to revisit newly created nodes during topdown traversal).
  tracker_.checkMarkNodeVisitedIfFrontier(node);
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onNodeDestroy(NodeType * node)
{
  // The node is already dead, so removing it can never add anything to the frontier
  tracker_.removeNode(node);
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onInputCreate(Input * input)
{
  const auto node = TryGetOwnerNode<Node>(*input);
  if (!node)
    return;

  if (isOutputActivated(*input->origin()))
  {
    tracker_.incActivationCount(node, node->ninputs());
  }
  else
  {
    tracker_.checkNodeDeactivation(node, node->ninputs());
  }
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onInputChange(
    Input * input,
    Output * oldOrigin,
    Output * newOrigin)
{
  const auto node = TryGetOwnerNode<Node>(*input);
  if (!node)
    return;

  int change = 0;
  if (isOutputActivated(*newOrigin))
  {
    change += 1;
  }
  if (isOutputActivated(*oldOrigin))
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

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onInputDestroy(Input * input)
{
  const auto node = TryGetOwnerNode<Node>(*input);
  if (!node)
    return;

  if (isOutputActivated(*input->origin()))
  {
    tracker_.decActivationCount(node, node->ninputs() - 1);
  }
  else
  {
    tracker_.checkNodeActivation(node, node->ninputs() - 1);
  }
}

template<bool IsConst>
BottomUpTraverserGeneric<IsConst>::~BottomUpTraverserGeneric() noexcept = default;

template<bool IsConst>
BottomUpTraverserGeneric<IsConst>::BottomUpTraverserGeneric(RegionType * region)
    : observer_(*region, *this)
{
  for (auto & node : region->BottomNodes())
  {
    tracker_.checkNodeActivation(&node, node.numSuccessors());
  }

  for (auto result : region->Results())
  {
    if (auto node = TryGetOwnerNode<Node>(*result->origin()))
    {
      tracker_.incActivationCount(node, node->numSuccessors());
    }
  }
}

template<bool IsConst>
bool
BottomUpTraverserGeneric<IsConst>::isInputActivated(const Input & input)
{
  if (auto node = TryGetOwnerNode<Node>(input))
  {
    return tracker_.isNodeVisited(node);
  }

  // The output is a region result, always considered activated
  return true;
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::markAsVisited(NodeType & node)
{
  tracker_.checkMarkNodeVisitedIfFrontier(&node);
  for (const auto & input : node.Inputs())
  {
    if (auto predecessor = TryGetOwnerNode<Node>(*input.origin()))
    {
      tracker_.incActivationCount(predecessor, predecessor->numSuccessors());
    }
  }
}

template<bool IsConst>
typename BottomUpTraverserGeneric<IsConst>::NodeType *
BottomUpTraverserGeneric<IsConst>::next()
{
  auto node = tracker_.peek();
  if (!node)
    return nullptr;

  markAsVisited(*node);
  return node;
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onNodeCreate(NodeType * node)
{
  // The new node should never be visited, so activate all its predecessors immediately
  markAsVisited(*node);
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onNodeDestroy(NodeType * node)
{
  // Require nodes we remove to have been visited.
  // This ensures that removing a node never causes other nodes to be added to the frontier
  if (!tracker_.isNodeVisited(node))
    throw std::logic_error("Removing unvisited node");

  for (const auto & input : node->Inputs())
  {
    if (auto pred = TryGetOwnerNode<Node>(*input.origin()))
    {
      // Set the threshold to 0 here: The predecessor node is
      // still connected, so its successor count is not correct.
      // However, if the predecessor is on the frontier, it will
      // remain on the frontier after this removal.
      tracker_.decActivationCount(pred, 0);
    }
  }
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onInputCreate(Input * input)
{
  const auto node = TryGetOwnerNode<Node>(*input->origin());
  if (!node)
    return;

  if (isInputActivated(*input))
  {
    tracker_.incActivationCount(node, node->numSuccessors());
  }
  else
  {
    tracker_.checkNodeDeactivation(node, node->numSuccessors());
  }
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onInputChange(
    Input * input,
    Output * oldOrigin,
    Output * newOrigin)
{
  const auto inputActive = isInputActivated(*input);

  if (auto oldNode = TryGetOwnerNode<Node>(*oldOrigin))
  {
    if (inputActive)
    {
      tracker_.decActivationCount(oldNode, oldNode->numSuccessors());
    }
    else
    {
      // The oldNode might have just lost its final non-active user
      tracker_.checkNodeActivation(oldNode, oldNode->numSuccessors());
    }
  }

  if (auto newNode = TryGetOwnerNode<Node>(*newOrigin))
  {
    if (inputActive)
    {
      tracker_.incActivationCount(newNode, newNode->numSuccessors());
    }
    else
    {
      tracker_.checkNodeDeactivation(newNode, newNode->numSuccessors());
    }
  }
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onInputDestroy(Input * input)
{
  const auto node = TryGetOwnerNode<Node>(*input->origin());
  if (!node)
    return;

  if (isInputActivated(*input))
  {
    tracker_.decActivationCount(node, node->numSuccessors() - 1);
  }
  else
  {
    tracker_.checkNodeActivation(node, node->numSuccessors() - 1);
  }
}

template<typename NodeType>
bool
TraversalTracker<NodeType>::isNodeVisited(NodeType * node) const
{
  auto i = states_.find(node);
  return i == states_.end() ? false : i->second.state == TraversalNodestate::behind;
}

template<typename NodeType>
void
TraversalTracker<NodeType>::checkNodeActivation(NodeType * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ TraversalNodestate::ahead }).first;
  if (i->second.activationCount >= threshold && i->second.state == TraversalNodestate::ahead)
  {
    frontier_.push_back(node);
    i->second.pos = std::prev(frontier_.end());
    i->second.state = TraversalNodestate::frontier;
  }
}

template<typename NodeType>
void
TraversalTracker<NodeType>::checkNodeDeactivation(NodeType * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ TraversalNodestate::ahead }).first;
  if (i->second.activationCount < threshold && i->second.state == TraversalNodestate::frontier)
  {
    frontier_.erase(i->second.pos);
    i->second.pos = frontier_.end();
    i->second.state = TraversalNodestate::ahead;
  }
}

template<typename NodeType>
void
TraversalTracker<NodeType>::checkMarkNodeVisitedIfFrontier(NodeType * node)
{
  auto i = states_.emplace(node, State{ TraversalNodestate::ahead }).first;
  if (i->second.state == TraversalNodestate::frontier)
  {
    frontier_.erase(i->second.pos);
    i->second.pos = frontier_.end();
    i->second.state = TraversalNodestate::behind;
  }
}

template<typename NodeType>
void
TraversalTracker<NodeType>::incActivationCount(NodeType * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ TraversalNodestate::ahead }).first;
  i->second.activationCount += 1;
  checkNodeActivation(node, threshold);
}

template<typename NodeType>
void
TraversalTracker<NodeType>::decActivationCount(NodeType * node, std::size_t threshold)
{
  auto i = states_.emplace(node, State{ TraversalNodestate::ahead }).first;
  i->second.activationCount -= 1;
  checkNodeDeactivation(node, threshold);
}

template<typename NodeType>
void
TraversalTracker<NodeType>::removeNode(NodeType * node)
{
  if (const auto it = states_.find(node); it != states_.end())
  {
    if (it->second.state == TraversalNodestate::frontier)
      frontier_.erase(it->second.pos);
    states_.erase(it);
  }
}

template<typename NodeType>
NodeType *
TraversalTracker<NodeType>::peek()
{
  return frontier_.empty() ? nullptr : frontier_.front();
}

// Explicit instantiation of all versions
template class TopDownTraverserGeneric<false>;
template class TopDownTraverserGeneric<true>;
template class BottomUpTraverserGeneric<false>;
template class BottomUpTraverserGeneric<true>;

}

}
