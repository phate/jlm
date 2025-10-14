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

template<typename NodeType>
void
TraverserTracker<NodeType>::addNodeWithActivations(
    NodeType & node,
    size_t activations,
    size_t threshold)
{
  const auto [_, added] = activations_.emplace(&node, activations);
  JLM_ASSERT(added);

  JLM_ASSERT(activations <= threshold);

  // If the new node has enough activations, add it to the frontier
  if (activations == threshold)
    frontier_.emplace(node.GetNodeId(), &node);
}

template<typename NodeType>
void
TraverserTracker<NodeType>::increaseActivationCount(NodeType & node, size_t threshold)
{
  auto & activations = activations_[&node];
  JLM_ASSERT(activations != VISITED);
  JLM_ASSERT(activations < threshold);

  activations++;

  // If the node now has enough activations, add it to the frontier
  if (activations == threshold)
    frontier_.emplace(node.GetNodeId(), &node);
}

template<typename NodeType>
void
TraverserTracker<NodeType>::decreaseActivationCount(NodeType & node, size_t threshold)
{
  auto & activations = activations_[&node];
  JLM_ASSERT(activations != VISITED);
  JLM_ASSERT(activations > 0);

  activations--;

  // If the node dropped below the threshold, ensure it is not on the frontier
  if (activations < threshold)
    frontier_.erase(node.GetNodeId());
}

template<typename NodeType>
void
TraverserTracker<NodeType>::onThresholdIncrease(NodeType & node, size_t threshold)
{
  JLM_ASSERT(activations_[&node] < threshold);
  frontier_.erase(node.GetNodeId());
}

template<typename NodeType>
void
TraverserTracker<NodeType>::onThresholdDecrease(NodeType & node, size_t threshold)
{
  auto & activations = activations_[&node];
  JLM_ASSERT(activations != VISITED);
  JLM_ASSERT(activations <= threshold);

  if (activations == threshold)
    frontier_.emplace(node.GetNodeId(), &node);
}

template<typename NodeType>
NodeType *
TraverserTracker<NodeType>::popFrontier()
{
  if (frontier_.empty())
    return nullptr;

  const auto node = frontier_.begin()->second;
  frontier_.erase(frontier_.begin());
  return node;
}

template<typename NodeType>
void
TraverserTracker<NodeType>::markAsVisited(NodeType & node)
{
  // The node should not be on the frontier
  JLM_ASSERT(frontier_.count(node.GetNodeId()) == 0);

  auto & activations = activations_[&node];

  // The node should not already be marked as visited
  JLM_ASSERT(activations != VISITED);
  activations = VISITED;
}

template<typename NodeType>
bool
TraverserTracker<NodeType>::hasBeenVisited(NodeType & node)
{
  return activations_[&node] == VISITED;
}

template<typename NodeType>
void
TraverserTracker<NodeType>::removeNode(NodeType & node)
{
  activations_.erase(&node);
  frontier_.erase(node.GetNodeId());
}

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
    tracker_.addNodeWithActivations(node, 0, node.ninputs());
  }

  for (auto argument : region->Arguments())
  {
    for (const auto & user : argument->Users())
    {
      if (auto node = TryGetOwnerNode<Node>(user))
      {
        tracker_.increaseActivationCount(*node, node->ninputs());
      }
    }
  }
}

template<bool IsConst>
bool
TopDownTraverserGeneric<IsConst>::isOutputActivated(const Output & output)
{
  if (auto node = TryGetOwnerNode<Node>(output))
  {
    return tracker_.hasBeenVisited(*node);
  }

  // The output is a region argument, always considered activated
  return true;
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::markAsVisited(NodeType & node)
{
  tracker_.markAsVisited(node);
  for (const auto & output : node.Outputs())
  {
    for (const auto & user : output.Users())
    {
      if (auto successor = TryGetOwnerNode<Node>(user))
      {
        tracker_.increaseActivationCount(*successor, successor->ninputs());
      }
    }
  }
}

template<bool IsConst>
typename TopDownTraverserGeneric<IsConst>::NodeType *
TopDownTraverserGeneric<IsConst>::next()
{
  auto node = tracker_.popFrontier();
  if (!node)
    return nullptr;

  markAsVisited(*node);
  return node;
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onNodeCreate(NodeType * node)
{
  size_t activations = 0;
  for (const auto & input : node->Inputs())
  {
    if (isOutputActivated(*input.origin()))
    {
      activations++;
    }
  }

  // New nodes where all predecessors are already visited should not be visited
  if (activations < node->ninputs())
    tracker_.addNodeWithActivations(*node, activations, node->ninputs());
  else
    markAsVisited(*node);
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onNodeDestroy(NodeType * node)
{
  // The node is already dead, so removing it does not add anything to the frontier
  tracker_.removeNode(*node);
}

template<bool IsConst>
void
TopDownTraverserGeneric<IsConst>::onInputCreate(Input * input)
{
  const auto node = TryGetOwnerNode<Node>(*input);
  if (!node)
    return;

  if (isOutputActivated(*input->origin()))
    tracker_.increaseActivationCount(*node, node->ninputs());
  else
    tracker_.onThresholdIncrease(*node, node->ninputs());
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
    tracker_.increaseActivationCount(*node, node->ninputs());
  }
  else if (change == -1)
  {
    tracker_.decreaseActivationCount(*node, node->ninputs());
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
    tracker_.decreaseActivationCount(*node, node->ninputs() - 1);
  else
    tracker_.onThresholdDecrease(*node, node->ninputs() - 1);
}

template<bool IsConst>
BottomUpTraverserGeneric<IsConst>::~BottomUpTraverserGeneric() noexcept = default;

template<bool IsConst>
BottomUpTraverserGeneric<IsConst>::BottomUpTraverserGeneric(RegionType * region)
    : observer_(*region, *this)
{
  for (auto & node : region->BottomNodes())
  {
    tracker_.addNodeWithActivations(node, 0, node.numSuccessors());
  }

  for (auto result : region->Results())
  {
    if (auto node = TryGetOwnerNode<Node>(*result->origin()))
    {
      tracker_.increaseActivationCount(*node, node->numSuccessors());
    }
  }
}

template<bool IsConst>
bool
BottomUpTraverserGeneric<IsConst>::isInputActivated(const Input & input)
{
  if (auto node = TryGetOwnerNode<Node>(input))
  {
    return tracker_.hasBeenVisited(*node);
  }

  // The output is a region result, always considered activated
  return true;
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::markAsVisited(NodeType & node)
{
  tracker_.markAsVisited(node);
  for (const auto & input : node.Inputs())
  {
    if (auto predecessor = TryGetOwnerNode<Node>(input))
    {
      tracker_.increaseActivationCount(*predecessor, predecessor->numSuccessors());
    }
  }
}

template<bool IsConst>
typename BottomUpTraverserGeneric<IsConst>::NodeType *
BottomUpTraverserGeneric<IsConst>::next()
{
  auto node = tracker_.popFrontier();
  if (!node)
    return nullptr;

  markAsVisited(*node);
  return node;
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onNodeCreate(NodeType * node)
{
  // The new node should never be visited
  markAsVisited(*node);

  // Any predecessor of the new node should see the new user(s) as activated
  for (const auto & input : node->Inputs())
  {
    if (auto predecessor = TryGetOwnerNode<Node>(*input.origin()))
    {
      tracker_.increaseActivationCount(*predecessor, predecessor->numSuccessors());
    }
  }
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onNodeDestroy(NodeType * node)
{
  // When this method is called, the node has yet to be destroyed.
  // Instead we add up the number of users that will be removed per node, and apply each sum
  std::unordered_map<NodeType *, size_t> thresholdDecrease;
  for (const auto & input : node->Inputs())
  {
    if (auto predecessor = TryGetOwnerNode<Node>(*input.origin()))
    {
      thresholdDecrease[predecessor]++;
    }
  }

  for (const auto & [predecessor, decrease] : thresholdDecrease)
  {
    tracker_.onThresholdDecrease(*predecessor, predecessor->numSuccessors() - decrease);
  }

  tracker_.removeNode(*node);
}

template<bool IsConst>
void
BottomUpTraverserGeneric<IsConst>::onInputCreate(Input * input)
{
  const auto node = TryGetOwnerNode<Node>(*input->origin());
  if (!node)
    return;

  if (isInputActivated(*input))
    tracker_.increaseActivationCount(*node, node->numSuccessors());
  else
    tracker_.onThresholdIncrease(*node, node->numSuccessors());
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
      tracker_.decreaseActivationCount(*oldNode, oldNode->numSuccessors());
    }
    else
    {
      tracker_.onThresholdDecrease(*oldNode, oldNode->numSuccessors());
    }
  }

  if (auto newNode = TryGetOwnerNode<Node>(*newOrigin))
  {
    if (inputActive)
    {
      tracker_.increaseActivationCount(*newNode, newNode->numSuccessors());
    }
    else
    {
      tracker_.onThresholdIncrease(*newNode, newNode->numSuccessors());
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
    tracker_.decreaseActivationCount(*node, node->numSuccessors() - 1);
  else
    tracker_.onThresholdDecrease(*node, node->numSuccessors() - 1);
}

// Explicit instantiation of all versions
template class TopDownTraverserGeneric<false>;
template class TopDownTraverserGeneric<true>;
template class BottomUpTraverserGeneric<false>;
template class BottomUpTraverserGeneric<true>;

}

}
