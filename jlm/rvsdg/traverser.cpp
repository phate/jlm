/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::rvsdg
{

namespace detail
{
template<bool IsConst>
TopDownTraverserGeneric<IsConst>::~TopDownTraverserGeneric() noexcept = default;

template<bool IsConst>
TopDownTraverserGeneric<IsConst>::TopDownTraverserGeneric(RegionType * region)
{
  for (auto & node : region->Nodes())
  {
    unvisited_.Insert(&node);

    // Build the initial frontier from nodes that have no predecessor nodes
    bool onFrontier = true;
    for (auto & input : node.Inputs())
    {
      if (TryGetOwnerNode<Node>(*input.origin()))
      {
        onFrontier = false;
        break;
      }
    }
    if (onFrontier)
      frontier_.emplace(node.GetNodeId(), &node);
  }
}

template<bool IsConst>
bool
TopDownTraverserGeneric<IsConst>::allPredecessorsVisited(const Node & node) const noexcept
{
  for (auto & input : node.Inputs())
  {
    if (auto predecessor = TryGetOwnerNode<Node>(*input.origin());
        predecessor && unvisited_.Contains(predecessor))
      return false;
  }

  return true;
}

template<bool IsConst>
bool
TopDownTraverserGeneric<IsConst>::prepareFrontier()
{
  if (!frontier_.empty())
    return true;

  if (unvisited_.IsEmpty())
    return false;

  // We still have unvisited nodes, despite the frontier being empty
  // This can happen in rare cases when nodes ahead of the frontier change their inputs.
  // Go over the unvisited nodes again to create a new frontier
  for (auto & node : unvisited_.Items())
  {
    if (allPredecessorsVisited(*node))
    {
      frontier_.emplace(node->GetNodeId(), node);
    }
  }

  JLM_ASSERT(!frontier_.empty());
  return true;
}

template<bool IsConst>
typename TopDownTraverserGeneric<IsConst>::NodeType *
TopDownTraverserGeneric<IsConst>::popFrontier()
{
  while (true)
  {
    if (!prepareFrontier())
      return nullptr;

    NodeType * node = frontier_.begin()->second;
    frontier_.erase(frontier_.begin());

    // For non-const traversals, we must double check that all predecessors have been visited
    if (IsConst || allPredecessorsVisited(*node))
      return node;
  }
}

template<bool IsConst>
typename TopDownTraverserGeneric<IsConst>::NodeType *
TopDownTraverserGeneric<IsConst>::next()
{
  NodeType * node = popFrontier();
  if (!node)
    return nullptr;

  unvisited_.Remove(node);
  for (auto & output : node->Outputs())
  {
    for (const auto & user : output.Users())
    {
      if (auto userNode = TryGetOwnerNode<Node>(user);
          userNode && unvisited_.Contains(userNode) && frontier_.count(userNode->GetNodeId()) == 0
          && allPredecessorsVisited(*userNode))
      {
        frontier_.emplace(userNode->GetNodeId(), userNode);
      }
    }
  }

  return node;
}

template<bool IsConst>
BottomUpTraverserGeneric<IsConst>::~BottomUpTraverserGeneric() noexcept = default;

template<bool IsConst>
BottomUpTraverserGeneric<IsConst>::BottomUpTraverserGeneric(RegionType * region)
{
  for (auto & node : region->Nodes())
  {
    unvisited_.Insert(&node);

    // Build the initial frontier from nodes that have no successor nodes
    bool onFrontier = true;
    for (auto & output : node.Outputs())
    {
      for (auto & user : output.Users())
      {
        if (TryGetOwnerNode<Node>(user))
        {
          onFrontier = false;
          break;
        }
      }
    }
    if (onFrontier)
      frontier_.emplace(node.GetNodeId(), &node);
  }
}

template<bool IsConst>
bool
BottomUpTraverserGeneric<IsConst>::allSuccessorsVisited(const Node & node) const noexcept
{
  for (auto & output : node.Outputs())
  {
    for (auto & user : output.Users())
    {
      if (auto successor = TryGetOwnerNode<Node>(user); successor && unvisited_.Contains(successor))
        return false;
    }
  }

  return true;
}

template<bool IsConst>
bool
BottomUpTraverserGeneric<IsConst>::prepareFrontier()
{
  if (!frontier_.empty())
    return true;

  if (unvisited_.IsEmpty())
    return false;

  // We still have unvisited nodes, despite the frontier being empty
  // This can happen in rare cases when nodes ahead of the frontier change users.
  // Go over the unvisited nodes again to create a new frontier
  for (auto & node : unvisited_.Items())
  {
    if (allSuccessorsVisited(*node))
    {
      frontier_.emplace(node->GetNodeId(), node);
    }
  }

  JLM_ASSERT(!frontier_.empty());
  return true;
}

template<bool IsConst>
typename BottomUpTraverserGeneric<IsConst>::NodeType *
BottomUpTraverserGeneric<IsConst>::popFrontier()
{
  while (true)
  {
    if (!prepareFrontier())
      return nullptr;

    NodeType * node = frontier_.begin()->second;
    frontier_.erase(frontier_.begin());

    // For non-const traversals, we must double check that all successors have been visited
    if (IsConst || allSuccessorsVisited(*node))
      return node;
  }
}

template<bool IsConst>
typename BottomUpTraverserGeneric<IsConst>::NodeType *
BottomUpTraverserGeneric<IsConst>::next()
{
  NodeType * node = popFrontier();
  if (!node)
    return nullptr;

  unvisited_.Remove(node);
  for (auto & input : node->Inputs())
  {
    if (auto originNode = TryGetOwnerNode<Node>(*input.origin());
        originNode && unvisited_.Contains(originNode)
        && frontier_.count(originNode->GetNodeId()) == 0 && allSuccessorsVisited(*originNode))
    {
      frontier_.emplace(originNode->GetNodeId(), originNode);
    }
  }

  return node;
}

// Explicit instantiation of the const and non-const classes
template class TopDownTraverserGeneric<true>;
template class TopDownTraverserGeneric<false>;

template class BottomUpTraverserGeneric<true>;
template class BottomUpTraverserGeneric<false>;
}

}
