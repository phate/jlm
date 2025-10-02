/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRAVERSER_HPP
#define JLM_RVSDG_TRAVERSER_HPP

#include <jlm/util/HashSet.hpp>

#include <map>

namespace jlm::rvsdg
{

class Graph;
class Input;
class Node;
class Output;
class Region;

namespace detail
{

template<typename Traverser, typename NodeType>
class TraverserIterator
{
public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = NodeType *;
  using pointer = value_type *;
  using reference = value_type &;

  constexpr explicit TraverserIterator(
      Traverser * traverser = nullptr,
      NodeType * node = nullptr) noexcept
      : traverser_(traverser),
        node_(node)
  {}

  TraverserIterator &
  operator++() noexcept
  {
    node_ = traverser_->next();
    return *this;
  }

  bool
  operator==(const TraverserIterator & other) const noexcept
  {
    return traverser_ == other.traverser_ && node_ == other.node_;
  }

  bool
  operator!=(const TraverserIterator & other) const noexcept
  {
    return !(*this == other);
  }

  reference
  operator*() noexcept
  {
    return node_;
  }

  pointer
  operator->() noexcept
  {
    return &node_;
  }

private:
  Traverser * traverser_;
  NodeType * node_;
};

template<bool IsConst>
class TopDownTraverserGeneric final
{
public:
  using NodeType = std::conditional_t<IsConst, const Node, Node>;
  using RegionType = std::conditional_t<IsConst, const Region, Region>;
  using iterator = TraverserIterator<TopDownTraverserGeneric, NodeType>;

  ~TopDownTraverserGeneric() noexcept;

  explicit TopDownTraverserGeneric(RegionType * region);

  NodeType *
  next();

  [[nodiscard]] iterator
  begin()
  {
    return iterator(this, next());
  }

  [[nodiscard]] iterator
  end()
  {
    return iterator(this, nullptr);
  }

private:
  bool
  allPredecessorsVisited(const Node & node) const noexcept;

  /**
   * If the frontier is empty, while some nodes have yet to be visited,
   * this method builds a new frontier.
   * @return true if there is a frontier, false is all nodes have been visited.
   */
  [[nodiscard]] bool
  prepareFrontier();

  /**
   * Extracts a node from the frontier.
   * In non-const traversals, it double checks that the node is actually ready to be visited.
   * @returns the popped node, or nullptr if all nodes have been visited.
   */
  [[nodiscard]] NodeType *
  popFrontier();

  // The set of nodes that are yet to be visited
  util::HashSet<NodeType *> unvisited_;
  // The frontier, containing nodes that are ready to be visited.
  // If inputs have changed during traversal, nodes can be on the frontier incorrectly.
  // uses Node:Id as key for determinism
  std::map<size_t, NodeType *> frontier_;
};

template<bool IsConst>
class BottomUpTraverserGeneric final
{
public:
  using NodeType = std::conditional_t<IsConst, const Node, Node>;
  using RegionType = std::conditional_t<IsConst, const Region, Region>;
  using iterator = TraverserIterator<BottomUpTraverserGeneric, NodeType>;

  ~BottomUpTraverserGeneric() noexcept;

  explicit BottomUpTraverserGeneric(RegionType * region);

  NodeType *
  next();

  [[nodiscard]] iterator
  begin()
  {
    return iterator(this, next());
  }

  [[nodiscard]] iterator
  end()
  {
    return iterator(this, nullptr);
  }

private:
  bool
  allSuccessorsVisited(const Node & node) const noexcept;

  /**
   * If the frontier is empty, while some nodes have yet to be visited,
   * this method builds a new frontier.
   * @return true if there is a frontier, false is all nodes have been visited.
   */
  [[nodiscard]] bool
  prepareFrontier();

  /**
   * Extracts a node from the frontier.
   * In non-const traversals, it double checks that the node is actually ready to be visited.
   * @returns the popped node, or nullptr if all nodes have been visited.
   */
  [[nodiscard]] NodeType *
  popFrontier();

  // The set of nodes that are yet to be visited
  util::HashSet<NodeType *> unvisited_;
  // The frontier, containing nodes that are ready to be visited.
  // If outputs have gained new users during traversal, nodes can be on the frontier incorrectly.
  // uses Node:Id as key for determinism
  std::map<size_t, NodeType *> frontier_;
};

}

/** \brief TopDown Traverser
 *
 * The topdown traverser visits the nodes of a region, starting at the nodes that have no inputs
 * besides region arguments, visiting every node until the bottom of the region is reached.
 * A node is only visited once all its predecessors have been visited,
 * even if its inputs have been changed during traversal.
 * Nodes created during traversal are <em>never</em> visited.
 *
 * It is forbidden to delete nodes that have not yet been visited during traversal.
 * Deleting the current node is allowed.
 *
 * The main usage of the topdown traverser is for replacing subgraphs
 * in the already visited part of a region.
 *
 * Nodes can either be visited by repeated calls to next(), or using begin() and end().
 */
using TopDownTraverser = detail::TopDownTraverserGeneric<false>;
using TopDownConstTraverser = detail::TopDownTraverserGeneric<true>;

/**
 * \brief BottomUp Traverser
 *
 * The bottom up traverser visits all nodes in the region, starting at nodes that have no outputs
 * besides region results, visiting every node until the top of the region is reached.
 * A node is only visited once all its successors have been visited,
 * even if it has gained new successors during traversal.
 * Nodes created during traversal are <em>never</em> visited.
 *
 * It is forbidden to delete nodes that have not yet been visited during traversal.
 * Deleting the current node is allowed.
 *
 * Nodes can either be visited by repeated calls to next(), or using begin() and end().
 */
using BottomUpTraverser = detail::BottomUpTraverserGeneric<false>;
using BottomUpConstTraverser = detail::BottomUpTraverserGeneric<true>;
}

#endif
