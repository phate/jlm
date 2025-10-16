/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRAVERSER_HPP
#define JLM_RVSDG_TRAVERSER_HPP

#include <jlm/rvsdg/region.hpp>

#include <limits>
#include <map>
#include <unordered_map>

namespace jlm::rvsdg
{

class Graph;
class Input;
class Node;
class Output;
class Region;

namespace detail
{

template<typename NodeType>
class TraverserTracker final
{
public:
  static constexpr size_t VISITED = std::numeric_limits<size_t>::max();

  /**
   * Adds a new node to the tracker with a starting set of activations.
   * If the threshold is met, the node is also added to the frontier.
   * @param node a new node, not previously seen by the tracker
   * @param activations the number of activations the node starts with
   * @param threshold the node's initial threshold
   */
  void
  addNodeWithActivations(NodeType & node, size_t activations, size_t threshold);

  /**
   * Increases the activation number of the given \p node.
   * If the node is new to the tracker, it gets added.
   * If the node has enough activations to meet its threshold, it is added to the frontier.
   * @param node the activated node
   * @param threshold the node's threshold
   */
  void
  increaseActivationCount(NodeType & node, size_t threshold);

  /**
   * Decreases the activation number of the given \p node.
   * If the new activation is below the threshold, it is removed from the frontier.
   * @param node the activated node
   * @param threshold the node's threshold
   */
  void
  decreaseActivationCount(NodeType & node, size_t threshold);

  /**
   * Call when the threshold increased without a corresponding increase in activation count
   * @param node the node whose threshold changed
   * @param threshold the new threshold value
   */
  void
  onThresholdIncrease(NodeType & node, size_t threshold);

  /**
   * Call when the threshold decreased without a corresponding decrease in activation count
   * @param node the node whose threshold changed
   * @param threshold the new threshold value
   */
  void
  onThresholdDecrease(NodeType & node, size_t threshold);

  /**
   * Extracts one node from the frontier, deterministically.
   * @return the popped node, or nullptr if the frontier is empty.
   */
  NodeType * popFrontier();

  /**
   * Marks the given \p node as visited.
   * The node must not be on the frontier, and not already marked as visited.
   * @param node the node to be marked
   */
  void
  markAsVisited(NodeType & node);

  /**
   * Checks if the given \p node has been visited
   * @param node the node in question
   * @return true if the node is marked as visited, false otherwise
   */
  bool
  hasBeenVisited(NodeType & node);

  /**
   * Removes any recollection of the given \p node existing.
   * @param node the node in question
   */
  void removeNode(NodeType & node);

private:

  // The number of activations a node has seen.
  // All nodes whose activations = numInputs are on the frontier
  // Once a node is visited, its activation count is set to VISITED
  std::unordered_map<NodeType *, size_t> activations_;
  // The frontier queue, indexed by NodeId
  std::map<typename NodeType::Id, NodeType *> frontier_;
};

template<typename Traverser, typename NodeType>
class TraverserIterator
{
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = NodeType *;
  using difference_type = ssize_t;
  using pointer = value_type *;
  using reference = value_type &;

  constexpr TraverserIterator(Traverser * traverser = nullptr, NodeType * node = nullptr) noexcept
      : traverser_(traverser),
        node_(node)
  {}

  const TraverserIterator &
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

/**
 * Region observer that forwards all calls to the given traverser
 * @tparam Traverser the type of the target traverser
 */
template<typename Traverser>
class ForwardingObserver final : public RegionObserver
{
public:
  ~ForwardingObserver() noexcept override;

  ForwardingObserver(Region & region, Traverser & traverser);

  void
  onNodeCreate(Node * node) override;

  void
  onNodeDestroy(Node * node) override;

  void
  onInputCreate(Input * input) override;

  void
  onInputChange(Input * input, Output * old_origin, Output * new_origin) override;

  void
  onInputDestroy(Input * input) override;

private:
  Traverser & traverser_;
};

/**
 * A fake region observer that does not register itself with the region.
 * @tparam Traverser the type of the target traverser
 */
template<typename Traverser>
class DummyObserver final
{
public:
  DummyObserver(const Region & region, Traverser & traverser);
};

template<bool IsConst>
class TopDownTraverserGeneric final
{
public:
  using NodeType = std::conditional_t<IsConst, const Node, Node>;
  using RegionType = std::conditional_t<IsConst, const Region, Region>;
  using ObserverType = std::conditional_t<
      IsConst,
      DummyObserver<TopDownTraverserGeneric>,
      ForwardingObserver<TopDownTraverserGeneric>>;
  using iterator = TraverserIterator<TopDownTraverserGeneric, NodeType>;

  friend class ForwardingObserver<TopDownTraverserGeneric>;

  ~TopDownTraverserGeneric() noexcept;

  explicit TopDownTraverserGeneric(RegionType * region);

  NodeType *
  next();

  iterator
  begin()
  {
    return iterator(this, next());
  }

  iterator
  end()
  {
    return iterator(this, nullptr);
  }

private:
  bool
  isOutputActivated(const Output & output);

  void markAsVisited(NodeType & node);

  void
  onNodeCreate(NodeType * node);

  void
  onNodeDestroy(NodeType * node);

  void
  onInputCreate(Input * input);

  void
  onInputChange(Input * in, Output * old_origin, Output * new_origin);

  void
  onInputDestroy(Input * input);

  TraverserTracker<NodeType> tracker_;
  ObserverType observer_;
};

template<bool IsConst>
class BottomUpTraverserGeneric final
{
public:
  using NodeType = std::conditional_t<IsConst, const Node, Node>;
  using RegionType = std::conditional_t<IsConst, const Region, Region>;
  using ObserverType = std::conditional_t<
      IsConst,
      DummyObserver<BottomUpTraverserGeneric>,
      ForwardingObserver<BottomUpTraverserGeneric>>;
  using iterator = TraverserIterator<BottomUpTraverserGeneric, NodeType>;

  friend class ForwardingObserver<BottomUpTraverserGeneric>;

  ~BottomUpTraverserGeneric() noexcept;

  explicit BottomUpTraverserGeneric(RegionType * region);

  NodeType *
  next();

  iterator
  begin()
  {
    return iterator(this, next());
  }

  iterator
  end()
  {
    return iterator(this, nullptr);
  }

private:
  bool
  isInputActivated(const Input & output);

  void markAsVisited(NodeType & node);

  void
  onNodeCreate(NodeType * node);

  void
  onNodeDestroy(NodeType * node);

  void
  onInputCreate(Input * input);

  void
  onInputChange(Input * in, Output * old_origin, Output * new_origin);

  void
  onInputDestroy(Input * input);

  TraverserTracker<NodeType> tracker_;
  ObserverType observer_;
};

}

/** \brief TopDown Traverser
 *
 * The top down traverser visits all nodes in a region, starting at the nodes that have no inputs
 * besides graph arguments, i.e. from the topmost nodes to the nodes at the bottom.
 * Nodes created during traversal are not visited.
 *
 * The main usage of the top down traverser is for analyzing graphs, or replacing subgraphs in the
 * already visited part of a region.
 *
 * An alternative to traversing all nodes using next() is the utilization of begin() and end().
 */
using TopDownTraverser = detail::TopDownTraverserGeneric<false>;
/** \brief Const top down traverser
 *
 * A version of \ref TopDownTraverser that does not support the region changing during traversal.
 */
using TopDownConstTraverser = detail::TopDownTraverserGeneric<true>;

/** \brief BottomUp Traverser
 *
 * The bottom up traverser visits all nodes in a region, starting at the nodes that have no users
 * besides graph results, i.e. from the bottommost nodes to the nodes at the top.
 * Nodes created during traversal are never visited.
 *
 * An alternative to traversing all nodes using next() is the utilization of begin() and end().
 */
using BottomUpTraverser = detail::BottomUpTraverserGeneric<false>;
/** \brief Const bottom up traverser
 *
 * A version of \ref BottomUpTraverser that does not support the region changing during traversal.
 */
using BottomUpConstTraverser = detail::BottomUpTraverserGeneric<false>;

}

#endif
