/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRAVERSER_HPP
#define JLM_RVSDG_TRAVERSER_HPP

#include <jlm/rvsdg/region.hpp>

#include <list>
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

enum class TraversalNodestate
{
  ahead = -1,
  frontier = 0,
  behind = +1
};

/* support class to track traversal states of nodes */
template<typename NodeType>
class TraversalTracker final
{
public:
  /** \brief Determines whether node has been visited already. */
  bool
  isNodeVisited(NodeType * node) const;

  /** \brief Checks activation count whether node is ready for visiting. */
  void
  checkNodeActivation(NodeType * node, std::size_t threshold);

  /** \brief Checks activation count whether node is no longer ready for visiting. */
  void
  checkNodeDeactivation(NodeType * node, std::size_t threshold);

  /** \brief Marks a node visited if it is currently ready for visiting. */
  void
  checkMarkNodeVisitedIfFrontier(NodeType * node);

  /** \brief Increments activation count; adds to frontier if threshold is met. */
  void
  incActivationCount(NodeType * node, std::size_t threshold);

  /** \brief Decrements activation count; removes from frontier if threshold is no longer met. */
  void
  decActivationCount(NodeType * node, std::size_t threshold);

  /** \brief Removes any state associated with the given node */
  void
  removeNode(NodeType * node);

  NodeType *
  peek();

private:
  using FrontierList = std::list<NodeType *>;

  struct State
  {
    TraversalNodestate state = TraversalNodestate::ahead;
    std::size_t activationCount = 0;
    typename FrontierList::iterator pos = {};
  };

  std::unordered_map<NodeType *, State> states_;
  FrontierList frontier_;
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

  constexpr TraverserIterator(Traverser * traverser, NodeType * node) noexcept
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

  void
  markAsVisited(NodeType & node);

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

  TraversalTracker<NodeType> tracker_;
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

  void
  markAsVisited(NodeType & node);

  void
  onNodeCreate(NodeType * node);

  void
  onNodeDestroy(NodeType * node);

  void
  onInputCreate(Input * input);

  void
  onInputChange(Input * input, Output * oldOrigin, Output * newOrigin);

  void
  onInputDestroy(Input * input);

  TraversalTracker<NodeType> tracker_;
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
