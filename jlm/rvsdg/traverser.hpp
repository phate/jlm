/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRAVERSER_HPP
#define JLM_RVSDG_TRAVERSER_HPP

#include <jlm/rvsdg/tracker.hpp>

namespace jlm::rvsdg
{

class Graph;
class input;
class output;

namespace detail
{

template<typename T>
class TraverserIterator
{
public:
  typedef std::input_iterator_tag iterator_category;
  typedef Node * value_type;
  typedef ssize_t difference_type;
  typedef value_type * pointer;
  typedef value_type & reference;

  constexpr explicit TraverserIterator(T * traverser = nullptr, Node * node = nullptr) noexcept
      : traverser_(traverser),
        node_(node)
  {}

  inline const TraverserIterator &
  operator++() noexcept
  {
    node_ = traverser_->next();
    return *this;
  }

  inline bool
  operator==(const TraverserIterator & other) const noexcept
  {
    return traverser_ == other.traverser_ && node_ == other.node_;
  }

  inline bool
  operator!=(const TraverserIterator & other) const noexcept
  {
    return !(*this == other);
  }

  inline value_type &
  operator*() noexcept
  {
    return node_;
  }

  inline value_type
  operator->() noexcept
  {
    return node_;
  }

private:
  T * traverser_;
  Node * node_;
};

}

enum class traversal_nodestate
{
  ahead = -1,
  frontier = 0,
  behind = +1
};

/* support class to track traversal states of nodes */
class TraversalTracker final
{
public:
  explicit inline TraversalTracker(Graph * graph);

  inline traversal_nodestate
  get_nodestate(Node * node);

  inline void
  set_nodestate(Node * node, traversal_nodestate state);

  inline Node *
  peek_top();

  inline Node *
  peek_bottom();

private:
  Tracker tracker_;
};

/** \brief TopDown Traverser
 *
 * The topdown traverser visits a regions' nodes starting at the nodes with the lowest depth to the
 * nodes with the highest depth, i.e. from the topmost nodes to the nodes at the bottom. The
 * traverser guarantees that newly created nodes are never visited iff the created nodes replace
 * already traversed nodes, including the current node under inspection, and iff the edges of the
 * inputs of the newly created nodes originate from already traversed nodes. Otherwise, newly
 * created nodes might also be traversed. The main usage of the topdown traverser is for replacing
 * subgraphs in the already visited part of a region.
 *
 * The topdown traverser associates three distinct states with any node in the region throughout
 * traversal:
 *
 * 1. <b>ahead</b>: Nodes that have not been visited yet and are not yet marked for visitation.
 * 2. <b>frontier</b>: Nodes that are marked for visitation.
 * 3. <b>behind</b>: Nodes that were already visited.
 *
 * All nodes are by default in state <em>ahead</em>. The topdown_traverser() constructor associates
 * the <em>frontier</em> state with the top-most nodes in the region, <em>i.e.</em>, all nodes that
 * have no inputs or only region arguments as origins. The next() method can then be used to
 * traverse these <em>frontier</em> nodes. Before a <em>frontier</em> node is returned by next(), it
 * is marked as <em>behind</em> and all nodes that depend on its outputs are transferred from the
 * <em>ahead</em> state to state <em>frontier</em>. The repeated invocation of next() traverses all
 * nodes in the region.
 *
 * A newly created node is marked as <em>behind</em> iff all the nodes' predecessors are marked as
 * behind. Otherwise, it is marked as <em>frontier</em>.
 *
 * An alternative to traversing all nodes using next() is the utilization of begin() and end().
 *
 * @see node::depth()
 */
class TopDownTraverser final
{
public:
  ~TopDownTraverser() noexcept;

  explicit TopDownTraverser(Region * region);

  Node *
  next();

  [[nodiscard]] rvsdg::Region *
  region() const noexcept
  {
    return region_;
  }

  typedef detail::TraverserIterator<TopDownTraverser> iterator;
  typedef Node * value_type;

  inline iterator
  begin()
  {
    return iterator(this, next());
  }

  inline iterator
  end()
  {
    return iterator(this, nullptr);
  }

private:
  bool
  predecessors_visited(const Node * node) noexcept;

  void
  node_create(Node * node);

  void
  input_change(input * in, output * old_origin, output * new_origin);

  rvsdg::Region * region_;
  TraversalTracker tracker_;
  std::vector<jlm::util::Callback> callbacks_;
};

class BottomUpTraverser final
{
public:
  ~BottomUpTraverser() noexcept;

  explicit BottomUpTraverser(Region * region, bool revisit = false);

  Node *
  next();

  [[nodiscard]] rvsdg::Region *
  region() const noexcept
  {
    return region_;
  }

  typedef detail::TraverserIterator<BottomUpTraverser> iterator;
  typedef Node * value_type;

  inline iterator
  begin()
  {
    return iterator(this, next());
  }

  inline iterator
  end()
  {
    return iterator(this, nullptr);
  }

private:
  void
  node_create(Node * node);

  void
  node_destroy(Node * node);

  void
  input_change(input * in, output * old_origin, output * new_origin);

  rvsdg::Region * region_;
  TraversalTracker tracker_;
  std::vector<jlm::util::Callback> callbacks_;
  traversal_nodestate new_node_state_;
};

/* traversal tracker implementation */

TraversalTracker::TraversalTracker(Graph * graph)
    : tracker_(graph, 2)
{}

traversal_nodestate
TraversalTracker::get_nodestate(Node * node)
{
  return static_cast<traversal_nodestate>(tracker_.get_nodestate(node));
}

void
TraversalTracker::set_nodestate(Node * node, traversal_nodestate state)
{
  tracker_.set_nodestate(node, static_cast<size_t>(state));
}

Node *
TraversalTracker::peek_top()
{
  return tracker_.peek_top(static_cast<size_t>(traversal_nodestate::frontier));
}

Node *
TraversalTracker::peek_bottom()
{
  return tracker_.peek_bottom(static_cast<size_t>(traversal_nodestate::frontier));
}

}

#endif
