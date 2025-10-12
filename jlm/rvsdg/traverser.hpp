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
  inline traversal_nodestate
  get_nodestate(Node * node);

  inline void
  set_nodestate(Node * node, traversal_nodestate state);

  inline Node *
  peek();

private:
  using FrontierList = std::list<Node *>;

  struct State
  {
    traversal_nodestate state = traversal_nodestate::ahead;
    FrontierList::iterator pos = {};
  };

  std::unordered_map<Node *, State> states_;
  FrontierList frontier_;
};

/** \brief TopDown Traverser
 *
 * The topdown traverser visits a regions' nodes starting at the nodes that have no inputs
 * besides graph arguments, i.e. from the topmost nodes to the nodes at the bottom. The
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
 */
class TopDownTraverser final
{
public:
  ~TopDownTraverser() noexcept;

  explicit TopDownTraverser(Region * region);

  Node *
  next();

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
  class Observer final : public RegionObserver
  {
  public:
    ~Observer() noexcept override;

    Observer(Region & region, TopDownTraverser & traverser);

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
    TopDownTraverser & traverser_;
  };

  bool
  predecessors_visited(const Node * node) noexcept;

  void
  node_create(Node * node);

  void
  input_change(Input * in, Output * old_origin, Output * new_origin);

  TraversalTracker tracker_;
  Observer observer_;
};

class BottomUpTraverser final
{
public:
  ~BottomUpTraverser() noexcept;

  explicit BottomUpTraverser(Region * region);

  Node *
  next();

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
  class Observer final : public RegionObserver
  {
  public:
    ~Observer() noexcept override;

    Observer(Region & region, BottomUpTraverser & traverser);

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
    BottomUpTraverser & traverser_;
  };

  void
  node_create(Node * node);

  void
  node_destroy(Node * node);

  void
  input_change(Input * in, Output * old_origin, Output * new_origin);

  TraversalTracker tracker_;
  Observer observer_;
};

}

#endif
