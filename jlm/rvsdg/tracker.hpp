/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRACKER_HPP
#define JLM_RVSDG_TRACKER_HPP

#include <jlm/util/callbacks.hpp>

#include <unordered_map>

namespace jlm::rvsdg
{

static const size_t tracker_nodestate_none = (size_t)-1;

class Graph;
class Node;
class Region;
class tracker_depth_state;
class TrackerNodeState;

bool
has_active_trackers(const Graph * graph);

/**
 * Track states of nodes within the graph. Each node can logically be in
 * one of the numbered states, plus another "initial" state. All nodes are
 * at the beginning assumed to be implicitly in this "initial" state.
 */
struct Tracker
{
public:
  ~Tracker() noexcept;

  Tracker(Graph * graph, size_t nstates);

  /* get state of the node */
  ssize_t
  get_nodestate(Node * node);

  /* set state of the node */
  void
  set_nodestate(Node * node, size_t state);

  /* get one of the top nodes for the given state */
  Node *
  peek_top(size_t state) const;

  /* get one of the bottom nodes for the given state */
  Node *
  peek_bottom(size_t state) const;

  [[nodiscard]] Graph *
  graph() const noexcept
  {
    return graph_;
  }

private:
  jlm::rvsdg::TrackerNodeState *
  nodestate(Node * node);

  void
  node_depth_change(Node * node, size_t old_depth);

  void
  node_destroy(Node * node);

  jlm::rvsdg::Graph * graph_;

  /* FIXME: need RAII idiom for state reservation */
  std::vector<std::unique_ptr<tracker_depth_state>> states_;

  jlm::util::Callback depth_callback_, destroy_callback_;

  std::unordered_map<Node *, std::unique_ptr<TrackerNodeState>> nodestates_;
};

class TrackerNodeState
{
  friend Tracker;

public:
  explicit TrackerNodeState(Node * node)
      : state_(tracker_nodestate_none),
        node_(node)
  {}

  TrackerNodeState(const TrackerNodeState &) = delete;

  TrackerNodeState(TrackerNodeState &&) = delete;

  TrackerNodeState &
  operator=(const TrackerNodeState &) = delete;

  TrackerNodeState &
  operator=(TrackerNodeState &&) = delete;

  [[nodiscard]] Node *
  node() const noexcept
  {
    return node_;
  }

  inline size_t
  state() const noexcept
  {
    return state_;
  }

private:
  size_t state_;
  Node * node_;
};

}

#endif
