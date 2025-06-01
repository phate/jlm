/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/tracker.hpp>

using namespace std::placeholders;

namespace
{

typedef std::unordered_set<const jlm::rvsdg::Graph *> tracker_set;

tracker_set *
active_trackers()
{
  static std::unique_ptr<tracker_set> trackers;
  if (!trackers)
    trackers.reset(new tracker_set());

  return trackers.get();
}

void
register_tracker(const jlm::rvsdg::tracker * tracker)
{
  active_trackers()->insert(tracker->graph());
}

void
unregister_tracker(const jlm::rvsdg::tracker * tracker)
{
  active_trackers()->erase(tracker->graph());
}

}

namespace jlm::rvsdg
{

bool
has_active_trackers(const Graph * graph)
{
  auto at = active_trackers();
  return at->find(graph) != at->end();
}

/* tracker depth state */

class tracker_depth_state
{
public:
  inline tracker_depth_state()
      : count_(0),
        top_depth_(0),
        bottom_depth_(0)
  {}

  tracker_depth_state(const tracker_depth_state &) = delete;

  tracker_depth_state(tracker_depth_state &&) = delete;

  tracker_depth_state &
  operator=(const tracker_depth_state &) = delete;

  tracker_depth_state &
  operator=(tracker_depth_state &&) = delete;

  inline TrackerNodeState *
  peek_top() const noexcept
  {
    return count_ ? *nodestates_.at(top_depth_).begin() : nullptr;
  }

  inline TrackerNodeState *
  peek_bottom() const noexcept
  {
    return count_ ? *nodestates_.at(bottom_depth_).begin() : nullptr;
  }

  inline void
  add(TrackerNodeState * nodestate, size_t depth)
  {
    auto it = nodestates_.find(depth);
    if (it != nodestates_.end())
      it->second.insert(nodestate);
    else
      nodestates_[depth] = { nodestate };

    count_++;
    if (count_ == 1)
    {
      top_depth_ = depth;
      bottom_depth_ = depth;
    }
    else
    {
      if (depth < top_depth_)
        top_depth_ = depth;
      if (depth > bottom_depth_)
        bottom_depth_ = depth;
    }
  }

  inline void
  remove(TrackerNodeState * nodestate, size_t depth)
  {
    nodestates_[depth].erase(nodestate);

    count_--;
    if (count_ == 0)
      return;

    if (depth == top_depth_)
    {
      while (nodestates_[top_depth_].empty())
        top_depth_++;
    }

    if (depth == bottom_depth_)
    {
      while (nodestates_[bottom_depth_].empty())
        bottom_depth_--;
    }

    JLM_ASSERT(top_depth_ <= bottom_depth_);
  }

  inline TrackerNodeState *
  pop_top()
  {
    auto nodestate = peek_top();
    if (nodestate)
      remove(nodestate, top_depth_);

    return nodestate;
  }

  inline TrackerNodeState *
  pop_bottom()
  {
    auto nodestate = peek_bottom();
    if (nodestate)
      remove(nodestate, bottom_depth_);

    return nodestate;
  }

private:
  size_t count_;
  size_t top_depth_;
  size_t bottom_depth_;
  std::unordered_map<size_t, std::unordered_set<TrackerNodeState *>> nodestates_;
};

/* tracker */

tracker::~tracker() noexcept
{
  unregister_tracker(this);
}

tracker::tracker(Graph * graph, size_t nstates)
    : graph_(graph),
      states_(nstates)
{
  for (size_t n = 0; n < states_.size(); n++)
    states_[n] = std::make_unique<tracker_depth_state>();

  depth_callback_ =
      on_node_depth_change.connect(std::bind(&tracker::node_depth_change, this, _1, _2));
  destroy_callback_ = on_node_destroy.connect(std::bind(&tracker::node_destroy, this, _1));

  register_tracker(this);
}

void
tracker::node_depth_change(Node * node, size_t old_depth)
{
  auto nstate = nodestate(node);
  if (nstate->state() < states_.size())
  {
    states_[nstate->state()]->remove(nstate, old_depth);
    states_[nstate->state()]->add(nstate, node->depth());
  }
}

void
tracker::node_destroy(Node * node)
{
  auto nstate = nodestate(node);
  if (nstate->state() < states_.size())
    states_[nstate->state()]->remove(nstate, node->depth());

  nodestates_.erase(node);
}

ssize_t
tracker::get_nodestate(Node * node)
{
  return nodestate(node)->state();
}

void
tracker::set_nodestate(Node * node, size_t state)
{
  auto nstate = nodestate(node);
  if (nstate->state() != state)
  {
    if (nstate->state() < states_.size())
      states_[nstate->state()]->remove(nstate, node->depth());

    nstate->state_ = state;
    if (nstate->state() < states_.size())
      states_[nstate->state()]->add(nstate, node->depth());
  }
}

Node *
tracker::peek_top(size_t state) const
{
  JLM_ASSERT(state < states_.size());

  auto nodestate = states_[state]->pop_top();
  if (nodestate)
  {
    nodestate->state_ = tracker_nodestate_none;
    return nodestate->node();
  }

  return nullptr;
}

Node *
tracker::peek_bottom(size_t state) const
{
  JLM_ASSERT(state < states_.size());

  auto nodestate = states_[state]->pop_bottom();
  if (nodestate)
  {
    nodestate->state_ = tracker_nodestate_none;
    return nodestate->node();
  }

  return nullptr;
}

jlm::rvsdg::TrackerNodeState *
tracker::nodestate(Node * node)
{
  auto it = nodestates_.find(node);
  if (it != nodestates_.end())
    return it->second.get();

  nodestates_[node] = std::make_unique<jlm::rvsdg::TrackerNodeState>(node);
  return nodestates_[node].get();
}

}
