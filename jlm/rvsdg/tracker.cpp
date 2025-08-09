/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/tracker.hpp>

#include <set>

using namespace std::placeholders;

namespace jlm::rvsdg
{

struct cmp
{
  bool
  operator()(const TrackerNodeState * lhs, const TrackerNodeState * rhs) const
  {
    return lhs->node()->GetNodeId() < rhs->node()->GetNodeId();
  }
};

class TrackerDepthState
{
public:
  TrackerDepthState()
      : count_(0),
        top_depth_(0),
        bottom_depth_(0)
  {}

  TrackerDepthState(const TrackerDepthState &) = delete;

  TrackerDepthState(TrackerDepthState &&) = delete;

  TrackerDepthState &
  operator=(const TrackerDepthState &) = delete;

  TrackerDepthState &
  operator=(TrackerDepthState &&) = delete;

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
  std::unordered_map<size_t, std::set<TrackerNodeState *, cmp>> nodestates_;
};

Tracker::~Tracker() noexcept
{
  [[maybe_unused]] const auto isUnregistered = GetRegion().UnregisterTracker(*this);
  JLM_ASSERT(isUnregistered);
}

Tracker::Tracker(Region & region, size_t nstates)
    : Region_(&region),
      states_(nstates)
{
  for (size_t n = 0; n < states_.size(); n++)
    states_[n] = std::make_unique<TrackerDepthState>();

  depth_callback_ =
      on_node_depth_change.connect(std::bind(&Tracker::node_depth_change, this, _1, _2));
  destroy_callback_ = on_node_destroy.connect(std::bind(&Tracker::node_destroy, this, _1));

  [[maybe_unused]] const auto isRegistered = region.RegisterTracker(*this);
  JLM_ASSERT(isRegistered);
}

void
Tracker::node_depth_change(Node * node, size_t old_depth)
{
  auto nstate = nodestate(node);
  if (nstate->state() < states_.size())
  {
    states_[nstate->state()]->remove(nstate, old_depth);
    states_[nstate->state()]->add(nstate, node->depth());
  }
}

void
Tracker::node_destroy(Node * node)
{
  auto nstate = nodestate(node);
  if (nstate->state() < states_.size())
    states_[nstate->state()]->remove(nstate, node->depth());

  nodestates_.erase(node);
}

ssize_t
Tracker::get_nodestate(Node * node)
{
  return nodestate(node)->state();
}

void
Tracker::set_nodestate(Node * node, size_t state)
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
Tracker::peek_top(size_t state) const
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
Tracker::peek_bottom(size_t state) const
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
Tracker::nodestate(Node * node)
{
  auto it = nodestates_.find(node);
  if (it != nodestates_.end())
    return it->second.get();

  nodestates_[node] = std::make_unique<jlm::rvsdg::TrackerNodeState>(node);
  return nodestates_[node].get();
}

}
