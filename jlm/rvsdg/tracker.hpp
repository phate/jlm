/*
 * Copyright 2010 2011 2012 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRACKER_HPP
#define JLM_RVSDG_TRACKER_HPP

#include <stdbool.h>
#include <stddef.h>

#include <unordered_map>

#include <jlm/util/callbacks.hpp>

namespace jive {

static const size_t tracker_nodestate_none = (size_t) -1;

class graph;
class node;
class region;
class tracker_depth_state;
class tracker_nodestate;

bool
has_active_trackers(const jive::graph * graph);

/* Track states of nodes within the graph. Each node can logically be in
 * one of the numbered states, plus another "initial" state. All nodes are
 * at the beginning assumed to be implicitly in this "initial" state. */
struct tracker {
public:
	~tracker() noexcept;
	
	tracker(jive::graph * graph, size_t nstates);

	/* get state of the node */
	ssize_t
	get_nodestate(jive::node * node);

	/* set state of the node */
	void
	set_nodestate(jive::node * node, size_t state);

	/* get one of the top nodes for the given state */
	jive::node *
	peek_top(size_t state) const;

	/* get one of the bottom nodes for the given state */
	jive::node *
	peek_bottom(size_t state) const;

	inline jive::graph *
	graph() const noexcept
	{
		return graph_;
	}

private:
	jive::tracker_nodestate *
	nodestate(jive::node * node);

	void
	node_depth_change(jive::node * node, size_t old_depth);

	void
	node_destroy(jive::node * node);

	jive::graph * graph_;

	/* FIXME: need RAII idiom for state reservation */
	std::vector<std::unique_ptr<tracker_depth_state>> states_;

	callback depth_callback_, destroy_callback_;

	std::unordered_map<jive::node*, std::unique_ptr<jive::tracker_nodestate>> nodestates_;
};

class tracker_nodestate {
	friend tracker;
public:
	inline
	tracker_nodestate(jive::node * node)
	: state_(tracker_nodestate_none)
	, node_(node)
	{}

	tracker_nodestate(const tracker_nodestate&) = delete;

	tracker_nodestate(tracker_nodestate&&) = delete;

	tracker_nodestate &
	operator=(const tracker_nodestate&) = delete;

	tracker_nodestate &
	operator=(tracker_nodestate&&) = delete;

	inline jive::node *
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
	jive::node * node_;
};

}

#endif
