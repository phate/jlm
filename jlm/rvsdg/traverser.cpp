/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/traverser.hpp>

using namespace std::placeholders;

/* top down traverser */

namespace jive {

topdown_traverser::~topdown_traverser() noexcept {}

topdown_traverser::topdown_traverser(jive::region * region)
	: region_(region)
	, tracker_(region->graph())
{
	for (auto & node : region->top_nodes)
		tracker_.set_nodestate(&node, traversal_nodestate::frontier);

	for (size_t n = 0; n < region->narguments(); n++) {
		auto argument = region->argument(n);
		for (const auto & user : *argument) {
			if (!is<node_input>(*user))
				continue;

			auto node = static_cast<node_input*>(user)->node();
			if (!predecessors_visited(node))
				continue;

			tracker_.set_nodestate(node, traversal_nodestate::frontier);
		}
	}

	callbacks_.push_back(on_node_create.connect(
		std::bind(&topdown_traverser::node_create, this, _1)));
	callbacks_.push_back(on_input_change.connect(
		std::bind(&topdown_traverser::input_change, this, _1, _2, _3)));
}

bool
topdown_traverser::predecessors_visited(const jive::node * node) noexcept
{
	for (size_t n = 0; n < node->ninputs(); n++) {
		auto predecessor = node_output::node(node->input(n)->origin());
		if (!predecessor)
			continue;

		if (tracker_.get_nodestate(predecessor) != traversal_nodestate::behind)
			return false;
	}

	return true;
}

jive::node *
topdown_traverser::next()
{
	jive::node * node = tracker_.peek_top();
	if (!node) return nullptr;

	tracker_.set_nodestate(node, traversal_nodestate::behind);
	for (size_t n = 0; n < node->noutputs(); n++) {
		for (const auto & user : *node->output(n)) {
			if (!is<node_input>(*user))
				continue;

			auto node = static_cast<node_input*>(user)->node();
			if (tracker_.get_nodestate(node) == traversal_nodestate::ahead)
				tracker_.set_nodestate(node, traversal_nodestate::frontier);
		}
	}

	return node;
}

void
topdown_traverser::node_create(jive::node * node)
{
	if (node->region() != region())
		return;

	if (predecessors_visited(node))
		tracker_.set_nodestate(node, traversal_nodestate::behind);
	else
		tracker_.set_nodestate(node, traversal_nodestate::frontier);
}

void
topdown_traverser::input_change(input * in, output * old_origin, output * new_origin)
{
	if (in->region() != region() || !is<node_input>(*in))
		return;

	auto node = static_cast<node_input*>(in)->node();
	auto state = tracker_.get_nodestate(node);
	
	/* ignore nodes that have been traversed already, or that are already
	marked for later traversal */
	if (state != traversal_nodestate::ahead)
		return;
	
	/* make sure node is visited eventually, might now be visited earlier
	as depth of the node could be lowered */
	tracker_.set_nodestate(node, traversal_nodestate::frontier);
}

/* bottom up traverser */

bottomup_traverser::~bottomup_traverser() noexcept {}

bottomup_traverser::bottomup_traverser(jive::region * region, bool revisit)
	: region_(region)
	, tracker_(region->graph())
	, new_node_state_(revisit ? traversal_nodestate::frontier : traversal_nodestate::behind)
{
	for (auto & node : region->bottom_nodes)
		tracker_.set_nodestate(&node, traversal_nodestate::frontier);

	for (size_t n = 0; n < region->nresults(); n++) {
		auto node = node_output::node(region->result(n)->origin());
		if (node && !node->has_successors())
			tracker_.set_nodestate(node, traversal_nodestate::frontier);
	}

	callbacks_.push_back(on_node_create.connect(
		std::bind(&bottomup_traverser::node_create, this, _1)));
	callbacks_.push_back(on_node_destroy.connect(
		std::bind(&bottomup_traverser::node_destroy, this, _1)));
	callbacks_.push_back(on_input_change.connect(
		std::bind(&bottomup_traverser::input_change, this, _1, _2, _3)));
}

jive::node *
bottomup_traverser::next()
{
	auto node = tracker_.peek_bottom();
	if (!node) return nullptr;

	tracker_.set_nodestate(node, traversal_nodestate::behind);
	for (size_t n = 0; n < node->ninputs(); n++) {
		auto producer = node_output::node(node->input(n)->origin());
		if (producer && tracker_.get_nodestate(producer) == traversal_nodestate::ahead)
			tracker_.set_nodestate(producer, traversal_nodestate::frontier);
	}
	return node;
}

void
bottomup_traverser::node_create(jive::node * node)
{
	if (node->region() != region())
		return;

	tracker_.set_nodestate(node, new_node_state_);
}

void
bottomup_traverser::node_destroy(jive::node * node)
{
	if (node->region() != region())
		return;

	for (size_t n = 0; n < node->ninputs(); n++) {
		auto producer = node_output::node(node->input(n)->origin());
		if (producer && tracker_.get_nodestate(producer) == traversal_nodestate::ahead)
			tracker_.set_nodestate(producer, traversal_nodestate::frontier);
	}
}

void
bottomup_traverser::input_change(input * in, output * old_origin, output * new_origin)
{
	if (in->region() != region()
	|| !is<node_input>(*in)
	|| !is<node_output>(old_origin))
		return;

	auto node = node_output::node(old_origin);
	traversal_nodestate state = tracker_.get_nodestate(node);
	
	/* ignore nodes that have been traversed already, or that are already
	marked for later traversal */
	if (state != traversal_nodestate::ahead)
		return;
	
	/* make sure node is visited eventually, might now be visited earlier
	as there (potentially) is one less obstructing node below */
	tracker_.set_nodestate(node, traversal_nodestate::frontier);
}

}
