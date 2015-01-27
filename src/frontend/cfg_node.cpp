/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/frontend/cfg.hpp>
#include <jlm/frontend/cfg_node.hpp>
#include <jlm/frontend/clg.hpp>

#include <string.h>

#include <algorithm>

namespace jlm {
namespace frontend {

cfg_edge::cfg_edge(cfg_node * source, cfg_node * sink, size_t index) noexcept
	: source_(source)
	, sink_(sink)
	, index_(index)
{}

cfg_node::~cfg_node() {}

cfg_edge *
cfg_node::add_outedge(cfg_node * successor, size_t index)
{
	std::unique_ptr<cfg_edge> edge(new cfg_edge(this, successor, index));
	cfg_edge * e = edge.get();
	outedges_.insert(std::move(edge));
	successor->inedges_.insert(e);
	edge.release();
	return e;
}

void
cfg_node::remove_outedge(cfg_edge * edge)
{
	std::unique_ptr<cfg_edge> e(edge);
	std::unordered_set<std::unique_ptr<cfg_edge>>::const_iterator it = outedges_.find(e);
	if (it != outedges_.end()) {
		JIVE_DEBUG_ASSERT(edge->source() == this);
		edge->sink()->inedges_.erase(edge);
		outedges_.erase(it);
	}
	e.release();
}

void
cfg_node::remove_outedges()
{
	while (outedges_.size() != 0) {
		JIVE_DEBUG_ASSERT(outedges_.begin()->get()->source() == this);
		remove_outedge(outedges_.begin()->get());
	}
}

size_t
cfg_node::noutedges() const noexcept
{
	return outedges_.size();
}

std::vector<cfg_edge*>
cfg_node::outedges() const
{
	std::vector<cfg_edge*> edges;
	std::unordered_set<std::unique_ptr<cfg_edge>>::const_iterator it;
	for ( it = outedges_.begin(); it != outedges_.end(); it++) {
		JIVE_DEBUG_ASSERT(it->get()->source() == this);
		edges.push_back(it->get());
	}

	return edges;
}

void
cfg_node::divert_inedges(cfg_node * new_successor)
{
	std::unordered_set<cfg_edge*>::const_iterator it;
	for (it = inedges_.begin(); it != inedges_.end(); it++) {
		JIVE_DEBUG_ASSERT((*it)->sink() == this);
		(*it)->divert(new_successor);
		new_successor->inedges_.insert(*it);
	}
	inedges_.clear();
}

void
cfg_node::remove_inedges()
{
	while (inedges_.size() != 0) {
		cfg_edge * edge = *inedges_.begin();
		JIVE_DEBUG_ASSERT(edge->sink() == this);
		edge->source()->remove_outedge(edge);
	}
}

size_t
cfg_node::ninedges() const noexcept
{
	return inedges_.size();
}

std::vector<cfg_edge*>
cfg_node::inedges() const
{
	std::vector<cfg_edge*> edges;
	std::unordered_set<cfg_edge*>::const_iterator it;
	for ( it = inedges_.begin(); it != inedges_.end(); it++) {
		JIVE_DEBUG_ASSERT((*it)->sink() == this);
		edges.push_back(*it);
	}

	return edges;
}

bool
cfg_node::no_predecessor() const noexcept
{
	return ninedges() == 0;
}

bool
cfg_node::single_predecessor() const noexcept
{
	if (ninedges() == 0)
		return false;

	std::unordered_set<cfg_edge*>::const_iterator it;
	for (it = inedges_.begin(); it != inedges_.end(); it++) {
		JIVE_DEBUG_ASSERT((*it)->sink() == this);
		if ((*it)->source() != (*inedges_.begin())->source())
			return false;
	}

	return true;
}

bool
cfg_node::no_successor() const noexcept
{
	return noutedges() == 0;
}

bool
cfg_node::single_successor() const noexcept
{
	if (noutedges() == 0)
		return false;

	std::unordered_set<std::unique_ptr<cfg_edge>>::const_iterator it;
	for (it = outedges_.begin(); it != outedges_.end(); it++) {
		JIVE_DEBUG_ASSERT(it->get()->source() == this);
		if ((*it)->sink() != (*outedges_.begin())->sink())
			return false;
	}

	return true;
}

bool
cfg_node::has_selfloop_edge() const noexcept
{
	std::vector<cfg_edge*> edges = outedges();
	for (auto edge : edges) {
		if (edge->is_selfloop())
			return true;
	}

	return false;
}

}
}
