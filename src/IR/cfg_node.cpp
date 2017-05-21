/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/IR/basic_block.hpp>
#include <jlm/IR/cfg.hpp>
#include <jlm/IR/cfg_node.hpp>
#include <jlm/IR/clg.hpp>

#include <string.h>

#include <algorithm>

namespace jlm {

/* attribute */

attribute::~attribute()
{}

/* edge */

cfg_edge::cfg_edge(cfg_node * source, cfg_node * sink, size_t index) noexcept
	: source_(source)
	, sink_(sink)
	, index_(index)
{}

void
cfg_edge::divert(cfg_node * new_sink)
{
	if (sink_ == new_sink)
		return;

	sink_->inedges_.remove(this);
	sink_ = new_sink;
	new_sink->inedges_.push_back(this);
}

cfg_node *
cfg_edge::split()
{
	auto bb = create_basic_block_node(source_->cfg());
	auto i = sink_->inedges_.erase(std::find(sink_->inedges_.begin(), sink_->inedges_.end(), this));

	std::unique_ptr<cfg_edge> edge(new jlm::cfg_edge(bb, sink_, 0));
	cfg_edge * e = edge.get();
	bb->outedges_.insert(std::move(edge));
	sink_->inedges_.insert(i, e);

	sink_ = bb;
	bb->inedges_.push_back(this);

	return bb;
}

cfg_node::~cfg_node() {}

cfg_edge *
cfg_node::add_outedge(cfg_node * successor, size_t index)
{
	std::unique_ptr<cfg_edge> edge(new cfg_edge(this, successor, index));
	cfg_edge * e = edge.get();
	outedges_.insert(std::move(edge));
	successor->inedges_.push_back(e);
	edge.release();
	return e;
}

void
cfg_node::remove_outedge(cfg_edge * edge)
{
	std::unique_ptr<cfg_edge> e(edge);
	std::unordered_set<std::unique_ptr<cfg_edge>>::const_iterator it = outedges_.find(e);
	if (it != outedges_.end()) {
		JLM_DEBUG_ASSERT(edge->source() == this);
		edge->sink()->inedges_.remove(edge);
		outedges_.erase(it);
	}
	e.release();
}

void
cfg_node::remove_outedges()
{
	while (outedges_.size() != 0) {
		JLM_DEBUG_ASSERT(outedges_.begin()->get()->source() == this);
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
		JLM_DEBUG_ASSERT(it->get()->source() == this);
		edges.push_back(it->get());
	}

	return edges;
}

void
cfg_node::divert_inedges(cfg_node * new_successor)
{
	while (inedges_.size())
		inedges_.front()->divert(new_successor);
}

void
cfg_node::remove_inedges()
{
	while (inedges_.size() != 0) {
		cfg_edge * edge = *inedges_.begin();
		JLM_DEBUG_ASSERT(edge->sink() == this);
		edge->source()->remove_outedge(edge);
	}
}

size_t
cfg_node::ninedges() const noexcept
{
	return inedges_.size();
}

std::list<cfg_edge*>
cfg_node::inedges() const
{
	return inedges_;
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

	for (auto i = inedges_.begin(); i != inedges_.end(); i++) {
		JLM_DEBUG_ASSERT((*i)->sink() == this);
		if ((*i)->source() != (*inedges_.begin())->source())
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
		JLM_DEBUG_ASSERT(it->get()->source() == this);
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

std::string
cfg_node::debug_string() const
{
	return "node";
}

}
