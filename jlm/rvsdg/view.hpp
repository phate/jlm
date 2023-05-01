/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_VIEW_HPP
#define JLM_RVSDG_VIEW_HPP

#include <jlm/rvsdg/graph.hpp>

#include <string>

namespace jive {

class region;

void
view(const jive::region * region, FILE * out);

static inline void
view(const jive::graph & graph, FILE * out)
{
	return view(graph.root(), out);
}

std::string
view(const jive::region * region);

std::string
region_tree(const jive::region * region);

void
region_tree(const jive::region * region, FILE * out);

std::string
to_xml(const jive::region * region);

void
view_xml(const jive::region * region, FILE * out);

}

#endif
