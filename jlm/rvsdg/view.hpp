/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_VIEW_HPP
#define JLM_RVSDG_VIEW_HPP

#include <jlm/rvsdg/graph.hpp>

#include <string>

namespace jlm::rvsdg
{

class region;

void
view(const jlm::rvsdg::region * region, FILE * out);

static inline void
view(const jlm::rvsdg::graph & graph, FILE * out)
{
  return view(graph.root(), out);
}

std::string
view(const jlm::rvsdg::region * region);

std::string
region_tree(const jlm::rvsdg::region * region);

void
region_tree(const jlm::rvsdg::region * region, FILE * out);

std::string
to_xml(const jlm::rvsdg::region * region);

void
view_xml(const jlm::rvsdg::region * region, FILE * out);

}

#endif
