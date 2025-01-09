/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_VIEW_HPP
#define JLM_RVSDG_VIEW_HPP

#include <jlm/rvsdg/graph.hpp>

#include <string>
#include <unordered_map>

namespace jlm::rvsdg
{

class Region;

/**
 * Prints the given rvsdg region to a string,
 * through recursive traversal of nodes and subregions.
 * All outputs are given unique names to show dataflow in the graph.
 * @param region the region to be printed
 * @return the string describing the region.
 * @see view(region, map)
 */
std::string
view(const rvsdg::Region * region);

/**
 * Prints the given rvsdg region to a string, and exposes the unique name given to each output.
 * @param region the region to be printed
 * @param map the map where each rvsdg::output is mapped to its unique name.
 * Outputs without names will have a name added.
 * @return the string describing the region.
 */
std::string
view(const rvsdg::Region * region, std::unordered_map<const output *, std::string> & map);

/**
 * Recursively traverses and prints the given rvsdg region to the given file.
 * @param region the region to be printed.
 * @param out the file to be written to.
 */
void
view(const rvsdg::Region * region, FILE * out);

/**
 * Recursively traverses and prints the root region of the given rvsdg graph to the given file.
 * @param graph the rvsdg graph to be printed.
 * @param out the file to be written to.
 */
inline void
view(const Graph & graph, FILE * out)
{
  return view(&graph.GetRootRegion(), out);
}

std::string
to_xml(const rvsdg::Region * region);

void
view_xml(const rvsdg::Region * region, FILE * out);

}

#endif
