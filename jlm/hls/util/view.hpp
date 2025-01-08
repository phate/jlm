/*
 * Copyright 2024 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_UTIL_VIEW_HPP
#define JLM_BACKEND_HLS_UTIL_VIEW_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <string>

namespace jlm::hls
{

std::string
region_to_dot(rvsdg::Region * region);

std::string
to_dot(rvsdg::Region * region);

void
view_dot(rvsdg::Region * region, FILE * out);

void
dump_dot(llvm::RvsdgModule & rvsdgModule, const std::string & file_name);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_UTIL_VIEW_HPP
