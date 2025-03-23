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
region_to_dot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color);

std::string
to_dot(
    rvsdg::Region * region,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color);

void
view_dot(rvsdg::Region * region, FILE * out);
void
view_dot(
    rvsdg::Region * region,
    FILE * out,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color);

void
dump_dot(
    llvm::RvsdgModule & rvsdgModule,
    const std::string & file_name,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color);
void

dump_dot(llvm::RvsdgModule & rvsdgModule, const std::string & file_name);

void
dump_dot(
    rvsdg::Region * region,
    const std::string & file_name,
    std::unordered_map<rvsdg::output *, std::string> & o_color,
    std::unordered_map<rvsdg::input *, std::string> & i_color);
void
dump_dot(rvsdg::Region * region, const std::string & file_name);

void
dot_to_svg(const std::string & file_name);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_UTIL_VIEW_HPP
