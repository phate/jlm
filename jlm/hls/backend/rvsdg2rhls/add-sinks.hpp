/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_SINKS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_SINKS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

void
add_sinks(rvsdg::Region * region);

void
add_sinks(jlm::llvm::RvsdgModule & rm);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_SINKS_HPP
