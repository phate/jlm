/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_BUFFERS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_BUFFERS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

void
add_buffers(rvsdg::Region * region, bool pass_through);

void
add_buffers(llvm::RvsdgModule & rm, bool pass_through);


void setMemoryLatency(size_t memoryLatency);
}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_BUFFERS_HPP
