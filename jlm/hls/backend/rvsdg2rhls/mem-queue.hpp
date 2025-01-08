/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
mem_queue(rvsdg::Region * region);

void
mem_queue(llvm::RvsdgModule & rm);

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_QUEUE_HPP
