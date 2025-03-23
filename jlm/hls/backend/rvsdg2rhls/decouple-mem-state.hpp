/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
decouple_mem_state(rvsdg::Region * region);

void
decouple_mem_state(llvm::RvsdgModule & rm);

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP
