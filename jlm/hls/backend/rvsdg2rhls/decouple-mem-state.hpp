/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
decouple_mem_state(rvsdg::Region * region);

void
decouple_mem_state(llvm::RvsdgModule & rm);

void
convert_loop_state_to_lcb(rvsdg::input * loop_state_input);
}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP
