/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP

#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

bool
remove_unused_loop_outputs(hls::loop_node * ln);

bool
remove_unused_loop_backedges(hls::loop_node * ln);

bool
remove_loop_passthrough(hls::loop_node * ln);

bool
remove_unused_loop_inputs(hls::loop_node * ln);

bool
dne(rvsdg::Region * sr);

void
dne(llvm::RvsdgModule & rm);

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP
