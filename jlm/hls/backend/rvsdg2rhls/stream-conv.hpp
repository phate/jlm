/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
stream_conv(llvm::RvsdgModule & rm);

}
#endif // JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP
