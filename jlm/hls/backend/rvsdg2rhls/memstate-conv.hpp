/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEMSTATE_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEMSTATE_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
memstate_conv(rvsdg::Region * region);

void
memstate_conv(llvm::RvsdgModule & rm);

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEMSTATE_CONV_HPP
