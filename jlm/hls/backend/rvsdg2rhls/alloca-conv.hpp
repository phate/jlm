/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_ALLOCA_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_ALLOCA_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::hls
{

void
alloca_conv(rvsdg::Region * region);

void
alloca_conv(llvm::RvsdgModule & rm);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_ALLOCA_CONV_HPP
