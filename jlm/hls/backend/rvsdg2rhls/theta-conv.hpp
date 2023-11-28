/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_THETA_CONV_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_THETA_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::hls
{

void
theta_conv(jlm::llvm::RvsdgModule & rm);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_THETA_CONV_HPP
