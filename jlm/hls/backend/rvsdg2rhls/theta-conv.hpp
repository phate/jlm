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

/**
 * Converts an rvsdg::theta_node to an hls::loop_node.
 *
 * @param rvsdgModule The RVSDG module the transformation is performed on.
 */
void
theta_conv(jlm::llvm::RvsdgModule & rvsdgModule);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_THETA_CONV_HPP
