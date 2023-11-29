/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_GAMMA_CONV_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_GAMMA_CONV_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm::hls
{

void
ConvertGammaNodes(llvm::RvsdgModule & rvsdgModule, bool allowSpeculation);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_GAMMA_CONV_HPP
