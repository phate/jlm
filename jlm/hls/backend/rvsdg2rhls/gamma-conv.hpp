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

bool
gamma_can_be_spec(jlm::rvsdg::gamma_node *gamma);

void
gamma_conv_nonspec(jlm::rvsdg::gamma_node *gamma);

void
gamma_conv_spec(jlm::rvsdg::gamma_node *gamma);

void
gamma_conv(jlm::rvsdg::region *region, bool allow_speculation=true);

void
gamma_conv(llvm::RvsdgModule &rm, bool allow_speculation=true);

}

#endif //JLM_HLS_BACKEND_RVSDG2RHLS_GAMMA_CONV_HPP
