/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MERGE_GAMMA_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MERGE_GAMMA_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>

namespace jlm::hls
{

void
merge_gamma(jlm::rvsdg::region * region);

void
merge_gamma(llvm::RvsdgModule & rm);

bool
merge_gamma(jlm::rvsdg::gamma_node * gamma);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MERGE_GAMMA_HPP
