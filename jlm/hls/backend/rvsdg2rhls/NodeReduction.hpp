/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_NODEREDUCTION_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_NODEREDUCTION_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

bool
NodeReduction(rvsdg::Region * region);

void
NodeReduction(llvm::RvsdgModule & rvsdgModule);

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_NODEREDUCTION_HPP
