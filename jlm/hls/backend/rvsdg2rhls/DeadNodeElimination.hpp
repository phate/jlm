/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_DEADNODEELIMINATION_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_DEADNODEELIMINATION_HPP

#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

void
EliminateDeadNodes(llvm::RvsdgModule & rvsdgModule);

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_DEADNODEELIMINATION_HPP
