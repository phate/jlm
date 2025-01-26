/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/region.hpp>

namespace jlm::hls
{

bool
is_passthrough(const rvsdg::output * arg);

bool
is_passthrough(const rvsdg::input * res);

llvm::lambda::node *
remove_lambda_passthrough(llvm::lambda::node * ln);

void
remove_region_passthrough(const rvsdg::RegionArgument * arg);

void
remove_gamma_passthrough(rvsdg::GammaNode * gn);

void
remove_unused_state(llvm::RvsdgModule & rm);

void
remove_unused_state(rvsdg::Region * region, bool can_remove_arguments = true);

void
RemoveInvariantMemoryStateEdges(jlm::rvsdg::RegionResult * memoryState);

void
RemoveLambdaInvariantMemoryStateEdges(llvm::RvsdgModule & rvsdgModule);

void
RemoveLambdaInvariantStateEdges(llvm::RvsdgModule & rvsdgModule);

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
