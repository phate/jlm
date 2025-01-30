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

rvsdg::LambdaNode *
remove_lambda_passthrough(rvsdg::LambdaNode * ln);

void
remove_region_passthrough(const rvsdg::RegionArgument * arg);

void
remove_gamma_passthrough(rvsdg::GammaNode * gn);

void
remove_unused_state(llvm::RvsdgModule & rm);

void
remove_unused_state(rvsdg::Region * region, bool can_remove_arguments = true);

/**
 * @brief Removes invariant state edges from the lambdas in the RVSDG module.
 *
 * The pass replaces the lambda with a new function signature if a state edge is found to be
 * invariant.
 *
 * @param rvsdgModule The RVSDG module for which to remove invariant state edges.
 */
void
RemoveInvariantLambdaStateEdges(llvm::RvsdgModule & rvsdgModule);

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
