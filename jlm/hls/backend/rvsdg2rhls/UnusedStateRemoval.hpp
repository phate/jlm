/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP

namespace jlm::llvm
{
class RvsdgModule;
}

namespace jlm::hls
{

/**
 * Remove invariant values from gamma, theta, and lambda nodes.
 *
 * @param rvsdgModule The RVSDG module the optimization is performed on.
 *
 * FIXME: This entire transformation can be expressed using llvm::InvariantValueRedirection and
 * llvm::DeadNodeElimination, and should be replaced by them. The llvm::DeadNodeElimination would
 * need to be extended to remove unused state edges in lambda nodes though.
 */
void
RemoveUnusedStates(llvm::RvsdgModule & rvsdgModule);

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

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP
