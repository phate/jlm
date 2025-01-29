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

/**
 * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
 *
 * The pass checks for a LambdaExitMemoryStateMerge and if found checks if there is any invariant
 * edges to its corresponding LambdaEntryMemoryStateSplit node. Any found invariant memory state
 * edge(s) are removed. The memory state split and merge nodes are removed if there is only a single
 * none-invariant edge.
 *
 * @param memoryState The lambda region result for which invariant memory state edges are to be
 * removed.
 */
void
RemoveInvariantMemoryStateEdges(rvsdg::RegionResult * memoryState);

/**
 * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
 *
 * The pass checks all lambdas in the module for invariant memory state edges between a
 * LambdaEntreyMemoryStateSplit and LambdaExitMemoryStateMerge node and removes them. The memory
 * state split and merge nodes are removed if there is only a single none-invariant edge.
 *
 * @param rvsdgModule The RVSDG moduled for which invariant memory state edges in all lambda nodes
 * are to be removed.
 */
void
RemoveLambdaInvariantMemoryStateEdges(llvm::RvsdgModule & rvsdgModule);

/**
 * @brief Removes invariant state edges from the lambdas in the RVSDG module.
 *
 * The pass replaces the lambda with a new function signature if a state edge is found to be
 * invariant.
 *
 * @param rvsdgModule The RVSDG module for which to remove invariant state edges.
 */
void
RemoveLambdaInvariantStateEdges(llvm::RvsdgModule & rvsdgModule);

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_REMOVE_UNUSED_STATE_HPP
