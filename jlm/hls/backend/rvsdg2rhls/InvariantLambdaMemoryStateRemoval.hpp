/*
 * Copyright 2025 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

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
RemoveInvariantLambdaMemoryStateEdges(llvm::RvsdgModule & rvsdgModule);

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
