/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_LAMBDAMEMORYSTATE_HPP
#define JLM_LLVM_IR_LAMBDAMEMORYSTATE_HPP

/**
 * \brief Global memory state passed between functions.
 *
 * This file contains various helpers to manage the memory state
 * as it is passed between llvm functions represented as lambda
 * operations, and the chosen memory model for this mapping.
 */

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

/**
 * Determines the formal argument representing global memory state
 *
 * \param lambdaNode
 *   The lambda node to query the memory state for.
 *
 * \returns
 *   The memory state argument of the lambda subregion.
 *
 * \pre
 *   The \p lambdaNode must conform to the modelling assumptions behind
 *   llvm function representation as lambdas and memory state encoding.
 */
[[nodiscard]] rvsdg::Output &
GetMemoryStateRegionArgument(const rvsdg::LambdaNode & lambdaNode) noexcept;

/**
 * Determines the formal return value representing global memory state
 *
 * \param lambdaNode
 *   The lambda node to query the memory state for.
 *
 * \returns
 *   The memory state result of the lambda subregion.
 *
 * \pre
 *   The \p lambdaNode must conform to the modelling assumptions behind
 *   llvm function representation as lambdas and memory state encoding.
 */
[[nodiscard]] rvsdg::Input &
GetMemoryStateRegionResult(const rvsdg::LambdaNode & lambdaNode) noexcept;

/**
 * Determines the memory state split node at entry.
 *
 * \param lambdaNode
 *   The lambda node to query the memory state entry split node for.
 *
 * \returns
 *   The LambdaEntryMemoryStateSplitOperation node connected to the memory
 *   state input if present, otherwise nullptr.
 *
 * \pre
 *   The \p lambdaNode must conform to the modelling assumptions behind
 *   llvm function representation as lambdas and memory state encoding.
 *
 * \see GetMemoryStateExitMerge()
 */
rvsdg::SimpleNode *
GetMemoryStateEntrySplit(const rvsdg::LambdaNode & lambdaNode) noexcept;

/**
 * Determines the memory state merge node at exit.
 *
 * \param lambdaNode
 *   The lambda node to query the memory state exit mux node for.
 *
 * \returns
 *   The LambdaEntryMemoryStateMergeOperation node connected to the memory
 *   state input if present, otherwise nullptr.
 *
 * \pre
 *   The \p lambdaNode must conform to the modelling assumptions behind
 *   llvm function representation as lambdas and memory state encoding.
 *
 * \see GetMemoryStateEntrySplit()
 */
[[nodiscard]] rvsdg::SimpleNode *
GetMemoryStateExitMerge(const rvsdg::LambdaNode & lambdaNode) noexcept;

}

#endif
