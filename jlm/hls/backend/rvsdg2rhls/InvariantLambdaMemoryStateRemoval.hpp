/*
 * Copyright 2025 Magnus Själander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

/**
 * @brief Remove invariant memory state edges between Lambda[Entry/Exit]MemoryState nodes.
 *
 * The transformation goes through all lambdas in the RVSDG module and checks if the lambda is only
 * exported. If the lambda is only exported then a check is made for a LambdaExitMemoryStateMerge.
 * If one is found, a check for any invariant edges to its corresponding LambdaEntryMemoryStateSplit
 * node is performed. Any found invariant memory state edges are removed. The memory state split and
 * merge nodes are removed if there is only a single none-invariant edge that remains.
 */
class InvariantLambdaMemoryStateRemoval final : public rvsdg::Transformation
{
public:
  virtual ~InvariantLambdaMemoryStateRemoval();

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;
};

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_INVARIANTLAMBDAMEMORYSTATEREMOVAL_HPP
