/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

/**
 * Remove invariant values from gamma, theta, and lambda nodes.
 *
 * FIXME: This entire transformation can be expressed using llvm::InvariantValueRedirection and
 * llvm::UnusedStateRemoval, and should be replaced by them. The llvm::UnusedStateRemoval would
 * need to be extended to remove unused state edges in lambda nodes though.
 */
class UnusedStateRemoval final : public rvsdg::Transformation
{
public:
  ~UnusedStateRemoval() noexcept override;

  UnusedStateRemoval();

  UnusedStateRemoval(const UnusedStateRemoval &) = delete;

  UnusedStateRemoval(UnusedStateRemoval &&) = delete;

  UnusedStateRemoval &
  operator=(const UnusedStateRemoval &) = delete;

  UnusedStateRemoval &
  operator=(UnusedStateRemoval &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    UnusedStateRemoval unusedStateRemoval;
    unusedStateRemoval.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_UNUSEDSTATEREMOVAL_HPP
