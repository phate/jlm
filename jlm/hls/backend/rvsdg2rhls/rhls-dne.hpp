/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class RhlsDeadNodeElimination final : public rvsdg::Transformation
{
public:
  ~RhlsDeadNodeElimination() noexcept override;

  RhlsDeadNodeElimination();

  RhlsDeadNodeElimination(const RhlsDeadNodeElimination &) = delete;

  RhlsDeadNodeElimination &
  operator=(const RhlsDeadNodeElimination &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  bool
  Run(rvsdg::Region & region, util::StatisticsCollector & statisticsCollector);

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    RhlsDeadNodeElimination deadNoneElimination;
    deadNoneElimination.Run(rvsdgModule, statisticsCollector);
  }
};

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_RHLS_DNE_HPP
