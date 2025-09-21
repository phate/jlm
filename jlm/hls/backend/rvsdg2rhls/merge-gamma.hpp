/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MERGE_GAMMA_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MERGE_GAMMA_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class GammaMerge final : public rvsdg::Transformation
{
public:
  ~GammaMerge() noexcept override;

  GammaMerge();

  GammaMerge(const GammaMerge &) = delete;

  GammaMerge(GammaMerge &&) = delete;

  GammaMerge &
  operator=(const GammaMerge &) = delete;

  GammaMerge &
  operator=(GammaMerge &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    GammaMerge gammaMerge;
    gammaMerge.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MERGE_GAMMA_HPP
