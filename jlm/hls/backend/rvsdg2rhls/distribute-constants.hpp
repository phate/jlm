/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_DISTRIBUTE_CONSTANTS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_DISTRIBUTE_CONSTANTS_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class ConstantDistribution final : rvsdg::Transformation
{
public:
  ~ConstantDistribution() noexcept override;

  ConstantDistribution();

  ConstantDistribution(const ConstantDistribution &) = delete;

  ConstantDistribution(ConstantDistribution &&) = delete;

  ConstantDistribution &
  operator=(const ConstantDistribution &) = delete;

  ConstantDistribution &
  operator=(ConstantDistribution &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    ConstantDistribution constantDistribution;
    constantDistribution.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif
