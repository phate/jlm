/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_ALLOCA_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_ALLOCA_CONV_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class AllocaNodeConversion final : public rvsdg::Transformation
{
public:
  ~AllocaNodeConversion() noexcept override;

  AllocaNodeConversion();

  AllocaNodeConversion(const AllocaNodeConversion &) = delete;

  AllocaNodeConversion &
  operator=(const AllocaNodeConversion &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    AllocaNodeConversion allocaNodeConversion;
    allocaNodeConversion.Run(rvsdgModule, statisticsCollector);
  }
};

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_ALLOCA_CONV_HPP
