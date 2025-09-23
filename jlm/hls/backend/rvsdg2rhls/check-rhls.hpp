/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_CHECK_RHLS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_CHECK_RHLS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class RhlsVerification final : public rvsdg::Transformation
{
public:
  ~RhlsVerification() noexcept override;

  RhlsVerification();

  RhlsVerification(const RhlsVerification &) = delete;

  RhlsVerification &
  operator=(const RhlsVerification &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    RhlsVerification rhlsVerification;
    rhlsVerification.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_CHECK_RHLS_HPP
