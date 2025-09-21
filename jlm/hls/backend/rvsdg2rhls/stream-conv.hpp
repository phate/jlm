/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class StreamConversion final : public rvsdg::Transformation
{
public:
  ~StreamConversion() noexcept override;

  StreamConversion();

  StreamConversion(const StreamConversion &) = delete;

  StreamConversion &
  operator=(const StreamConversion &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    StreamConversion streamConversion;
    streamConversion.Run(rvsdgModule, statisticsCollector);
  }
};

}
#endif // JLM_HLS_BACKEND_RVSDG2RHLS_STREAM_CONV_HPP
