/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEMSTATE_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEMSTATE_CONV_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class Region;
}

namespace jlm::hls
{

/**
 * Replaces LambdaEntryMemoryStateSplitOperation and MemoryStateSplitOperation nodes with
 * ForkOperation nodes.
 */
class MemoryStateSplitConversion final : public rvsdg::Transformation
{
public:
  ~MemoryStateSplitConversion() noexcept override;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector);

private:
  static void
  ConvertMemoryStateSplitsInRegion(rvsdg::Region & region);
};

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEMSTATE_CONV_HPP
