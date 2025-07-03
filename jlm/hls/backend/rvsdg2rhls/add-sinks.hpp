/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_SINKS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_SINKS_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class Region;
}

namespace jlm::hls
{

/**
 * Adds sink nodes to every output that has no users.
 */
class SinkInsertion final : public rvsdg::Transformation
{
public:
  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector);

private:
  static void
  AddSinksToRegion(rvsdg::Region & region);
};

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_SINKS_HPP
