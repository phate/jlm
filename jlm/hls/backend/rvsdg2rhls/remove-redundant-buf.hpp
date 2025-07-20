/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_REDUNDANT_BUF_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_REDUNDANT_BUF_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class Output;
class Region;
}

namespace jlm::hls
{

/**
 * Replace BufferOperation nodes whose operands can be traced to a LoadOperation,
 * LocalLoadOperation, StoreOperation, or LocalStoreOperation with a passthrough BufferOperation
 * node.
 *
 * FIXME: This pass should be renamed as it technically does not eliminate BufferOperations, but
 * just converts them to passthrough BufferOperations
 */
class RedundantBufferElimination final : public rvsdg::Transformation
{
public:
  ~RedundantBufferElimination() noexcept override;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector);

private:
  static void
  HandleRegion(rvsdg::Region & region);

  [[nodiscard]] static bool
  CanTraceToLoadOrStore(const rvsdg::Output & output);
};

}

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_REMOVE_REDUNDANT_BUF_HPP
