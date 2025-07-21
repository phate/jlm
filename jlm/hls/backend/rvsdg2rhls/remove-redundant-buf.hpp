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
 * Replace \ref BufferOperation nodes with \ref MemoryStateType operands that can be
 * traced to a \ref LoadOperation, \ref LocalLoadOperation, \ref StoreOperation, or \ref
 * LocalStoreOperation with a passthrough \ref BufferOperation node.
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
