/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_ADD_BUFFERS_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_ADD_BUFFERS_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

void
setMemoryLatency(size_t memoryLatency);

class BufferInsertion final : public rvsdg::Transformation
{
public:
  ~BufferInsertion() noexcept override;

  BufferInsertion();

  BufferInsertion(const BufferInsertion &) = delete;

  BufferInsertion &
  operator=(const BufferInsertion &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    BufferInsertion bufferInsertion;
    bufferInsertion.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_ADD_BUFFERS_HPP
