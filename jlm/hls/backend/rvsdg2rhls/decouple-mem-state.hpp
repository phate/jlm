/*
 * Copyright 2025 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP
#define JLM_HLS_BACKEND_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

void
convert_loop_state_to_lcb(rvsdg::Input * loop_state_input);

class MemoryStateDecoupling final : public rvsdg::Transformation
{
public:
  ~MemoryStateDecoupling() noexcept override;

  MemoryStateDecoupling();

  MemoryStateDecoupling(const MemoryStateDecoupling &) = delete;

  MemoryStateDecoupling &
  operator=(const MemoryStateDecoupling &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector)
  {
    MemoryStateDecoupling memoryStateDecoupler;
    memoryStateDecoupler.Run(rvsdgModule, statisticsCollector);
  }
};

}

#endif // JLM_HLS_BACKEND_RVSDG2RHLS_DECOUPLE_MEM_STATE_HPP
