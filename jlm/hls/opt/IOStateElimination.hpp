/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_IOSTATEELIMINATION_HPP
#define JLM_HLS_OPT_IOSTATEELIMINATION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::hls
{

class IOStateElimination final : public rvsdg::Transformation
{
public:
  ~IOStateElimination() noexcept override;

  IOStateElimination()
      : Transformation("IOStateElimination")
  {}

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;
};

}

#endif
