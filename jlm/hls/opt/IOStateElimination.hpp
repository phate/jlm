/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_OPT_IOSTATEELIMINATION_HPP
#define JLM_HLS_OPT_IOSTATEELIMINATION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class Output;
class Region;
}

namespace jlm::hls
{

/**
 * This pass ensures that all IO state outputs in the graph become dead by making the IO state
 * inputs of all nodes directly dependent on the IO state argument of the lambda nodes. This
 * effectively renders all nodes independent with respect to IO states.
 *
 * FIXME: This pass is misnamed.
 * FIXME: We might want to merge this pass with the jlm::hls::IOBarrierRemoval transformation
 */
class IOStateElimination final : public rvsdg::Transformation
{
public:
  ~IOStateElimination() noexcept override;

  IOStateElimination()
      : Transformation("IOStateElimination")
  {}

  IOStateElimination(const IOStateElimination &) = delete;

  IOStateElimination &
  operator=(const IOStateElimination &) = delete;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  eliminateIOStates(rvsdg::Region & region, rvsdg::Output & ioStateArgument);
};

}

#endif
