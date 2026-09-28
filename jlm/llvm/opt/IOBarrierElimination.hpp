/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_IOBARRIERELIMINATION_HPP
#define JLM_LLVM_OPT_IOBARRIERELIMINATION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

class IOBarrierElimination final : public rvsdg::Transformation
{
  class Context;
  class HoistContext;
  class Statistics;

public:
  ~IOBarrierElimination() override;

  IOBarrierElimination();

  IOBarrierElimination(const IOBarrierElimination &) = delete;

  IOBarrierElimination(IOBarrierElimination &&) = delete;

  IOBarrierElimination &
  operator=(const IOBarrierElimination &) = delete;

  IOBarrierElimination &
  operator=(IOBarrierElimination &&) = delete;

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  static void
  normalizeMemoryHoistBarriers(rvsdg::Region & region);

private:
  void
  markOutputs(const rvsdg::Region & region);

  void
  propagateSize(rvsdg::Graph & graph);

  void
  encodeSize(rvsdg::Graph & graph);

  // FIXME: All this hoisting logic was duplicated from the NodeHoisting.[cpp/hpp]. Unify it again.
  void
  hoistMemoryBarriers(rvsdg::Region & region);

  void
  computeTargetRegions(const rvsdg::Region & region);

  rvsdg::Region &
  computeTargetRegion(const rvsdg::Node & node) const;

  rvsdg::Region &
  computeTargetRegion(const rvsdg::Output & output) const;

  void
  hoistNodes(rvsdg::Region & region) const;

  void
  hoistNode(rvsdg::SimpleNode & mhbNode) const;

  static rvsdg::Input &
  getUserFromTargetRegion(rvsdg::Input & input, rvsdg::Region & targetRegion);

  static std::vector<rvsdg::Input *>
  getUsersFromTargetRegion(rvsdg::Node & node, rvsdg::Region & targetRegion);

  std::unique_ptr<Context> context_{};
  std::unique_ptr<HoistContext> hoistContext_{};
};

}

#endif
