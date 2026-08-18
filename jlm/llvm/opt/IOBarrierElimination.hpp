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

private:
  void
  markRegion(rvsdg::Region & region);

  void
  markNode(const rvsdg::Node & node);

  void
  sweepRegion(rvsdg::Region & region);

  static void
  removeIOBarrierNode(rvsdg::Node & node);

  std::unique_ptr<Context> context_{};
};

}

#endif
