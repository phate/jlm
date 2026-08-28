/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_IOBARRIERREDIRECTION_HPP
#define JLM_LLVM_OPT_IOBARRIERREDIRECTION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

class IOBarrierRedirection final : public rvsdg::Transformation
{
  class Statistics;

public:
  ~IOBarrierRedirection() override;

  IOBarrierRedirection();

  IOBarrierRedirection(const IOBarrierRedirection &) = delete;

  IOBarrierRedirection(IOBarrierRedirection &&) = delete;

  IOBarrierRedirection &
  operator=(const IOBarrierRedirection &) = delete;

  IOBarrierRedirection &
  operator=(IOBarrierRedirection &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  redirectInRegion(rvsdg::Region & region);

  static void
  redirectIOBarrierNode(rvsdg::SimpleNode & ioBarrierNode);
};

}

#endif
