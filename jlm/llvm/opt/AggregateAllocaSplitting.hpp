/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_AGGREGATEALLOCASPLITTING_HPP
#define JLM_LLVM_OPT_AGGREGATEALLOCASPLITTING_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

class AggregateAllocaSplitting final : public rvsdg::Transformation
{
  struct Context;
  class Statistics;
  struct AllocaTraceInfo;

public:
  ~AggregateAllocaSplitting() noexcept override;

  AggregateAllocaSplitting();

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  splitAllocaNodes(rvsdg::RvsdgModule & rvsdgModule);

  std::vector<AllocaTraceInfo>
  findSplitableAllocaNodes(rvsdg::Region & region) const;

  static void
  splitAllocaNode(const AllocaTraceInfo & allocaTraceInfo);

  static bool
  checkGetElementPtrUsers(const rvsdg::SimpleNode & gepNode);

  static std::optional<AllocaTraceInfo>
  isSplitable(rvsdg::SimpleNode & allocaNode);

  static bool
  isSplitableType(const rvsdg::Type & type);

  std::unique_ptr<Context> context_{};
};

}

#endif
