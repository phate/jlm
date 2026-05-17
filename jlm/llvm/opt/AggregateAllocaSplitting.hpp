/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_AGGREGATEALLOCASPLITTING_HPP
#define JLM_LLVM_OPT_AGGREGATEALLOCASPLITTING_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{

/**
 * \brief Aggregate Alloca Splitting Transformation
 *
 * Aggregate Alloca Splitting splits up \ref AllocaOperation nodes with aggregate types, i.e.,
 * struct and array types, into multiple \ref AllocaOperation nodes with simple types, i.e., integer
 * or floating-point types. The intention is to simplify the processing of these \ref
 * AllocaOperation nodes by passes such as \ref StoreValueForwarding.
 */
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
