/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_OPTIMIZATIONSEQUENCE_HPP
#define JLM_LLVM_OPT_OPTIMIZATIONSEQUENCE_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm::llvm
{

/**
 * Sequentially applies a list of optimizations to an Rvsdg.
 */
class OptimizationSequence final : public optimization
{
public:
  class Statistics;

  ~OptimizationSequence() noexcept override;

  explicit OptimizationSequence(std::vector<std::unique_ptr<optimization>> optimizations)
      : Optimizations_(std::move(optimizations))
  {}

  void
  run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(
      RvsdgModule & rvsdgModule,
      util::StatisticsCollector & statisticsCollector,
      std::vector<std::unique_ptr<optimization>> optimizations)
  {
    OptimizationSequence sequentialApplication(std::move(optimizations));
    sequentialApplication.run(rvsdgModule, statisticsCollector);
  }

private:
  std::vector<std::unique_ptr<optimization>> Optimizations_;
};

}

#endif // JLM_LLVM_OPT_OPTIMIZATIONSEQUENCE_HPP
