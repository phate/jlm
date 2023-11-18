/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_OPTIMIZATION_HPP
#define JLM_LLVM_OPT_OPTIMIZATION_HPP

#include <vector>

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

class RvsdgModule;

/**
 * \brief Optimization pass interface
 */
class optimization
{
public:
  virtual ~optimization();

  /**
   * \brief Perform optimization
   *
   * This method is expected to be called multiple times. An
   * implementation is required to reset the objects' internal state
   * to ensure correct behavior after every invocation.
   *
   * \param module RVSDG module the optimization is performed on.
   * \param statisticsCollector Statistics collector for collecting optimization statistics.
   */
  virtual void
  run(RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector) = 0;
};

}

#endif
