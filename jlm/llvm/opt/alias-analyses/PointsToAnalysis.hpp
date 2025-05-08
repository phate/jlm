/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOANALYSIS_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_POINTSTOANALYSIS_HPP

#include <memory>

namespace jlm::rvsdg
{
class RvsdgModule;
}

namespace jlm::util
{
class StatisticsCollector;
}

namespace jlm::llvm
{

namespace aa
{

class PointsToGraph;

/**
 * \brief Points-to Analysis Interface
 */
class PointsToAnalysis
{
public:
  virtual ~PointsToAnalysis() = default;

  /**
   * \brief Analyze RVSDG module
   *
   * \param module RVSDG module the analysis is performed on.
   * \param statisticsCollector Statistics collector for collecting analysis statistics.
   *
   * \return A PointsTo graph.
   */
  virtual std::unique_ptr<PointsToGraph>
  Analyze(const rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) = 0;
};

}
}

#endif
