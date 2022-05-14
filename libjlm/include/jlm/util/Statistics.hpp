/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATISTICS_HPP
#define JLM_UTIL_STATISTICS_HPP

#include <jlm/util/file.hpp>

#include <unordered_set>

namespace jlm {

class Statistics;

class StatisticsDescriptor final {
public:
  enum class StatisticsId {
    Aggregation,
    Annotation,
    BasicEncoderEncoding,
    CommonNodeElimination,
    ControlFlowRecovery,
    DataNodeToDelta,
    DeadNodeElimination,
    FunctionInlining,
    InvariantValueRedirection,
    JlmToRvsdgConversion,
    LoopUnrolling,
    PullNodes,
    PushNodes,
    ReduceNodes,
    RvsdgConstruction,
    RvsdgDestruction,
    RvsdgOptimization,
    SteensgaardAnalysis,
    SteensgaardPointsToGraphConstruction,
    ThetaGammaInversion
  };

	StatisticsDescriptor()
	: StatisticsDescriptor(
    std::string("/tmp/jlm-stats.log"),
    {})
	{}

  explicit
  StatisticsDescriptor(std::unordered_set<StatisticsId> printStatistics)
  : StatisticsDescriptor(
    std::string("tmp/jlm-stats.log"),
    std::move(printStatistics))
  {}

	StatisticsDescriptor(
    const jlm::filepath & path,
    std::unordered_set<StatisticsId> printStatistics)
	: file_(path)
  , printStatistics_(std::move(printStatistics))
	{
		file_.open("a");
	}


	const jlm::filepath &
	filepath() const noexcept
	{
		return file_.path();
	}

	void
	set_file(const jlm::filepath & path) noexcept
	{
		file_ = jlm::file(path);
		file_.open("a");
	}

  void
  SetPrintStatisticsIds(std::unordered_set<StatisticsId> printStatistics)
  {
    printStatistics_ = std::move(printStatistics);
  }

  /** \brief Prints statistics to file.
   *
   * Prints \p statistics to the statistics file iff the \p statistics'
   * StatisticsId was set with SetPrintStatisticsIds().
   *
   * @param statistics The statistics that is printed.
   *
   * @see SetPrintStatisticsIds()
   * @see IsPrintable()
   */
	void
	PrintStatistics(const Statistics & statistics) const noexcept;

  bool
  IsPrintable(StatisticsId id) const
  {
    return printStatistics_.find(id) != printStatistics_.end();
  }

private:
	jlm::file file_;
  std::unordered_set<StatisticsId> printStatistics_;
};

class Statistics {
public:
  virtual
  ~Statistics();

  explicit
  Statistics(const StatisticsDescriptor::StatisticsId & statisticsId)
  : StatisticsId_(statisticsId)
  {}

  StatisticsDescriptor::StatisticsId
  GetStatisticsId() const noexcept
  {
    return StatisticsId_;
  }

  virtual std::string
  ToString() const = 0;

private:
  StatisticsDescriptor::StatisticsId StatisticsId_;
};

}

#endif
