/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATS_HPP
#define JLM_UTIL_STATS_HPP

#include <jlm/util/file.hpp>

#include <unordered_set>

namespace jlm {

class stat {
public:
	virtual
	~stat();

	virtual std::string
	to_str() const = 0;
};

class StatisticsDescriptor final {
public:
  enum class StatisticsId {
    Aggregation,
    Annotation,
    CommonNodeElimination,
    ControlFlowRecovery,
    DeadNodeElimination,
    FunctionInlining,
    InvariantValueReduction,
    JlmToRvsdgConversion,
    LoopUnrolling,
    PullNodes,
    PushNodes,
    ReduceNodes,
    RvsdgConstruction,
    RvsdgDestruction,
    RvsdgOptimization,
    SteensgaardAnalysis,
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

	void
	print_stat(const stat & s) const noexcept
	{
		fprintf(file_.fd(), "%s\n", s.to_str().c_str());
	}

  bool
  IsPrintable(StatisticsId id) const
  {
    return printStatistics_.find(id) != printStatistics_.end();
  }

private:
	jlm::file file_;
  std::unordered_set<StatisticsId> printStatistics_;
};

}

#endif
