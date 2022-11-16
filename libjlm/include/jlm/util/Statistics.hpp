/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATISTICS_HPP
#define JLM_UTIL_STATISTICS_HPP

#include <jlm/util/file.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm {

/**
 * \brief Statistics Interface
 */
class Statistics {
public:
  enum class Id {
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

  virtual
  ~Statistics();

  explicit
  Statistics(const Statistics::Id & statisticsId)
    : StatisticsId_(statisticsId)
  {}

  [[nodiscard]] Statistics::Id
  GetId() const noexcept
  {
    return StatisticsId_;
  }

  [[nodiscard]] virtual std::string
  ToString() const = 0;

private:
  Statistics::Id StatisticsId_;
};

class StatisticsDescriptor final {
public:
  StatisticsDescriptor()
    : StatisticsDescriptor(
    std::string("/tmp/jlm-stats.log"),
    {})
  {}

  explicit
  StatisticsDescriptor(HashSet<Statistics::Id> demandedStatistics)
    : StatisticsDescriptor(
    std::string("tmp/jlm-stats.log"),
    std::move(demandedStatistics))
  {}

  StatisticsDescriptor(
    const jlm::filepath & path,
    HashSet<Statistics::Id> demandedStatistics)
    : File_(path)
    , DemandedStatistics_(std::move(demandedStatistics))
  {
    File_.open("a");
  }


  const jlm::filepath &
  GetFilePath() const noexcept
  {
    return File_.path();
  }

  void
  SetFilePath(const jlm::filepath & path) noexcept
  {
    File_ = jlm::file(path);
    File_.open("a");
  }

  void
  SetDemandedStatistics(HashSet<Statistics::Id> printStatistics)
  {
    DemandedStatistics_ = std::move(printStatistics);
  }

  /** \brief Prints statistics to file.
   *
   * Prints \p statistics to the statistics file iff the \p statistics'
   * StatisticsId was set with SetDemandedStatistics().
   *
   * @param statistics The statistics that is printed.
   *
   * @see SetDemandedStatistics()
   * @see IsDemanded()
   */
  void
  PrintStatistics(const Statistics & statistics) const noexcept;

  /** \brief Checks if a statistics is demanded.
   *
   * @param id The Id of the statistics.
   * @return True if a statistics is demanded, otherwise false.
   */
  bool
  IsDemanded(Statistics::Id id) const
  {
    return DemandedStatistics_.Contains(id);
  }

private:
  jlm::file File_;
  HashSet<Statistics::Id> DemandedStatistics_;
};

}

#endif
