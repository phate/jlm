/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATISTICS_HPP
#define JLM_UTIL_STATISTICS_HPP

#include <jlm/util/file.hpp>
#include <jlm/util/HashSet.hpp>

#include <memory>

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

/**
 * Determines the settings of a StatisticsCollector.
 */
class StatisticsCollectorSettings final {
public:
  StatisticsCollectorSettings()
    : FilePath_("/tmp/jlm-stats.log")
  {}

  StatisticsCollectorSettings(
    jlm::filepath filePath,
    HashSet<Statistics::Id> demandedStatistics)
    : FilePath_(std::move(filePath))
    , DemandedStatistics_(std::move(demandedStatistics))
  {}

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

  const jlm::filepath &
  GetFilePath() const noexcept
  {
    return FilePath_;
  }

  void
  SetFilePath(jlm::filepath filePath)
  {
    FilePath_ = std::move(filePath);
  }

  void
  SetDemandedStatistics(HashSet<Statistics::Id> demandedStatistics)
  {
    DemandedStatistics_ = std::move(demandedStatistics);
  }

private:
  jlm::filepath FilePath_;
  HashSet<Statistics::Id> DemandedStatistics_;
};

/**
 * Collects and prints statistics.
 */
class StatisticsCollector final {
  class StatisticsIterator final : public std::iterator<std::forward_iterator_tag,
    const Statistics*, ptrdiff_t> {

    friend StatisticsCollector;

    explicit
    StatisticsIterator(const std::vector<std::unique_ptr<Statistics>>::const_iterator & it)
      : it_(it)
    {}

  public:
    [[nodiscard]] const Statistics *
    GetStatistics() const noexcept
    {
      return it_->get();
    }

    const Statistics &
    operator*() const
    {
      JLM_ASSERT(GetStatistics() != nullptr);
      return *GetStatistics();
    }

    const Statistics *
    operator->() const
    {
      return GetStatistics();
    }

    StatisticsIterator &
    operator++()
    {
      ++it_;
      return *this;
    }

    StatisticsIterator
    operator++(int)
    {
      StatisticsIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const StatisticsIterator & other) const
    {
      return it_ == other.it_;
    }

    bool
    operator!=(const StatisticsIterator & other) const
    {
      return !operator==(other);
    }

  private:
    std::vector<std::unique_ptr<Statistics>>::const_iterator it_;
  };

public:
  using StatisticsRange = iterator_range<StatisticsIterator>;

  StatisticsCollector()
  {}

  explicit
  StatisticsCollector(StatisticsCollectorSettings settings)
    : Settings_(std::move(settings))
  {}

  const StatisticsCollectorSettings &
  GetSettings() const noexcept
  {
    return Settings_;
  }

  StatisticsRange
  CollectedStatistics() const noexcept
  {
    return {StatisticsIterator(CollectedStatistics_.begin()), StatisticsIterator(CollectedStatistics_.end())};
  }

  [[nodiscard]] size_t
  NumCollectedStatistics() const noexcept
  {
    return CollectedStatistics_.size();
  }

  /**
   * Add \p statistics to collected statistics. A statistics is only added if it is demanded.
   *
   * @param statistics The statistics to collect.
   *
   * @see StatisticsCollectorSettings::IsDemanded()
   */
  void
  CollectDemandedStatistics(std::unique_ptr<Statistics> statistics)
  {
    if (GetSettings().IsDemanded(statistics->GetId()))
      CollectedStatistics_.emplace_back(std::move(statistics));
  }

  /** \brief Print collected statistics to file.
   *
   * @see StatisticsCollectorSettings::GetFilePath()
   */
  void
  PrintStatistics() const;

private:
  StatisticsCollectorSettings Settings_;
  std::vector<std::unique_ptr<Statistics>> CollectedStatistics_;
};

}

#endif
