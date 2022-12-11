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
    MemoryNodeProvisioning,
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
  class StatisticsIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Statistics*;
    using difference_type = std::ptrdiff_t;
    using pointer = const Statistics**;
    using reference = const Statistics*&;

  private:
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
   * Checks if the pass statistics is demanded.
   *
   * @param statistics The statistics to check whether it is demanded.
   *
   * @return True if \p statistics is demanded, otherwise false.
   */
  [[nodiscard]] bool
  IsDemanded(const Statistics & statistics) const noexcept
  {
    return GetSettings().IsDemanded(statistics.GetId());
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
