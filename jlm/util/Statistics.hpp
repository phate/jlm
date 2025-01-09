/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATISTICS_HPP
#define JLM_UTIL_STATISTICS_HPP

#include <jlm/util/file.hpp>
#include <jlm/util/HashSet.hpp>
#include <jlm/util/time.hpp>

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace jlm::util
{

/**
 * \brief Statistics Interface
 */
class Statistics
{
public:
  enum class Id
  {
    FirstEnumValue, // must always be the first enum value, used for iteration

    Aggregation,
    AgnosticMemoryNodeProvisioning,
    AndersenAnalysis,
    Annotation,
    CommonNodeElimination,
    ControlFlowRecovery,
    DataNodeToDelta,
    DeadNodeElimination,
    FunctionInlining,
    InvariantValueRedirection,
    JlmToRvsdgConversion,
    LoopUnrolling,
    MemoryStateEncoder,
    PullNodes,
    PushNodes,
    ReduceNodes,
    RegionAwareMemoryNodeProvisioning,
    RvsdgConstruction,
    RvsdgDestruction,
    RvsdgOptimization,
    RvsdgTreePrinter,
    SteensgaardAnalysis,
    ThetaGammaInversion,
    TopDownMemoryNodeEliminator,

    LastEnumValue // must always be the last enum value, used for iteration
  };

  using Measurement = std::variant<std::string, int64_t, uint64_t, double>;
  // Lists are used instead of vectors to give stable references to members
  using MeasurementList = std::list<std::pair<std::string, Measurement>>;
  using TimerList = std::list<std::pair<std::string, util::timer>>;

  virtual ~Statistics();

  explicit Statistics(const Id & statisticsId)
      : StatisticsId_(statisticsId)
  {}

  Statistics(const Statistics::Id & statisticsId, util::filepath sourceFile)
      : StatisticsId_(statisticsId),
        SourceFile_(std::move(sourceFile))
  {}

  [[nodiscard]] Statistics::Id
  GetId() const noexcept
  {
    return StatisticsId_;
  }

  /**
   * @return a string identifying the type of this Statistics instance
   */
  [[nodiscard]] std::string_view
  GetName() const;

  /**
   * @return the source file that was worked on while capturing these statistics
   */
  [[nodiscard]] std::optional<filepath>
  GetSourceFile() const;

  /**
   * Converts the Statistics instance to a string containing all information it has.
   * Requires all timers to be stopped.
   *
   * @param fieldSeparator Separation character used between different measurements and/or timers.
   * @param nameValueSeparator Separation character used between the name and value of a measurement
   * or timer.
   *
   * @return a full serialized description of the Statistic instance
   */
  [[nodiscard]] std::string
  Serialize(char fieldSeparator, char nameValueSeparator) const;

  /**
   * Checks if a measurement with the given \p name exists.
   * @return true if the measurement exists, false otherwise.
   */
  [[nodiscard]] bool
  HasMeasurement(const std::string & name) const noexcept;

  /**
   * Gets the measurement with the given \p name, it must exist.
   * @return a reference to the measurement.
   */
  [[nodiscard]] const Measurement &
  GetMeasurement(const std::string & name) const;

  /**
   * Gets the value of the measurement with the given \p name.
   * Requires the measurement to exist and have the given type \tparam T.
   * @return the measurement's value.
   */
  template<typename T>
  [[nodiscard]] const T &
  GetMeasurementValue(const std::string & name) const
  {
    const auto & measurement = GetMeasurement(name);
    return std::get<T>(measurement);
  }

  /**
   * Retrieves the full list of measurements
   */
  [[nodiscard]] util::iterator_range<MeasurementList::const_iterator>
  GetMeasurements() const;

  /**
   * Checks if a timer with the given \p name exists.
   * @return true if the timer exists, false otherwise.
   */
  [[nodiscard]] bool
  HasTimer(const std::string & name) const noexcept;

  /**
   * Retrieves the measured time passed on the timer with the given \p name.
   * Requires the timer to exist, and not currently be running.
   * @return the timer's elapsed time in nanoseconds
   */
  [[nodiscard]] size_t
  GetTimerElapsedNanoseconds(const std::string & name) const
  {
    return GetTimer(name).ns();
  }

  /**
   * Retrieves the full list of timers
   */
  [[nodiscard]] util::iterator_range<TimerList::const_iterator>
  GetTimers() const;

protected:
  /**
   * Adds a measurement, identified by \p name, with the given value.
   * Requires that the measurement doesn't already exist.
   * @tparam T the type of the measurement, must be one of: std::string, int64_t, uint64_t, double
   */
  template<typename T>
  void
  AddMeasurement(std::string name, T value)
  {
    JLM_ASSERT(!HasMeasurement(name));
    Measurements_.emplace_back(std::make_pair(std::move(name), std::move(value)));
  }

  /**
   * Creates a new timer with the given \p name.
   * Requires that the timer does not already exist.
   * @return a reference to the timer
   */
  util::timer &
  AddTimer(std::string name);

  /**
   * Retrieves the timer with the given \p name.
   * Requires that the timer already exists.
   * @return a reference to the timer
   */
  [[nodiscard]] util::timer &
  GetTimer(const std::string & name);

  [[nodiscard]] const util::timer &
  GetTimer(const std::string & name) const;

  /**
   * Commonly used measurement and timer labels throughout statistics gathering.
   */
  struct Label
  {
    static inline const char * FunctionNameLabel_ = "Function";

    static inline const char * NumCfgNodes = "#CfgNodes";

    static inline const char * NumThreeAddressCodes = "#ThreeAddressCodes";

    static inline const char * NumRvsdgNodes = "#RvsdgNodes";
    static inline const char * NumRvsdgNodesBefore = "#RvsdgNodesBefore";
    static inline const char * NumRvsdgNodesAfter = "#RvsdgNodesAfter";

    static inline const char * NumRvsdgInputsBefore = "#RvsdgInputsBefore";
    static inline const char * NumRvsdgInputsAfter = "#RvsdgInputsAfter";

    inline static const char * NumPointsToGraphNodes = "#PointsToGraphNodes";
    inline static const char * NumPointsToGraphAllocaNodes = "#PointsToGraphAllocaNodes";
    inline static const char * NumPointsToGraphDeltaNodes = "#PointsToGraphDeltaNodes";
    inline static const char * NumPointsToGraphImportNodes = "#PointsToGraphImportNodes";
    inline static const char * NumPointsToGraphLambdaNodes = "#PointsToGraphLambdaNodes";
    inline static const char * NumPointsToGraphMallocNodes = "#PointsToGraphMallocNodes";
    inline static const char * NumPointsToGraphMemoryNodes = "#PointsToGraphMemoryNodes";
    inline static const char * NumPointsToGraphRegisterNodes = "#PointsToGraphRegisterNodes";
    inline static const char * NumPointsToGraphEscapedNodes = "#PointsToGraphEscapedNodes";
    inline static const char * NumPointsToGraphExternalMemorySources =
        "#PointsToGraphExternalMemorySources";
    inline static const char * NumPointsToGraphUnknownMemorySources =
        "#PointsToGraphUnknownMemorySources";

    inline static const char * NumPointsToGraphEdges = "#PointsToGraphEdges";
    inline static const char * NumPointsToGraphPointsToRelations =
        "#PointsToGraphPointsToRelations";

    static inline const char * Timer = "Time";
  };

private:
  Statistics::Id StatisticsId_;
  std::optional<filepath> SourceFile_;

  MeasurementList Measurements_;
  TimerList Timers_;
};

/**
 * Determines the settings of a StatisticsCollector.
 */
class StatisticsCollectorSettings final
{
public:
  StatisticsCollectorSettings()
      : FilePath_("")
  {}

  explicit StatisticsCollectorSettings(HashSet<Statistics::Id> demandedStatistics)
      : FilePath_(""),
        DemandedStatistics_(std::move(demandedStatistics))
  {}

  StatisticsCollectorSettings(filepath filePath, HashSet<Statistics::Id> demandedStatistics)
      : FilePath_(std::move(filePath)),
        DemandedStatistics_(std::move(demandedStatistics))
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

  const filepath &
  GetFilePath() const noexcept
  {
    return FilePath_;
  }

  void
  SetFilePath(filepath filePath)
  {
    FilePath_ = std::move(filePath);
  }

  void
  SetDemandedStatistics(HashSet<Statistics::Id> demandedStatistics)
  {
    DemandedStatistics_ = std::move(demandedStatistics);
  }

  [[nodiscard]] size_t
  NumDemandedStatistics() const noexcept
  {
    return DemandedStatistics_.Size();
  }

  [[nodiscard]] const HashSet<Statistics::Id> &
  GetDemandedStatistics() const noexcept
  {
    return DemandedStatistics_;
  }

  static filepath
  CreateUniqueStatisticsFile(const filepath & directory, const filepath & inputFile)
  {
    return filepath::CreateUniqueFileName(directory, inputFile.base() + "-", "-statistics.log");
  }

private:
  filepath FilePath_;
  HashSet<Statistics::Id> DemandedStatistics_;
};

/**
 * Collects and prints statistics.
 */
class StatisticsCollector final
{
  class StatisticsIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Statistics *;
    using difference_type = std::ptrdiff_t;
    using pointer = const Statistics **;
    using reference = const Statistics *&;

  private:
    friend StatisticsCollector;

    explicit StatisticsIterator(const std::vector<std::unique_ptr<Statistics>>::const_iterator & it)
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

  explicit StatisticsCollector(StatisticsCollectorSettings settings)
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
    return { StatisticsIterator(CollectedStatistics_.begin()),
             StatisticsIterator(CollectedStatistics_.end()) };
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
