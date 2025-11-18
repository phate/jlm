/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATISTICS_HPP
#define JLM_UTIL_STATISTICS_HPP

#include <jlm/util/file.hpp>
#include <jlm/util/HashSet.hpp>
#include <jlm/util/strfmt.hpp>
#include <jlm/util/time.hpp>

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

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
    AgnosticModRefSummarizer,
    AliasAnalysisPrecisionEvaluation,
    AndersenAnalysis,
    Annotation,
    CommonNodeElimination,
    ControlFlowRecovery,
    DataNodeToDelta,
    DeadNodeElimination,
    FunctionInlining,
    IfConversion,
    InvariantValueRedirection,
    JlmToRvsdgConversion,
    LoopUnrolling,
    LoopUnswitching,
    MemoryStateEncoder,
    PullNodes,
    PushNodes,
    ReduceNodes,
    RegionAwareModRefSummarizer,
    RvsdgConstruction,
    RvsdgDestruction,
    RvsdgOptimization,
    RvsdgTreePrinter,
    ScalarEvolution,

    LastEnumValue // must always be the last enum value, used for iteration
  };

  using Measurement = std::variant<std::string, int64_t, uint64_t, double>;
  // Lists are used instead of vectors to give stable references to members
  using MeasurementList = std::list<std::pair<std::string, Measurement>>;
  using TimerList = std::list<std::pair<std::string, util::Timer>>;

  virtual ~Statistics();

  Statistics(const Statistics::Id & statisticsId, util::FilePath sourceFile)
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
  [[nodiscard]] const util::FilePath &
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
  [[nodiscard]] IteratorRange<MeasurementList::const_iterator>
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
  [[nodiscard]] IteratorRange<TimerList::const_iterator>
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
  util::Timer &
  AddTimer(std::string name);

  /**
   * Retrieves the timer with the given \p name.
   * Requires that the timer already exists.
   * @return a reference to the timer
   */
  [[nodiscard]] util::Timer &
  GetTimer(const std::string & name);

  [[nodiscard]] const util::Timer &
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
    inline static const char * NumPointsToGraphExternallyAvailableNodes =
        "#PointsToGraphExternallyAvailableNodes";
    inline static const char * NumPointsToGraphNodesTargetsAllExternallyAvailable =
        "#PointsToGraphNodesTargetsAllExternallyAvailable";

    inline static const char * NumPointsToGraphExplicitEdges = "#PointsToGraphExplicitEdges";
    inline static const char * NumPointsToGraphEdges = "#PointsToGraphEdges";

    static inline const char * Timer = "Time";
  };

private:
  Statistics::Id StatisticsId_;
  util::FilePath SourceFile_;

  MeasurementList Measurements_;
  TimerList Timers_;
};

/**
 * Determines the settings of a StatisticsCollector.
 */
class StatisticsCollectorSettings final
{
public:
  /**
   * Creates settings for a StatisticsCollector that does not demand any statistics.
   * Uses the current working directory for any output files.
   */
  StatisticsCollectorSettings()
      : Directory_("."),
        ModuleName_("")
  {}

  /**
   * Creates settings for a StatisticsCollector that demands the given statistics be collected.
   * Uses the current working directory for any output files.
   * @param demandedStatistics a hash set of statistics ids to collect
   */
  explicit StatisticsCollectorSettings(HashSet<Statistics::Id> demandedStatistics)
      : DemandedStatistics_(std::move(demandedStatistics)),
        Directory_("."),
        ModuleName_("")
  {}

  /**
   * Creates settings for a StatisticsCollector that demands the given statistics be collected,
   * and specifies the output directory to place statistics and debug output.
   *
   * The directory does not need to exist, but its parent directory must exist.
   * The output directory is only created if output files are created.
   *
   * Output files get the given module name as a prefix,
   * in addition to a unique random string generated per StatisticsCollectorSettings.
   *
   * @param demandedStatistics a hash set of statistics ids to collect
   * @param directory the directory where statistics and debug output should be placed
   * @param moduleName a prefix given to all files created in the given directory
   */
  StatisticsCollectorSettings(
      HashSet<Statistics::Id> demandedStatistics,
      std::optional<FilePath> directory,
      std::string moduleName)
      : DemandedStatistics_(std::move(demandedStatistics)),
        Directory_(std::move(directory)),
        ModuleName_(std::move(moduleName))
  {}

  /**
   * Sets the hash set containing the Ids of statistics that should be collected
   * @param demandedStatistics the new hash set of demanded statistics
   */
  void
  SetDemandedStatistics(HashSet<Statistics::Id> demandedStatistics)
  {
    DemandedStatistics_ = std::move(demandedStatistics);
  }

  /**
   * @return the number of demanded statistics
   */
  [[nodiscard]] size_t
  NumDemandedStatistics() const noexcept
  {
    return DemandedStatistics_.Size();
  }

  /**
   * @return a hash set containing the Ids of statistics that should be collected
   */
  [[nodiscard]] const HashSet<Statistics::Id> &
  GetDemandedStatistics() const noexcept
  {
    return DemandedStatistics_;
  }

  /** \brief Checks if a statistics is demanded.
   *
   * @param id The Id of the statistics.
   * @return True if a statistics is demanded, otherwise false.
   */
  [[nodiscard]] bool
  isDemanded(Statistics::Id id) const noexcept
  {
    return DemandedStatistics_.Contains(id);
  }

  /**
   * @return true if a directory for outputting statistics and debug output files is specified,
   * otherwise false
   */
  [[nodiscard]] bool
  HasOutputDirectory() const noexcept
  {
    return Directory_.has_value();
  }

  /**
   * @return the directory used for outputting statistics and debug output files.
   */
  [[nodiscard]] const FilePath &
  GetOutputDirectory() const noexcept
  {
    JLM_ASSERT(Directory_.has_value());
    return Directory_.value();
  }

  /**
   * @return the directory used for outputting statistics and debug output files.
   *
   * \note If no output directory path is given, an assertion failure occurs. If the directory
   * does not exist yet, it is created.
   */
  [[nodiscard]] const FilePath &
  GetOrCreateOutputDirectory() const noexcept
  {
    JLM_ASSERT(Directory_.has_value());
    Directory_->CreateDirectory();
    return Directory_.value();
  }

  /**
   * Sets the directory used for outputting statistics and debug output files.
   * It does not need to exist, but its parent directory must always exist.
   * @param directory the directory to place statistics and debug output files in
   */
  void
  SetOutputDirectory(FilePath directory)
  {
    Directory_ = std::move(directory);
  }

  /**
   * @return the module name used as a prefix for all output files
   */
  [[nodiscard]] const std::string &
  GetModuleName() const noexcept
  {
    return ModuleName_;
  }

  /**
   * Sets the module name used as a prefix for all output files
   */
  void
  SetModuleName(std::string moduleName)
  {
    ModuleName_ = std::move(moduleName);
  }

  /**
   * @return the unique string included in the name of all created files
   */
  [[nodiscard]] const std::string &
  GetUniqueString() const noexcept
  {
    return UniqueString_;
  }

  /**
   * Sets the unique string to be included in the name of all created files
   * @param uniqueString the new unique string
   */
  void
  SetUniqueString(std::string uniqueString)
  {
    UniqueString_ = std::move(uniqueString);
  }

private:
  HashSet<Statistics::Id> DemandedStatistics_;
  std::optional<FilePath> Directory_;
  std::string ModuleName_;
  std::string UniqueString_ = CreateRandomAlphanumericString(6);
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
  using StatisticsRange = IteratorRange<StatisticsIterator>;

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
   * Checks statistics with the given id are demanded.
   *
   * @param id The statistics id to check.
   *
   * @return True if \p statistics with the given id are demanded, otherwise false.
   */
  [[nodiscard]] bool
  IsDemanded(Statistics::Id id) const noexcept
  {
    return GetSettings().isDemanded(id);
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
    return IsDemanded(statistics.GetId());
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
    if (IsDemanded(*statistics))
      CollectedStatistics_.emplace_back(std::move(statistics));
  }

  /**
   * \brief Print collected statistics to file.
   * If no statistics have been collected, this is a no-op.
   * @throws jlm::util::error if there are statistics to print, but no output directory in settings
   */
  void
  PrintStatistics();

  /**
   * Creates a unique file name in the statistics and debug output directory.
   * If a module name is specified in the settings, it is included in the file name.
   * If a unique string is specified in the settings, it is also included.
   * Lastly the given \p fileNameSuffix is used as the suffix for the file,
   * including an optional counter if \p includeCount is true.
   *
   * If the specified output directory does not exist, it is created.
   *
   * @param fileNameSuffix the output file name suffix, e.g., "statistics.log"
   * @param includeCount include a counter per suffix, to avoid naming collisions
   * @return a file representing the new output file
   * @throws jlm::util::error in any of the following cases:
   *  - no output directory has been specified in the StatisticsCollectorSettings
   *  - any issues occur with creating the output directory
   *  - the resulting output file already exists
   */
  [[nodiscard]] File
  createOutputFile(std::string fileNameSuffix, bool includeCount = false);

private:
  StatisticsCollectorSettings Settings_;
  std::vector<std::unique_ptr<Statistics>> CollectedStatistics_;

  // Counter used to give unique file names to output files that share suffix
  std::unordered_map<std::string, size_t> OutputFileCounter_;
};

}

#endif
