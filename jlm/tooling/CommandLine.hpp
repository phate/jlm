/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMANDLINE_HPP
#define JLM_TOOLING_COMMANDLINE_HPP

#include <jlm/llvm/opt/optimization.hpp>
#include <jlm/util/file.hpp>
#include <jlm/util/Statistics.hpp>

#include <vector>

namespace jlm::tooling
{

/**
 * Interface for the command line options of a Jlm command line tool.
 */
class CommandLineOptions
{
public:
  virtual ~CommandLineOptions();

  CommandLineOptions() = default;

  /**
   * Resets the state of the instance.
   */
  virtual void
  Reset() noexcept = 0;
};

class optimization;

/**
 * Command line options for the \a jlm-opt command line tool.
 */
class JlmOptCommandLineOptions final : public CommandLineOptions
{
public:
  enum class OutputFormat
  {
    Llvm,
    Xml
  };

  enum class OptimizationId
  {
    FirstEnumValue, // must always be the first enum value, used for iteration

    AASteensgaardAgnostic,
    AASteensgaardRegionAware,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::CommonNodeElimination instead.
     */
    cne,
    CommonNodeElimination,
    DeadNodeElimination,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::DeadNodeElimination instead.
     */
    dne,
    FunctionInlining,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::FunctionInlining instead.
     */
    iln,
    InvariantValueRedirection,
    LoopUnrolling,
    NodePullIn,
    NodePushOut,
    NodeReduction,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::NodePushOut instead.
     */
    psh,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::NodeReduction instead.
     */
    red,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::ThetaGammaInversion instead.
     */
    ivt,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::LoopUnrolling instead.
     */
    url,

    /**
     * \deprecated This flag is going to be removed in the future. Use \ref
     * OptimizationId::NodePullIn instead.
     */
    pll,
    ThetaGammaInversion,

    LastEnumValue // must always be the last enum value, used for iteration
  };

  JlmOptCommandLineOptions(
      util::filepath inputFile,
      util::filepath outputFile,
      OutputFormat outputFormat,
      util::StatisticsCollectorSettings statisticsCollectorSettings,
      std::vector<OptimizationId> optimizations)
      : InputFile_(std::move(inputFile)),
        OutputFile_(std::move(outputFile)),
        OutputFormat_(outputFormat),
        StatisticsCollectorSettings_(std::move(statisticsCollectorSettings)),
        OptimizationIds_(std::move(optimizations))
  {}

  void
  Reset() noexcept override;

  [[nodiscard]] const util::filepath &
  GetInputFile() const noexcept
  {
    return InputFile_;
  }

  [[nodiscard]] const util::filepath &
  GetOutputFile() const noexcept
  {
    return OutputFile_;
  }

  [[nodiscard]] OutputFormat
  GetOutputFormat() const noexcept
  {
    return OutputFormat_;
  }

  [[nodiscard]] const util::StatisticsCollectorSettings &
  GetStatisticsCollectorSettings() const noexcept
  {
    return StatisticsCollectorSettings_;
  }

  [[nodiscard]] const std::vector<OptimizationId> &
  GetOptimizationIds() const noexcept
  {
    return OptimizationIds_;
  }

  [[nodiscard]] std::vector<llvm::optimization *>
  GetOptimizations() const noexcept;

  static OptimizationId
  FromCommandLineArgumentToOptimizationId(const std::string & commandLineArgument);

  static util::Statistics::Id
  FromCommandLineArgumentToStatisticsId(const std::string & commandLineArgument);

  static const char *
  ToCommandLineArgument(OptimizationId optimizationId);

  static const char *
  ToCommandLineArgument(util::Statistics::Id statisticsId);

  static const char *
  ToCommandLineArgument(OutputFormat outputFormat);

  static llvm::optimization *
  GetOptimization(enum OptimizationId optimizationId);

  static std::unique_ptr<JlmOptCommandLineOptions>
  Create(
      util::filepath inputFile,
      util::filepath outputFile,
      OutputFormat outputFormat,
      util::StatisticsCollectorSettings statisticsCollectorSettings,
      std::vector<OptimizationId> optimizations)
  {
    return std::make_unique<JlmOptCommandLineOptions>(
        std::move(inputFile),
        std::move(outputFile),
        outputFormat,
        std::move(statisticsCollectorSettings),
        std::move(optimizations));
  }

private:
  util::filepath InputFile_;
  util::filepath OutputFile_;
  OutputFormat OutputFormat_;
  util::StatisticsCollectorSettings StatisticsCollectorSettings_;
  std::vector<OptimizationId> OptimizationIds_;

  struct OptimizationCommandLineArgument
  {
    inline static const char * AaSteensgaardAgnostic_ = "AASteensgaardAgnostic";
    inline static const char * AaSteensgaardRegionAware_ = "AASteensgaardRegionAware";
    inline static const char * CommonNodeElimination_ = "CommonNodeElimination";
    inline static const char * CommonNodeEliminationDeprecated_ = "cne";
    inline static const char * DeadNodeElimination_ = "DeadNodeElimination";
    inline static const char * DeadNodeEliminationDeprecated_ = "dne";
    inline static const char * FunctionInlining_ = "FunctionInlining";
    inline static const char * FunctionInliningDeprecated_ = "iln";
    inline static const char * InvariantValueRedirection_ = "InvariantValueRedirection";
    inline static const char * NodePullIn_ = "NodePullIn";
    inline static const char * NodePullInDeprecated_ = "pll";
    inline static const char * NodePushOut_ = "NodePushOut";
    inline static const char * NodePushOutDeprecated_ = "psh";
    inline static const char * ThetaGammaInversion_ = "ThetaGammaInversion";
    inline static const char * ThetaGammaInversionDeprecated_ = "ivt";
    inline static const char * LoopUnrolling_ = "LoopUnrolling";
    inline static const char * LoopUnrollingDeprecated_ = "url";
    inline static const char * NodeReduction_ = "NodeReduction";
    inline static const char * NodeReductionDeprecated_ = "red";
  };

  struct StatisticsCommandLineArgument
  {
    inline static const char * Aggregation_ = "print-aggregation-time";
    inline static const char * Annotation_ = "print-annotation-time";
    inline static const char * BasicEncoderEncoding_ = "print-basicencoder-encoding";
    inline static const char * CommonNodeElimination_ = "print-cne-stat";
    inline static const char * ControlFlowRecovery_ = "print-cfr-time";
    inline static const char * DataNodeToDelta_ = "printDataNodeToDelta";
    inline static const char * DeadNodeElimination_ = "print-dne-stat";
    inline static const char * FunctionInlining_ = "print-iln-stat";
    inline static const char * InvariantValueRedirection_ = "printInvariantValueRedirection";
    inline static const char * JlmToRvsdgConversion_ = "print-jlm-rvsdg-conversion";
    inline static const char * LoopUnrolling_ = "print-unroll-stat";
    inline static const char * MemoryNodeProvisioning_ = "print-memory-node-provisioning";
    inline static const char * PullNodes_ = "print-pull-stat";
    inline static const char * PushNodes_ = "print-push-stat";
    inline static const char * ReduceNodes_ = "print-reduction-stat";
    inline static const char * RvsdgConstruction_ = "print-rvsdg-construction";
    inline static const char * RvsdgDestruction_ = "print-rvsdg-destruction";
    inline static const char * RvsdgOptimization_ = "print-rvsdg-optimization";
    inline static const char * SteensgaardAnalysis_ = "print-steensgaard-analysis";
    inline static const char * ThetaGammaInversion_ = "print-ivt-stat";
  };
};

class JlcCommandLineOptions final : public CommandLineOptions
{
public:
  class Compilation;

  enum class OptimizationLevel
  {
    O0,
    O1,
    O2,
    O3,
  };

  enum class LanguageStandard
  {
    None,
    Gnu89,
    Gnu99,
    C89,
    C99,
    C11,
    Cpp98,
    Cpp03,
    Cpp11,
    Cpp14,
  };

  JlcCommandLineOptions()
      : OnlyPrintCommands_(false),
        GenerateDebugInformation_(false),
        Verbose_(false),
        Rdynamic_(false),
        Suppress_(false),
        UsePthreads_(false),
        Md_(false),
        OptimizationLevel_(OptimizationLevel::O0),
        LanguageStandard_(LanguageStandard::None),
        OutputFile_("a.out")
  {}

  static std::string
  ToString(const OptimizationLevel & optimizationLevel);

  static std::string
  ToString(const LanguageStandard & languageStandard);

  void
  Reset() noexcept override;

  bool OnlyPrintCommands_;
  bool GenerateDebugInformation_;
  bool Verbose_;
  bool Rdynamic_;
  bool Suppress_;
  bool UsePthreads_;

  bool Md_;

  OptimizationLevel OptimizationLevel_;
  LanguageStandard LanguageStandard_;

  util::filepath OutputFile_;
  std::vector<std::string> Libraries_;
  std::vector<std::string> MacroDefinitions_;
  std::vector<std::string> LibraryPaths_;
  std::vector<std::string> Warnings_;
  std::vector<std::string> IncludePaths_;
  std::vector<std::string> Flags_;
  std::vector<JlmOptCommandLineOptions::OptimizationId> JlmOptOptimizations_;
  util::HashSet<util::Statistics::Id> JlmOptPassStatistics_;

  std::vector<Compilation> Compilations_;
};

class JlcCommandLineOptions::Compilation
{
public:
  Compilation(
      util::filepath inputFile,
      util::filepath dependencyFile,
      util::filepath outputFile,
      std::string mT,
      bool requiresParsing,
      bool requiresOptimization,
      bool requiresAssembly,
      bool requiresLinking)
      : RequiresLinking_(requiresLinking),
        RequiresParsing_(requiresParsing),
        RequiresOptimization_(requiresOptimization),
        RequiresAssembly_(requiresAssembly),
        InputFile_(std::move(inputFile)),
        OutputFile_(std::move(outputFile)),
        DependencyFile_(std::move(dependencyFile)),
        Mt_(std::move(mT))
  {}

  [[nodiscard]] const util::filepath &
  InputFile() const noexcept
  {
    return InputFile_;
  }

  [[nodiscard]] const util::filepath &
  DependencyFile() const noexcept
  {
    return DependencyFile_;
  }

  [[nodiscard]] const util::filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  [[nodiscard]] const std::string &
  Mt() const noexcept
  {
    return Mt_;
  }

  void
  SetOutputFile(const util::filepath & outputFile)
  {
    OutputFile_ = outputFile;
  }

  [[nodiscard]] bool
  RequiresParsing() const noexcept
  {
    return RequiresParsing_;
  }

  [[nodiscard]] bool
  RequiresOptimization() const noexcept
  {
    return RequiresOptimization_;
  }

  [[nodiscard]] bool
  RequiresAssembly() const noexcept
  {
    return RequiresAssembly_;
  }

  [[nodiscard]] bool
  RequiresLinking() const noexcept
  {
    return RequiresLinking_;
  }

private:
  bool RequiresLinking_;
  bool RequiresParsing_;
  bool RequiresOptimization_;
  bool RequiresAssembly_;
  util::filepath InputFile_;
  util::filepath OutputFile_;
  util::filepath DependencyFile_;
  const std::string Mt_;
};

/**
 * Command line options for the \a jlm-hls command line tool.
 */
class JlmHlsCommandLineOptions final : public CommandLineOptions
{
public:
  enum class OutputFormat
  {
    Firrtl,
    Dot
  };

  JlmHlsCommandLineOptions()
      : InputFile_(""),
        OutputFolder_(""),
        OutputFormat_(OutputFormat::Firrtl),
        ExtractHlsFunction_(false),
        UseCirct_(false)
  {}

  void
  Reset() noexcept override;

  util::filepath InputFile_;
  util::filepath OutputFolder_;
  OutputFormat OutputFormat_;
  std::string HlsFunction_;
  bool ExtractHlsFunction_;
  bool UseCirct_;
};

/**
 * Command line options for the \a jhls command line tool.
 */
class JhlsCommandLineOptions final : public CommandLineOptions
{
public:
  class Compilation;

  enum class OptimizationLevel
  {
    O0,
    O1,
    O2,
    O3
  };

  enum class LanguageStandard
  {
    None,
    Gnu89,
    Gnu99,
    C89,
    C99,
    C11,
    Cpp98,
    Cpp03,
    Cpp11,
    Cpp14
  };

  JhlsCommandLineOptions()
      : OnlyPrintCommands_(false),
        GenerateDebugInformation_(false),
        Verbose_(false),
        Rdynamic_(false),
        Suppress_(false),
        UsePthreads_(false),
        GenerateFirrtl_(false),
        UseCirct_(false),
        Hls_(false),
        Md_(false),
        OptimizationLevel_(OptimizationLevel::O0),
        LanguageStandard_(LanguageStandard::None),
        OutputFile_("a.out")
  {}

  void
  Reset() noexcept override;

  bool OnlyPrintCommands_;
  bool GenerateDebugInformation_;
  bool Verbose_;
  bool Rdynamic_;
  bool Suppress_;
  bool UsePthreads_;
  bool GenerateFirrtl_;
  bool UseCirct_;
  bool Hls_;

  bool Md_;

  OptimizationLevel OptimizationLevel_;
  LanguageStandard LanguageStandard_;
  util::filepath OutputFile_;
  std::vector<std::string> Libraries_;
  std::vector<std::string> MacroDefinitions_;
  std::vector<std::string> LibraryPaths_;
  std::vector<std::string> Warnings_;
  std::vector<std::string> IncludePaths_;
  std::vector<std::string> Flags_;
  std::vector<std::string> JlmHls_;

  std::vector<Compilation> Compilations_;
  std::string HlsFunctionRegex_;
};

class JhlsCommandLineOptions::Compilation
{
public:
  Compilation(
      util::filepath inputFile,
      util::filepath dependencyFile,
      util::filepath outputFile,
      std::string mT,
      bool parse,
      bool optimize,
      bool assemble,
      bool link)
      : RequiresLinking_(link),
        RequiresParsing_(parse),
        RequiresOptimization_(optimize),
        RequiresAssembly_(assemble),
        InputFile_(std::move(inputFile)),
        OutputFile_(std::move(outputFile)),
        DependencyFile_(std::move(dependencyFile)),
        Mt_(std::move(mT))
  {}

  [[nodiscard]] const util::filepath &
  InputFile() const noexcept
  {
    return InputFile_;
  }

  [[nodiscard]] const util::filepath &
  DependencyFile() const noexcept
  {
    return DependencyFile_;
  }

  [[nodiscard]] const util::filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  [[nodiscard]] const std::string &
  Mt() const noexcept
  {
    return Mt_;
  }

  void
  SetOutputFile(const util::filepath & outputFile)
  {
    OutputFile_ = outputFile;
  }

  [[nodiscard]] bool
  RequiresParsing() const noexcept
  {
    return RequiresParsing_;
  }

  [[nodiscard]] bool
  RequiresOptimization() const noexcept
  {
    return RequiresOptimization_;
  }

  [[nodiscard]] bool
  RequiresAssembly() const noexcept
  {
    return RequiresAssembly_;
  }

  [[nodiscard]] bool
  RequiresLinking() const noexcept
  {
    return RequiresLinking_;
  }

private:
  bool RequiresLinking_;
  bool RequiresParsing_;
  bool RequiresOptimization_;
  bool RequiresAssembly_;
  util::filepath InputFile_;
  util::filepath OutputFile_;
  util::filepath DependencyFile_;
  const std::string Mt_;
};

/**
 * Interface for the command line parser of a Jlm command line tool.
 */
class CommandLineParser
{
public:
  /**
   * Exception thrown in case of command line parsing errors.
   */
  class Exception : public util::error
  {
  public:
    ~Exception() noexcept override;

    explicit Exception(const std::string & message)
        : util::error(message)
    {}
  };

  virtual ~CommandLineParser() noexcept;

  CommandLineParser() = default;

  virtual const CommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) = 0;
};

/**
 * Command line parser for \a jlc command line tool.
 */
class JlcCommandLineParser final : public CommandLineParser
{
public:
  ~JlcCommandLineParser() noexcept override;

  const JlcCommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) override;

private:
  static bool
  IsObjectFile(const util::filepath & file)
  {
    return file.suffix() == "o";
  }

  static util::filepath
  ToObjectFile(const util::filepath & file)
  {
    return { file.path() + file.base() + ".o" };
  }

  static util::filepath
  ToDependencyFile(const util::filepath & file)
  {
    return { file.path() + file.base() + ".d" };
  }

  JlcCommandLineOptions CommandLineOptions_;
};

/**
 * Command line parser for \a jlm-opt command line tool.
 */
class JlmOptCommandLineParser final : public CommandLineParser
{
public:
  ~JlmOptCommandLineParser() noexcept override;

  const JlmOptCommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) override;

  static const JlmOptCommandLineOptions &
  Parse(int argc, char ** argv);

private:
  std::unique_ptr<JlmOptCommandLineOptions> CommandLineOptions_;
};

/**
 * Command line parser for \a jlm-hls command line tool.
 */
class JlmHlsCommandLineParser final : public CommandLineParser
{
public:
  ~JlmHlsCommandLineParser() noexcept override;

  const JlmHlsCommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) override;

  static const JlmHlsCommandLineOptions &
  Parse(int argc, char ** argv);

private:
  JlmHlsCommandLineOptions CommandLineOptions_;
};

/**
 * Command line parser for \a jhls command line tool.
 */
class JhlsCommandLineParser final : public CommandLineParser
{
public:
  ~JhlsCommandLineParser() noexcept override;

  const JhlsCommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) override;

  static const JhlsCommandLineOptions &
  Parse(int argc, char ** arv);

private:
  static bool
  IsObjectFile(const util::filepath & file);

  static util::filepath
  CreateObjectFileFromFile(const util::filepath & f);

  static util::filepath
  CreateDependencyFileFromFile(const util::filepath & f);

  JhlsCommandLineOptions CommandLineOptions_;
};

}

#endif // JLM_TOOLING_COMMANDLINE_HPP
