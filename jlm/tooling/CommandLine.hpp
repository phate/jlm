/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMANDLINE_HPP
#define JLM_TOOLING_COMMANDLINE_HPP

#include <jlm/llvm/opt/RvsdgTreePrinter.hpp>
#include <jlm/util/BijectiveMap.hpp>
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
  enum class InputFormat
  {
    Llvm,
    Mlir,
  };

  enum class OutputFormat
  {
    FirstEnumValue, // must always be the first enum value, used for iteration

    Ascii,
    Dot,
    Llvm,
    Mlir,
    Tree,
    Xml,

    LastEnumValue // must always be the last enum value, used for iteration
  };

  enum class OptimizationId
  {
    FirstEnumValue, // must always be the first enum value, used for iteration

    AAAndersenAgnostic,
    AAAndersenRegionAware,
    AAAndersenTopDownLifetimeAware,
    AASteensgaardAgnostic,
    AASteensgaardRegionAware,
    CommonNodeElimination,
    DeadNodeElimination,
    FunctionInlining,
    InvariantValueRedirection,
    LoopUnrolling,
    NodePullIn,
    NodePushOut,
    NodeReduction,
    RvsdgTreePrinter,
    ThetaGammaInversion,

    LastEnumValue // must always be the last enum value, used for iteration
  };

  JlmOptCommandLineOptions(
      util::filepath inputFile,
      InputFormat inputFormat,
      util::filepath outputFile,
      OutputFormat outputFormat,
      util::StatisticsCollectorSettings statisticsCollectorSettings,
      llvm::RvsdgTreePrinter::Configuration rvsdgTreePrinterConfiguration,
      std::vector<OptimizationId> optimizations)
      : InputFile_(std::move(inputFile)),
        InputFormat_(inputFormat),
        OutputFile_(std::move(outputFile)),
        OutputFormat_(outputFormat),
        StatisticsCollectorSettings_(std::move(statisticsCollectorSettings)),
        OptimizationIds_(std::move(optimizations)),
        RvsdgTreePrinterConfiguration_(std::move(rvsdgTreePrinterConfiguration))
  {}

  void
  Reset() noexcept override;

  [[nodiscard]] const util::filepath &
  GetInputFile() const noexcept
  {
    return InputFile_;
  }

  [[nodiscard]] InputFormat
  GetInputFormat() const noexcept
  {
    return InputFormat_;
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

  [[nodiscard]] const llvm::RvsdgTreePrinter::Configuration &
  GetRvsdgTreePrinterConfiguration() const noexcept
  {
    return RvsdgTreePrinterConfiguration_;
  }

  static OptimizationId
  FromCommandLineArgumentToOptimizationId(const std::string & commandLineArgument);

  static util::Statistics::Id
  FromCommandLineArgumentToStatisticsId(const std::string & commandLineArgument);

  static const char *
  ToCommandLineArgument(OptimizationId optimizationId);

  static const char *
  ToCommandLineArgument(util::Statistics::Id statisticsId);

  static const char *
  ToCommandLineArgument(InputFormat inputFormat);

  static const char *
  ToCommandLineArgument(OutputFormat outputFormat);

  static std::unique_ptr<JlmOptCommandLineOptions>
  Create(
      util::filepath inputFile,
      InputFormat inputFormat,
      util::filepath outputFile,
      OutputFormat outputFormat,
      util::StatisticsCollectorSettings statisticsCollectorSettings,
      llvm::RvsdgTreePrinter::Configuration rvsdgTreePrinterConfiguration,
      std::vector<OptimizationId> optimizations)
  {
    return std::make_unique<JlmOptCommandLineOptions>(
        std::move(inputFile),
        inputFormat,
        std::move(outputFile),
        outputFormat,
        std::move(statisticsCollectorSettings),
        std::move(rvsdgTreePrinterConfiguration),
        std::move(optimizations));
  }

private:
  util::filepath InputFile_;
  InputFormat InputFormat_;
  util::filepath OutputFile_;
  OutputFormat OutputFormat_;
  util::StatisticsCollectorSettings StatisticsCollectorSettings_;
  std::vector<OptimizationId> OptimizationIds_;
  llvm::RvsdgTreePrinter::Configuration RvsdgTreePrinterConfiguration_;

  struct OptimizationCommandLineArgument
  {
    inline static const char * AaAndersenAgnostic_ = "AAAndersenAgnostic";
    inline static const char * AaAndersenRegionAware_ = "AAAndersenRegionAware";
    inline static const char * AaAndersenTopDownLifetimeAware_ = "AAAndersenTopDownLifetimeAware";
    inline static const char * AaSteensgaardAgnostic_ = "AASteensgaardAgnostic";
    inline static const char * AaSteensgaardRegionAware_ = "AASteensgaardRegionAware";
    inline static const char * CommonNodeElimination_ = "CommonNodeElimination";
    inline static const char * DeadNodeElimination_ = "DeadNodeElimination";
    inline static const char * FunctionInlining_ = "FunctionInlining";
    inline static const char * InvariantValueRedirection_ = "InvariantValueRedirection";
    inline static const char * NodePullIn_ = "NodePullIn";
    inline static const char * NodePushOut_ = "NodePushOut";
    inline static const char * ThetaGammaInversion_ = "ThetaGammaInversion";
    inline static const char * LoopUnrolling_ = "LoopUnrolling";
    inline static const char * NodeReduction_ = "NodeReduction";
    inline static const char * RvsdgTreePrinter_ = "RvsdgTreePrinter";
  };

  static const util::BijectiveMap<util::Statistics::Id, std::string_view> &
  GetStatisticsIdCommandLineArguments();

  static const std::unordered_map<OutputFormat, std::string_view> &
  GetOutputFormatCommandLineArguments();
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
        OutputFiles_(""),
        OutputFormat_(OutputFormat::Firrtl),
        ExtractHlsFunction_(false)
  {}

  void
  Reset() noexcept override;

  util::filepath InputFile_;
  util::filepath OutputFiles_;
  OutputFormat OutputFormat_;
  std::string HlsFunction_;
  bool ExtractHlsFunction_;
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
  ParseCommandLineArguments(int argc, const char * const * argv) = 0;
};

/**
 * Command line parser for \a jlc command line tool.
 */
class JlcCommandLineParser final : public CommandLineParser
{
public:
  ~JlcCommandLineParser() noexcept override;

  const JlcCommandLineOptions &
  ParseCommandLineArguments(int argc, const char * const * argv) override;

private:
  static bool
  IsObjectFile(const util::filepath & file)
  {
    return file.suffix() == "o";
  }

  static util::filepath
  ToObjectFile(const util::filepath & file)
  {
    return file.Dirname().Join(file.base() + ".o");
  }

  static util::filepath
  ToDependencyFile(const util::filepath & file)
  {
    return file.Dirname().Join(file.base() + ".d");
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
  ParseCommandLineArguments(int argc, const char * const * argv) override;

  static const JlmOptCommandLineOptions &
  Parse(int argc, const char * const * argv);

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
  ParseCommandLineArguments(int argc, const char * const * argv) override;

  static const JlmHlsCommandLineOptions &
  Parse(int argc, const char * const * argv);

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
  ParseCommandLineArguments(int argc, const char * const * argv) override;

  static const JhlsCommandLineOptions &
  Parse(int argc, const char * const * arv);

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
