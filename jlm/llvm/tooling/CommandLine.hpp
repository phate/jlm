/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_TOOLING_COMMANDLINE_HPP
#define JLM_LLVM_TOOLING_COMMANDLINE_HPP

#include <jlm/util/file.hpp>
#include <jlm/util/Statistics.hpp>

#include <vector>

namespace jlm
{

/**
 * Interface for the command line options of a Jlm command line tool.
 */
class CommandLineOptions {
public:
  virtual
  ~CommandLineOptions();

  CommandLineOptions()
  = default;

  /**
   * Resets the state of the instance.
   */
  virtual void
  Reset() noexcept = 0;
};

class JlcCommandLineOptions final : public CommandLineOptions {
public:
  class Compilation;

  enum class OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
  };

  enum class LanguageStandard {
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
    : OnlyPrintCommands_(false)
    , GenerateDebugInformation_(false)
    , Verbose_(false)
    , Rdynamic_(false)
    , Suppress_(false)
    , UsePthreads_(false)
    , Md_(false)
    , OptimizationLevel_(OptimizationLevel::O0)
    , LanguageStandard_(LanguageStandard::None)
    , OutputFile_("a.out")
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

  filepath OutputFile_;
  std::vector<std::string> Libraries_;
  std::vector<std::string> MacroDefinitions_;
  std::vector<std::string> LibraryPaths_;
  std::vector<std::string> Warnings_;
  std::vector<std::string> IncludePaths_;
  std::vector<std::string> Flags_;
  std::vector<std::string> JlmOptOptimizations_;

  std::vector<Compilation> Compilations_;
};

class JlcCommandLineOptions::Compilation {
public:
  Compilation(
    filepath inputFile,
    filepath dependencyFile,
    filepath outputFile,
    std::string mT,
    bool requiresParsing,
    bool requiresOptimization,
    bool requiresAssembly,
    bool requiresLinking)
    : RequiresLinking_(requiresLinking)
    , RequiresParsing_(requiresParsing)
    , RequiresOptimization_(requiresOptimization)
    , RequiresAssembly_(requiresAssembly)
    , InputFile_(std::move(inputFile))
    , OutputFile_(std::move(outputFile))
    , DependencyFile_(std::move(dependencyFile))
    , Mt_(std::move(mT))
  {}

  [[nodiscard]] const filepath &
  InputFile() const noexcept
  {
    return InputFile_;
  }

  [[nodiscard]] const filepath &
  DependencyFile() const noexcept
  {
    return DependencyFile_;
  }

  [[nodiscard]] const filepath &
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
  SetOutputFile(const filepath & outputFile)
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
  filepath InputFile_;
  filepath OutputFile_;
  filepath DependencyFile_;
  const std::string Mt_;
};

class optimization;

/**
 * Command line options for the \a jlm-opt command line tool.
 */
class JlmOptCommandLineOptions final : public CommandLineOptions {
public:
  enum class OutputFormat {
    Llvm,
    Xml
  };

  JlmOptCommandLineOptions()
    : InputFile_("")
    , OutputFile_("")
    , OutputFormat_(OutputFormat::Llvm)
  {}

  void
  Reset() noexcept override;

  filepath InputFile_;
  filepath OutputFile_;
  OutputFormat OutputFormat_;
  StatisticsCollectorSettings StatisticsCollectorSettings_;
  std::vector<optimization*> Optimizations_;
};

/**
 * Command line options for the \a jlm-hls command line tool.
 */
class JlmHlsCommandLineOptions final : public CommandLineOptions {
public:
  enum class OutputFormat {
    Firrtl,
    Dot
  };

  JlmHlsCommandLineOptions()
    : InputFile_("")
    , OutputFolder_("")
    , OutputFormat_(OutputFormat::Firrtl)
    , ExtractHlsFunction_(false)
    , UseCirct_(false)
  {}

  void
  Reset() noexcept override;

  filepath InputFile_;
  filepath OutputFolder_;
  OutputFormat OutputFormat_;
  std::string HlsFunction_;
  bool ExtractHlsFunction_;
  bool UseCirct_;
};

/**
 * Command line options for the \a jhls command line tool.
 */
class JhlsCommandLineOptions final : public CommandLineOptions {
public:
  class Compilation;

  enum class OptimizationLevel {
    O0,
    O1,
    O2,
    O3
  };

  enum class LanguageStandard {
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
    : OnlyPrintCommands_(false)
    , GenerateDebugInformation_(false)
    , Verbose_(false)
    , Rdynamic_(false)
    , Suppress_(false)
    , UsePthreads_(false)
    , GenerateFirrtl_(false)
    , UseCirct_(false)
    , Hls_(false)
    , Md_(false)
    , OptimizationLevel_(OptimizationLevel::O0)
    , LanguageStandard_(LanguageStandard::None)
    , OutputFile_("a.out")
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
  filepath OutputFile_;
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

class JhlsCommandLineOptions::Compilation {
public:
  Compilation(
    filepath inputFile,
    filepath dependencyFile,
    filepath outputFile,
    std::string mT,
    bool parse,
    bool optimize,
    bool assemble,
    bool link)
    : RequiresLinking_(link)
    , RequiresParsing_(parse)
    , RequiresOptimization_(optimize)
    , RequiresAssembly_(assemble)
    , InputFile_(std::move(inputFile))
    , OutputFile_(std::move(outputFile))
    , DependencyFile_(std::move(dependencyFile))
    , Mt_(std::move(mT))
  {}

  [[nodiscard]] const filepath &
  InputFile() const noexcept
  {
    return InputFile_;
  }

  [[nodiscard]] const filepath &
  DependencyFile() const noexcept
  {
    return DependencyFile_;
  }

  [[nodiscard]] const filepath &
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
  SetOutputFile(const filepath & outputFile)
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
  filepath InputFile_;
  filepath OutputFile_;
  filepath DependencyFile_;
  const std::string Mt_;
};

/**
 * Interface for the command line parser of a Jlm command line tool.
 */
class CommandLineParser {
public:
  virtual
  ~CommandLineParser() noexcept;

  CommandLineParser()
  = default;

  virtual const CommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) = 0;
};

/**
 * Command line parser for \a jlc command line tool.
 */
class JlcCommandLineParser final : public CommandLineParser {
public:
  ~JlcCommandLineParser() noexcept override;

  const JlcCommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) override;

private:
  static bool
  IsObjectFile(const filepath & file)
  {
    return file.suffix() == "o";
  }

  static filepath
  ToObjectFile(const filepath & file)
  {
    return {file.path() + file.base() + ".o"};
  }

  static filepath
  ToDependencyFile(const filepath & file)
  {
    return {file.path() + file.base() + ".d"};
  }

  JlcCommandLineOptions CommandLineOptions_;
};

/**
 * Command line parser for \a jlm-opt command line tool.
 */
class JlmOptCommandLineParser final : public CommandLineParser {
public:
  enum class OptimizationId {
    AASteensgaardAgnostic,
    AASteensgaardRegionAware,
    cne,
    dne,
    iln,
    InvariantValueRedirection,
    psh,
    red,
    ivt,
    url,
    pll,
  };

  ~JlmOptCommandLineParser() noexcept override;

  const JlmOptCommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) override;

  static const JlmOptCommandLineOptions &
  Parse(int argc, char ** argv);

private:
  static optimization *
  GetOptimization(enum OptimizationId optimizationId);

  JlmOptCommandLineOptions CommandLineOptions_;
};

/**
 * Command line parser for \a jlm-hls command line tool.
 */
class JlmHlsCommandLineParser final : public CommandLineParser {
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
class JhlsCommandLineParser final : public CommandLineParser {
public:
  ~JhlsCommandLineParser() noexcept override;

  const JhlsCommandLineOptions &
  ParseCommandLineArguments(int argc, char ** argv) override;

  static const JhlsCommandLineOptions &
  Parse(int argc, char ** arv);

private:
  static bool
  IsObjectFile(const filepath & file);

  static filepath
  CreateObjectFileFromFile(const filepath & f);

  static filepath
  CreateDependencyFileFromFile(const filepath & f);

  JhlsCommandLineOptions CommandLineOptions_;
};

}

#endif //JLM_LLVM_TOOLING_COMMANDLINE_HPP