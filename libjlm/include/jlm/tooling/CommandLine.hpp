/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMANDLINE_HPP
#define JLM_TOOLING_COMMANDLINE_HPP

#include <jlm/util/file.hpp>

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

}

#endif //JLM_TOOLING_COMMANDLINE_HPP