/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMAND_HPP
#define JLM_TOOLING_COMMAND_HPP

#include <jlm/tooling/CommandGraph.hpp>
#include <jlm/util/file.hpp>

#include <memory>
#include <string>

namespace jlm {

/** \brief Command class
 *
 * This class represents simple commands, such as \a mkdir or \a rm, that can be executed with the Run() method.
 */
class Command {
public:
  virtual
  ~Command();

  [[nodiscard]] virtual std::string
  ToString() const = 0;

  virtual void
  Run() const = 0;
};

/**
 * The PrintCommandsCommand class prints the commands of a command graph in topological order.
 */
class PrintCommandsCommand final : public Command {
public:
  ~PrintCommandsCommand() override;

  explicit
  PrintCommandsCommand(std::unique_ptr<CommandGraph> commandGraph)
    : CommandGraph_(std::move(commandGraph))
  {}

  PrintCommandsCommand(const PrintCommandsCommand&) = delete;

  PrintCommandsCommand(PrintCommandsCommand&&) = delete;

  PrintCommandsCommand &
  operator=(const PrintCommandsCommand&) = delete;

  PrintCommandsCommand &
  operator=(PrintCommandsCommand&&)	= delete;

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  static std::unique_ptr<CommandGraph>
  Create(std::unique_ptr<CommandGraph> commandGraph);

private:
  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    std::unique_ptr<CommandGraph> printedCommandGraph)
  {
    auto command = std::make_unique<PrintCommandsCommand>(std::move(printedCommandGraph));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

  std::unique_ptr<CommandGraph> CommandGraph_;
};

/**
 * The ClangCommand class represents the clang command line tool.
 */
class ClangCommand final : public Command {
public:
  ~ClangCommand() override;

  ClangCommand(
    std::vector<jlm::filepath> inputFiles,
    filepath outputFile,
    std::vector<std::string> libraryPaths,
    std::vector<std::string> libraries,
    bool usePthreads)
    : OutputFile_(std::move(outputFile))
    , Libraries_(std::move(libraries))
    , InputFiles_(std::move(inputFiles))
    , LibraryPaths_(std::move(libraryPaths))
    , UsePthreads_(usePthreads)
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] const filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  [[nodiscard]] const std::vector<filepath> &
  InputFiles() const noexcept
  {
    return InputFiles_;
  }

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const std::vector<jlm::filepath> & inputFiles,
    const filepath & outputFile,
    const std::vector<std::string> & libraryPaths,
    const std::vector<std::string> & libraries,
    bool usePthreads)
  {
    std::unique_ptr<ClangCommand> command(new ClangCommand(
      inputFiles,
      outputFile,
      libraryPaths,
      libraries,
      usePthreads));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  filepath OutputFile_;
  std::vector<std::string> Libraries_;
  std::vector<jlm::filepath> InputFiles_;
  std::vector<std::string> LibraryPaths_;
  bool UsePthreads_;
};

/**
 * The LlcCommand class represents the llc command line tool.
 */
class LlcCommand final : public Command {
public:
  enum class OptimizationLevel {
    O0,
    O1,
    O2,
    O3
  };

  enum class RelocationModel {
    Static,
    Pic
  };

  ~LlcCommand() override;

  LlcCommand(
    filepath inputFile,
    filepath outputFile,
    const OptimizationLevel & optimizationLevel,
    const RelocationModel & relocationModel)
    : OptimizationLevel_(optimizationLevel)
    , RelocationModel_(relocationModel)
    , InputFile_(std::move(inputFile))
    , OutputFile_(std::move(outputFile))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  [[nodiscard]] const filepath &
  OutputFile() const noexcept
  {
    return OutputFile_;
  }

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFile,
    const OptimizationLevel & optimizationLevel,
    const RelocationModel & relocationModel)
  {
    std::unique_ptr<LlcCommand> command(new LlcCommand(
      inputFile,
      outputFile,
      optimizationLevel,
      relocationModel));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  static std::string
  ToString(const OptimizationLevel & optimizationLevel);

  static std::string
  ToString(const RelocationModel & relocationModel);

  OptimizationLevel OptimizationLevel_;
  RelocationModel RelocationModel_;
  filepath InputFile_;
  filepath OutputFile_;
};

/**
 * The JlmOptCommand class represents the jlm-opt command line tool.
 */
class JlmOptCommand final : public Command {
public:
  enum class Optimization {
    AASteensgaardBasic,
    CommonNodeElimination,
    DeadNodeElimination,
    FunctionInlining,
    InvariantValueRedirection,
    LoopUnrolling,
    NodePullIn,
    NodePushOut,
    NodeReduction,
    ThetaGammaInversion
  };

  ~JlmOptCommand() override;

  JlmOptCommand(
    filepath inputFile,
    filepath outputFile,
    std::vector<Optimization> optimizations)
    : InputFile_(std::move(inputFile))
    , OutputFile_(std::move(outputFile))
    , Optimizations_(std::move(optimizations))
  {}

  [[nodiscard]] std::string
  ToString() const override;

  void
  Run() const override;

  static CommandGraph::Node &
  Create(
    CommandGraph & commandGraph,
    const filepath & inputFile,
    const filepath & outputFile,
    const std::vector<Optimization> & optimizations)
  {
    std::unique_ptr<JlmOptCommand> command(new JlmOptCommand(inputFile, outputFile, optimizations));
    return CommandGraph::Node::Create(commandGraph, std::move(command));
  }

private:
  static std::string
  ToString(const Optimization & optimization);

  filepath InputFile_;
  filepath OutputFile_;
  std::vector<Optimization> Optimizations_;
};

class prscmd final : public Command {
public:
  enum class LanguageStandard {
    Unspecified,
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

  enum class ClangArgument {
    DisableO0OptNone
  };

  virtual
  ~prscmd();

  prscmd(
    const jlm::filepath & ifile,
    filepath outputFile,
    const jlm::filepath & dependencyFile,
    const std::vector<std::string> & Ipaths,
    const std::vector<std::string> & Dmacros,
    const std::vector<std::string> & Wwarnings,
    const std::vector<std::string> & flags,
    bool verbose,
    bool rdynamic,
    bool suppress,
    bool pthread,
    bool MD,
    const std::string & mT,
    const LanguageStandard & languageStandard,
    std::vector<ClangArgument> clangArguments)
    : LanguageStandard_(languageStandard)
    , ifile_(ifile)
    , OutputFile_(std::move(outputFile))
    , Ipaths_(Ipaths)
    , Dmacros_(Dmacros)
    , Wwarnings_(Wwarnings)
    , flags_(flags)
    , verbose_(verbose)
    , rdynamic_(rdynamic)
    , suppress_(suppress)
    , pthread_(pthread)
    , MD_(MD)
    , mT_(mT)
    , dependencyFile_(dependencyFile)
    , ClangArguments_(std::move(clangArguments))
  {}

  virtual std::string
  ToString() const override;

  jlm::filepath
  ofile() const;

  virtual void
  Run() const override;

  static CommandGraph::Node *
  create(
    CommandGraph * pgraph,
    const jlm::filepath & ifile,
    const filepath & outputFile,
    const jlm::filepath & dependencyFile,
    const std::vector<std::string> & Ipaths,
    const std::vector<std::string> & Dmacros,
    const std::vector<std::string> & Wwarnings,
    const std::vector<std::string> & flags,
    bool verbose,
    bool rdynamic,
    bool suppress,
    bool pthread,
    bool MD,
    const std::string & mT,
    const LanguageStandard & languageStandard,
    const std::vector<ClangArgument> & clangArguments)
  {
    std::unique_ptr<prscmd> cmd(new prscmd(
      ifile,
      outputFile,
      dependencyFile,
      Ipaths,
      Dmacros,
      Wwarnings,
      flags,
      verbose,
      rdynamic,
      suppress,
      pthread,
      MD,
      mT,
      languageStandard,
      clangArguments));

    return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
  }

private:
  static std::string
  ToString(const LanguageStandard & languageStandard);

  static std::string
  ToString(const ClangArgument & clangArgument);

  static std::string
  replace_all(std::string str, const std::string& from, const std::string& to);

  LanguageStandard LanguageStandard_;
  jlm::filepath ifile_;
  filepath OutputFile_;
  std::vector<std::string> Ipaths_;
  std::vector<std::string> Dmacros_;
  std::vector<std::string> Wwarnings_;
  std::vector<std::string> flags_;
  bool verbose_;
  bool rdynamic_;
  bool suppress_;
  bool pthread_;
  bool MD_;
  std::string mT_;
  jlm::filepath dependencyFile_;
  std::vector<ClangArgument> ClangArguments_;
};

}

#endif
