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

}

#endif
