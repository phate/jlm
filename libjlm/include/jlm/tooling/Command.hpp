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
  virtual
  ~ClangCommand();

  ClangCommand(
    const std::vector<jlm::filepath> & ifiles,
    const jlm::filepath & ofile,
    const std::vector<std::string> & Lpaths,
    const std::vector<std::string> & libs,
    bool pthread)
    : ofile_(ofile)
    , libs_(libs)
    , ifiles_(ifiles)
    , Lpaths_(Lpaths)
    , pthread_(pthread)
  {}

  virtual std::string
  ToString() const override;

  virtual void
  Run() const override;

  inline const jlm::filepath &
  ofile() const noexcept
  {
    return ofile_;
  }

  inline const std::vector<jlm::filepath> &
  ifiles() const noexcept
  {
    return ifiles_;
  }

  static CommandGraph::Node *
  create(
    CommandGraph * pgraph,
    const std::vector<jlm::filepath> & ifiles,
    const jlm::filepath & ofile,
    const std::vector<std::string> & Lpaths,
    const std::vector<std::string> & libs,
    bool pthread)
  {
    std::unique_ptr<ClangCommand> cmd(new ClangCommand(ifiles, ofile, Lpaths, libs, pthread));
    return &CommandGraph::Node::Create(*pgraph, std::move(cmd));
  }

private:
  jlm::filepath ofile_;
  std::vector<std::string> libs_;
  std::vector<jlm::filepath> ifiles_;
  std::vector<std::string> Lpaths_;
  bool pthread_;
};

}

#endif
