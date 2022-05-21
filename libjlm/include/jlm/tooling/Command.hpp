/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_TOOLING_COMMAND_HPP
#define JLM_TOOLING_COMMAND_HPP

#include <jlm/tooling/CommandGraph.hpp>

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
  virtual
  ~PrintCommandsCommand();

  PrintCommandsCommand(
    std::unique_ptr<CommandGraph> pgraph)
    : pgraph_(std::move(pgraph))
  {}

  PrintCommandsCommand(const PrintCommandsCommand&) = delete;

  PrintCommandsCommand(PrintCommandsCommand&&) = delete;

  PrintCommandsCommand &
  operator=(const PrintCommandsCommand&) = delete;

  PrintCommandsCommand &
  operator=(PrintCommandsCommand&&)	= delete;

  virtual std::string
  ToString() const override;

  virtual void
  Run() const override;

  static CommandGraph::Node *
  create(
    CommandGraph * pgraph,
    std::unique_ptr<CommandGraph> pg)
  {
    return &CommandGraph::Node::Create(*pgraph, std::make_unique<PrintCommandsCommand>(std::move(pg)));
  }

private:
  std::unique_ptr<CommandGraph> pgraph_;
};

}

#endif
