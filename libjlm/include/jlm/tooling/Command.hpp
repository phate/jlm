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

class printcmd final : public Command {
public:
  virtual
  ~printcmd();

  printcmd(
    std::unique_ptr<CommandGraph> pgraph)
    : pgraph_(std::move(pgraph))
  {}

  printcmd(const printcmd&) = delete;

  printcmd(printcmd&&) = delete;

  printcmd &
  operator=(const printcmd&) = delete;

  printcmd &
  operator=(printcmd&&)	= delete;

  virtual std::string
  ToString() const override;

  virtual void
  Run() const override;

  static CommandGraph::Node *
  create(
    CommandGraph * pgraph,
    std::unique_ptr<CommandGraph> pg)
  {
    return &CommandGraph::Node::Create(*pgraph, std::make_unique<printcmd>(std::move(pg)));
  }

private:
  std::unique_ptr<CommandGraph> pgraph_;
};

}

#endif
