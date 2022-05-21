/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/Command.hpp>

namespace jlm {

Command::~Command()
= default;

PrintCommandsCommand::~PrintCommandsCommand()
{}

std::string
PrintCommandsCommand::ToString() const
{
  return "PRINTCMD";
}

void
PrintCommandsCommand::Run() const
{
  for (auto & node : CommandGraph::SortNodesTopological(*pgraph_)) {
    if (node != &pgraph_->GetEntryNode() && node != &pgraph_->GetExitNode())
      std::cout << node->GetCommand().ToString() << "\n";
  }
}

}
