/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/tooling/Command.hpp>
#include <jlm/tooling/LlvmPaths.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm {

Command::~Command()
= default;

PrintCommandsCommand::~PrintCommandsCommand()
= default;

std::string
PrintCommandsCommand::ToString() const
{
  return "PrintCommands";
}

void
PrintCommandsCommand::Run() const
{
  for (auto & node : CommandGraph::SortNodesTopological(*CommandGraph_)) {
    if (node != &CommandGraph_->GetEntryNode() && node != &CommandGraph_->GetExitNode())
      std::cout << node->GetCommand().ToString() << "\n";
  }
}

std::unique_ptr<CommandGraph>
PrintCommandsCommand::Create(std::unique_ptr<CommandGraph> commandGraph)
{
  std::unique_ptr<CommandGraph> newCommandGraph(new CommandGraph());
  auto & printCommandsNode = PrintCommandsCommand::Create(*newCommandGraph, std::move(commandGraph));
  newCommandGraph->GetEntryNode().AddEdge(printCommandsNode);
  printCommandsNode.AddEdge(newCommandGraph->GetExitNode());
  return newCommandGraph;
}

lnkcmd::~lnkcmd()
{}

std::string
lnkcmd::ToString() const
{
  std::string ifiles;
  for (const auto & ifile : ifiles_)
    ifiles += ifile.to_str() + " ";

  std::string Lpaths;
  for (const auto & Lpath : Lpaths_)
    Lpaths += "-L" + Lpath + " ";

  std::string libs;
  for (const auto & lib : libs_)
    libs += "-l" + lib + " ";

  std::string arguments;
  if (pthread_)
    arguments += "-pthread ";

  return strfmt(
    clangpath.to_str() + " "
    , "-no-pie -O0 "
    , arguments
    , ifiles
    , "-o ", ofile_.to_str(), " "
    , Lpaths
    , libs
  );
}

void
lnkcmd::Run() const
{
  if (system(ToString().c_str()))
    exit(EXIT_FAILURE);
}

}
