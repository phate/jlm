/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/tooling/Command.hpp>

#include <deque>
#include <functional>

namespace jlm {

/**
 * This class represents a dummy command that is used for the single entry node of the command graph. Its Run() method
 * does nothing.
 */
class EntryCommand final : public Command {
public:
  [[nodiscard]] std::string
  ToString() const override
  {
    return "Entry";
  }

  void
  Run() const override
  {}
};

/**
 * This class represents a dummy command that is used for the single exit node of the command graph. Its Run() method
 * does nothing.
 */
class ExitCommand final : public Command {
public:
  [[nodiscard]] std::string
  ToString() const override
  {
    return "Exit";
  }

  void
  Run() const override
  {}
};

CommandGraph::CommandGraph()
{
  EntryNode_ = &Node::Create(*this, std::make_unique<EntryCommand>());
  ExitNode_ = &Node::Create(*this, std::make_unique<ExitCommand>());
}

std::vector<CommandGraph::Node*>
CommandGraph::SortNodesTopological(const CommandGraph & commandGraph)
{
  std::vector<CommandGraph::Node*> nodes({&commandGraph.GetEntryNode()});
  std::deque<CommandGraph::Node*> to_visit({&commandGraph.GetEntryNode()});
  std::unordered_set<CommandGraph::Node*> visited({&commandGraph.GetEntryNode()});

  while (!to_visit.empty()) {
    auto node = to_visit.front();
    to_visit.pop_front();

    for (auto & edge : node->OutgoingEdges()) {
      if (visited.find(&edge.GetSink()) == visited.end()) {
        to_visit.push_back(&edge.GetSink());
        visited.insert(&edge.GetSink());
        nodes.push_back(&edge.GetSink());
      }
    }
  }

  return nodes;
}

void
CommandGraph::Run() const
{
  for (auto & node : CommandGraph::SortNodesTopological(*this))
    node->GetCommand().Run();
}

CommandGraph::Node::~Node()
= default;

CommandGraph::Node::Node(
  const CommandGraph & commandGraph,
  std::unique_ptr<Command> command)
  : CommandGraph_(commandGraph)
  , Command_(std::move(command))
{}

CommandGraph::Node::IncomingEdgeConstRange
CommandGraph::Node::IncomingEdges() const
{
  return {IncomingEdgeConstIterator(IncomingEdges_.begin()), IncomingEdgeConstIterator(IncomingEdges_.end())};
}

CommandGraph::Node::OutgoingEdgeConstRange
CommandGraph::Node::OutgoingEdges() const
{
  return {OutgoingEdgeConstIterator(OutgoingEdges_.begin()), OutgoingEdgeConstIterator(OutgoingEdges_.end())};
}

CommandGraph::Node &
CommandGraph::Node::Create(
  CommandGraph & commandGraph,
  std::unique_ptr<Command> command)
{
  std::unique_ptr<Node> node(new Node(commandGraph, std::move(command)));
  return commandGraph.AddNode(std::move(node));
}

}
