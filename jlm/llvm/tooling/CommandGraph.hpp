/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_TOOLING_COMMANDGRAPH_HPP
#define JLM_LLVM_TOOLING_COMMANDGRAPH_HPP

#include <jlm/util/common.hpp>
#include <jlm/util/iterator_range.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

namespace jlm {

class Command;

/**
 * A simple dependency graph for command execution.
 */
class CommandGraph final {
public:
  class Edge;
  class Node;

  ~CommandGraph()
  = default;

  CommandGraph();

  CommandGraph(const CommandGraph&) = delete;

  CommandGraph(CommandGraph&&) = delete;

  CommandGraph &
  operator=(const CommandGraph&) = delete;

  CommandGraph &
  operator=(CommandGraph&&) = delete;

  Node &
  GetEntryNode() const noexcept
  {
    return *EntryNode_;
  }

  Node &
  GetExitNode() const noexcept
  {
    return *ExitNode_;
  }

  size_t
  NumNodes() const noexcept
  {
    return Nodes_.size();
  }

  Node &
  AddNode(std::unique_ptr<Node> node)
  {
    auto pointer = node.get();
    Nodes_.insert(std::move(node));
    return *pointer;
  }

  void
  Run() const;

  static std::vector<CommandGraph::Node*>
  SortNodesTopological(const CommandGraph & commandGraph);

  static std::unique_ptr<CommandGraph>
  Create()
  {
    return std::make_unique<CommandGraph>();
  }

private:
  Node * ExitNode_;
  Node * EntryNode_;
  std::unordered_set<std::unique_ptr<Node>> Nodes_;
};

/**
 * Represents a dependency between two commands in the command graph. The dependency indicates that the command
 * represented by the GetSource() node should be executed before the command of the GetSink() node.
 */
class CommandGraph::Edge final {
public:
  Edge(
    Node & source,
    Node & sink)
    : Sink_(&sink)
    , Source_(&source)
  {}

  [[nodiscard]] Node &
  GetSource() const noexcept
  {
    return *Source_;
  }

  [[nodiscard]] Node &
  GetSink() const noexcept
  {
    return *Sink_;
  }

private:
  Node * Sink_;
  Node * Source_;
};

/**
 * Represents a single command in the command graph.
 */
class CommandGraph::Node final {
  class IncomingEdgeConstIterator;
  class OutgoingEdgeConstIterator;

  using IncomingEdgeConstRange = iterator_range<IncomingEdgeConstIterator>;
  using OutgoingEdgeConstRange = iterator_range<OutgoingEdgeConstIterator>;

public:
  ~Node();

private:
  Node(
    const CommandGraph & commandGraph,
    std::unique_ptr<Command> command);

public:
  Node(const Node&) = delete;

  Node(Node&&) = delete;

  Node &
  operator=(const Node&) = delete;

  Node &
  operator=(Node&&) = delete;

  const CommandGraph &
  GetCommandGraph() const noexcept
  {
    return CommandGraph_;
  }

  Command &
  GetCommand() const noexcept
  {
    return *Command_;
  }

  size_t
  NumIncomingEdges() const noexcept
  {
    return IncomingEdges_.size();
  }

  size_t
  NumOutgoingEdges() const noexcept
  {
    return OutgoingEdges_.size();
  }

  IncomingEdgeConstRange
  IncomingEdges() const;

  OutgoingEdgeConstRange
  OutgoingEdges() const;

  void
  AddEdge(Node & sink)
  {
    std::unique_ptr<Edge> edge(new Edge(*this, sink));
    auto pointer = edge.get();
    OutgoingEdges_.insert(std::move(edge));
    sink.IncomingEdges_.insert(pointer);
  }

  static Node &
  Create(
    CommandGraph & commandGraph,
    std::unique_ptr<Command> command);

private:
  const CommandGraph & CommandGraph_;
  std::unique_ptr<Command> Command_;
  std::unordered_set<Edge*> IncomingEdges_;
  std::unordered_set<std::unique_ptr<Edge>> OutgoingEdges_;
};

/** \brief Command graph node incoming edge const iterator
*/
class CommandGraph::Node::IncomingEdgeConstIterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Edge*;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge**;
  using reference = Edge*&;

private:
  friend Node;

  explicit
  IncomingEdgeConstIterator(const std::unordered_set<Edge*>::const_iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] const Edge *
  GetEdge() const noexcept
  {
    return *it_;
  }

  const Edge &
  operator*() const
  {
    JLM_ASSERT(GetEdge() != nullptr);
    return *GetEdge();
  }

  const Edge *
  operator->() const
  {
    return GetEdge();
  }

  IncomingEdgeConstIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  IncomingEdgeConstIterator
  operator++(int)
  {
    IncomingEdgeConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const IncomingEdgeConstIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const IncomingEdgeConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  std::unordered_set<Edge*>::const_iterator it_;
};

/** \brief Command graph node outgoing edge const iterator
*/
class CommandGraph::Node::OutgoingEdgeConstIterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Edge*;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge**;
  using reference = Edge*&;

private:
  friend Node;

  explicit
  OutgoingEdgeConstIterator(const std::unordered_set<std::unique_ptr<Edge>>::const_iterator & it)
    : it_(it)
  {}

public:
  [[nodiscard]] const Edge *
  GetEdge() const noexcept
  {
    return it_->get();
  }

  const Edge &
  operator*() const
  {
    JLM_ASSERT(GetEdge() != nullptr);
    return *GetEdge();
  }

  const Edge *
  operator->() const
  {
    return GetEdge();
  }

  OutgoingEdgeConstIterator &
  operator++()
  {
    ++it_;
    return *this;
  }

  OutgoingEdgeConstIterator
  operator++(int)
  {
    OutgoingEdgeConstIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool
  operator==(const OutgoingEdgeConstIterator & other) const
  {
    return it_ == other.it_;
  }

  bool
  operator!=(const OutgoingEdgeConstIterator & other) const
  {
    return !operator==(other);
  }

private:
  std::unordered_set<std::unique_ptr<Edge>>::const_iterator it_;
};

}

#endif
