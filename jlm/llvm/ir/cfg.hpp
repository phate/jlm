/*
 * Copyright 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */
#ifndef JLM_LLVM_IR_CFG_HPP
#define JLM_LLVM_IR_CFG_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/ir/variable.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/GraphWriter.hpp>
#include <jlm/util/IteratorWrapper.hpp>

namespace jlm::llvm
{

class clg_node;
class BasicBlock;
class ControlFlowGraph;
class ControlFlowGraphNode;
class InterProceduralGraphModule;
class ThreeAddressCode;

class ControlFlowGraphEdge final
{
public:
  ~ControlFlowGraphEdge() noexcept = default;

  ControlFlowGraphEdge(
      ControlFlowGraphNode * source,
      ControlFlowGraphNode * sink,
      size_t index) noexcept;

  ControlFlowGraphEdge(const ControlFlowGraphEdge & other) = delete;

  ControlFlowGraphEdge(ControlFlowGraphEdge && other) = default;

  ControlFlowGraphEdge &
  operator=(const ControlFlowGraphEdge & other) = delete;

  ControlFlowGraphEdge &
  operator=(ControlFlowGraphEdge && other) = default;

  void
  divert(ControlFlowGraphNode * new_sink);

  BasicBlock *
  split();

  [[nodiscard]] ControlFlowGraphNode *
  source() const noexcept
  {
    return source_;
  }

  [[nodiscard]] ControlFlowGraphNode *
  sink() const noexcept
  {
    return sink_;
  }

  /**
   * @return the index of this edge among the source's out edges
   */
  size_t
  index() const noexcept
  {
    return index_;
  }

  bool
  is_selfloop() const noexcept
  {
    return source_ == sink_;
  }

private:
  ControlFlowGraphNode * source_;
  ControlFlowGraphNode * sink_;
  size_t index_;

  friend ControlFlowGraphNode;
};

class ControlFlowGraphNode
{
  using inedge_iterator = util::
      PtrIterator<ControlFlowGraphEdge, std::unordered_set<ControlFlowGraphEdge *>::const_iterator>;
  using inedge_iterator_range = util::IteratorRange<inedge_iterator>;

  using outedge_iterator = util::PtrIterator<
      ControlFlowGraphEdge,
      std::vector<std::unique_ptr<ControlFlowGraphEdge>>::const_iterator>;
  using outedge_iterator_range = util::IteratorRange<outedge_iterator>;

public:
  virtual ~ControlFlowGraphNode() noexcept;

protected:
  explicit ControlFlowGraphNode(ControlFlowGraph & cfg)
      : cfg_(cfg)
  {}

public:
  ControlFlowGraph &
  cfg() const noexcept
  {
    return cfg_;
  }

  size_t
  NumOutEdges() const noexcept;

  [[nodiscard]] ControlFlowGraphEdge *
  OutEdge(size_t n) const
  {
    JLM_ASSERT(n < NumOutEdges());
    return outedges_[n].get();
  }

  outedge_iterator_range
  OutEdges() const
  {
    return outedge_iterator_range(
        outedge_iterator(outedges_.begin()),
        outedge_iterator(outedges_.end()));
  }

  ControlFlowGraphEdge *
  add_outedge(ControlFlowGraphNode * sink)
  {
    outedges_.push_back(std::make_unique<ControlFlowGraphEdge>(this, sink, NumOutEdges()));
    sink->inedges_.insert(outedges_.back().get());
    return outedges_.back().get();
  }

  void
  remove_outedge(size_t n)
  {
    JLM_ASSERT(n < NumOutEdges());
    auto edge = outedges_[n].get();

    edge->sink()->inedges_.erase(edge);
    for (size_t i = n + 1; i < NumOutEdges(); i++)
    {
      outedges_[i - 1] = std::move(outedges_[i]);
      outedges_[i - 1]->index_ = outedges_[i - 1]->index_ - 1;
    }
    outedges_.resize(NumOutEdges() - 1);
  }

  void
  remove_outedges()
  {
    while (NumOutEdges() != 0)
      remove_outedge(NumOutEdges() - 1);
  }

  size_t
  NumInEdges() const noexcept;

  inedge_iterator_range
  InEdges() const
  {
    return inedge_iterator_range(
        inedge_iterator(inedges_.begin()),
        inedge_iterator(inedges_.end()));
  }

  inline void
  divert_inedges(llvm::ControlFlowGraphNode * new_successor)
  {
    if (this == new_successor)
      return;

    while (NumInEdges())
      InEdges().begin()->divert(new_successor);
  }

  void
  remove_inedges();

  bool
  no_predecessor() const noexcept;

  bool
  single_predecessor() const noexcept;

  bool
  no_successor() const noexcept;

  bool
  single_successor() const noexcept;

  bool
  is_branch() const noexcept
  {
    return NumOutEdges() > 1;
  }

  bool
  has_selfloop_edge() const noexcept;

private:
  ControlFlowGraph & cfg_;
  std::vector<std::unique_ptr<ControlFlowGraphEdge>> outedges_;
  std::unordered_set<ControlFlowGraphEdge *> inedges_;

  friend ControlFlowGraphEdge;
};

template<class T>
static inline bool
is(const ControlFlowGraphNode * node) noexcept
{
  static_assert(
      std::is_base_of<ControlFlowGraphNode, T>::value,
      "Template parameter T must be derived from jlm::ControlFlowGraphNode.");

  return dynamic_cast<const T *>(node) != nullptr;
}

/** \brief Function argument
 */
class Argument final : public Variable
{
public:
  ~Argument() noexcept override;

  Argument(
      const std::string & name,
      std::shared_ptr<const jlm::rvsdg::Type> type,
      const AttributeSet & attributes)
      : Variable(std::move(type), name),
        attributes_(attributes)
  {}

  Argument(const std::string & name, std::shared_ptr<const jlm::rvsdg::Type> type)
      : Variable(std::move(type), name)
  {}

  Argument(
      const std::string & name,
      std::unique_ptr<jlm::rvsdg::Type> type,
      const AttributeSet & attributes)
      : Variable(std::move(type), name),
        attributes_(attributes)
  {}

  const AttributeSet &
  attributes() const noexcept
  {
    return attributes_;
  }

  static std::unique_ptr<Argument>
  create(
      const std::string & name,
      std::shared_ptr<const jlm::rvsdg::Type> type,
      const AttributeSet & attributes)
  {
    return std::make_unique<Argument>(name, std::move(type), attributes);
  }

  static std::unique_ptr<Argument>
  create(const std::string & name, std::shared_ptr<const jlm::rvsdg::Type> type)
  {
    return create(name, std::move(type), {});
  }

private:
  AttributeSet attributes_;
};

class EntryNode final : public ControlFlowGraphNode
{
public:
  ~EntryNode() noexcept override;

  explicit EntryNode(ControlFlowGraph & cfg)
      : ControlFlowGraphNode(cfg)
  {}

  size_t
  narguments() const noexcept
  {
    return arguments_.size();
  }

  const llvm::Argument *
  argument(size_t index) const
  {
    JLM_ASSERT(index < narguments());
    return arguments_[index].get();
  }

  llvm::Argument *
  append_argument(std::unique_ptr<llvm::Argument> arg)
  {
    arguments_.push_back(std::move(arg));
    return arguments_.back().get();
  }

  std::vector<llvm::Argument *>
  arguments() const noexcept
  {
    std::vector<llvm::Argument *> arguments;
    for (auto & argument : arguments_)
      arguments.push_back(argument.get());

    return arguments;
  }

private:
  std::vector<std::unique_ptr<llvm::Argument>> arguments_;
};

class ExitNode final : public ControlFlowGraphNode
{
public:
  ~ExitNode() noexcept override;

  explicit ExitNode(ControlFlowGraph & cfg)
      : ControlFlowGraphNode(cfg)
  {}

  size_t
  nresults() const noexcept
  {
    return results_.size();
  }

  const Variable *
  result(size_t index) const
  {
    JLM_ASSERT(index < nresults());
    return results_[index];
  }

  inline void
  append_result(const Variable * v)
  {
    results_.push_back(v);
  }

  const std::vector<const Variable *>
  results() const noexcept
  {
    return results_;
  }

private:
  std::vector<const Variable *> results_;
};

class ControlFlowGraph final
{
  using iterator =
      util::PtrIterator<BasicBlock, std::unordered_set<std::unique_ptr<BasicBlock>>::iterator>;
  using const_iterator = util::PtrIterator<
      const BasicBlock,
      std::unordered_set<std::unique_ptr<BasicBlock>>::const_iterator>;

public:
  ~ControlFlowGraph() noexcept = default;

  explicit ControlFlowGraph(InterProceduralGraphModule & im);

  ControlFlowGraph(const ControlFlowGraph &) = delete;

  ControlFlowGraph(ControlFlowGraph &&) = delete;

  ControlFlowGraph &
  operator=(const ControlFlowGraph &) = delete;

  ControlFlowGraph &
  operator=(ControlFlowGraph &&) = delete;

  inline const_iterator
  begin() const
  {
    return const_iterator(nodes_.begin());
  }

  inline iterator
  begin()
  {
    return iterator(nodes_.begin());
  }

  inline const_iterator
  end() const
  {
    return const_iterator(nodes_.end());
  }

  inline iterator
  end()
  {
    return iterator(nodes_.end());
  }

  EntryNode *
  entry() const noexcept
  {
    return entry_.get();
  }

  ExitNode *
  exit() const noexcept
  {
    return exit_.get();
  }

  BasicBlock *
  add_node(std::unique_ptr<BasicBlock> bb);

  iterator
  find_node(BasicBlock * bb);

  static ControlFlowGraph::iterator
  remove_node(ControlFlowGraph::iterator & it);

  static ControlFlowGraph::iterator
  remove_node(BasicBlock * bb);

  inline size_t
  nnodes() const noexcept
  {
    return nodes_.size();
  }

  [[nodiscard]] InterProceduralGraphModule &
  module() const noexcept
  {
    return module_;
  }

  rvsdg::FunctionType
  fcttype() const
  {
    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> arguments;
    for (size_t n = 0; n < entry()->narguments(); n++)
      arguments.push_back(entry()->argument(n)->Type());

    std::vector<std::shared_ptr<const jlm::rvsdg::Type>> results;
    for (size_t n = 0; n < exit()->nresults(); n++)
      results.push_back(exit()->result(n)->Type());

    return rvsdg::FunctionType(arguments, results);
  }

  static std::unique_ptr<ControlFlowGraph>
  create(InterProceduralGraphModule & im);

  static std::string
  ToAscii(const ControlFlowGraph & controlFlowGraph);

  /**
   * Converts the control flow graph to a DOT graph.
   *
   * @param writer A graph writer for converting the \p controlFlowGraph to DOT.
   * @param controlFlowGraph Control flow graph that is converted.
   * @return A DOT graph
   */
  static util::graph::Graph &
  toDot(util::graph::Writer & writer, const ControlFlowGraph & controlFlowGraph);

  /**
   * This function is meant to be used from the debugger. You can just
   * invoke it and a xdot window should pop up with a DOT visualization of the control flow graph.
   * This depends on xdot being in the PATH.
   */
  void
  view() const;

private:
  static std::string
  ToAscii(const EntryNode & entryNode);

  static std::string
  ToAscii(const ExitNode & exitNode);

  static std::string
  ToAscii(
      const BasicBlock & basicBlock,
      const std::unordered_map<ControlFlowGraphNode *, std::string> & labels);

  static std::string
  CreateTargets(
      const ControlFlowGraphNode & node,
      const std::unordered_map<ControlFlowGraphNode *, std::string> & labels);

  static std::unordered_map<ControlFlowGraphNode *, std::string>
  CreateLabels(const std::vector<ControlFlowGraphNode *> & nodes);

  InterProceduralGraphModule & module_;
  std::unique_ptr<ExitNode> exit_;
  std::unique_ptr<EntryNode> entry_;
  std::unordered_set<std::unique_ptr<BasicBlock>> nodes_;
};

std::vector<ControlFlowGraphNode *>
postorder(const ControlFlowGraph & cfg);

std::vector<ControlFlowGraphNode *>
reverse_postorder(const ControlFlowGraph & cfg);

/** Order CFG nodes breadth-first
 *
 * Note, all nodes that are not dominated by the entry node are ignored.
 *
 * param cfg Control flow graph
 *
 * return A vector with all CFG nodes ordered breadth-first
 */
std::vector<ControlFlowGraphNode *>
breadth_first(const ControlFlowGraph & cfg);

size_t
ntacs(const ControlFlowGraph & cfg);

/** \brief Strongly Connected Component
 */
class StronglyConnectedComponent final
{
  using constiterator = util::
      PtrIterator<ControlFlowGraphNode, std::unordered_set<ControlFlowGraphNode *>::const_iterator>;

public:
  explicit StronglyConnectedComponent(const std::unordered_set<ControlFlowGraphNode *> & nodes)
      : nodes_(nodes)
  {}

  constiterator
  begin() const;

  constiterator
  end() const;

  bool
  contains(ControlFlowGraphNode * node) const
  {
    return nodes_.find(node) != nodes_.end();
  }

  size_t
  nnodes() const noexcept
  {
    return nodes_.size();
  }

private:
  std::unordered_set<ControlFlowGraphNode *> nodes_{};
};

/** \brief Strongly Connected Component Structure
 *
 * This class computes the structure of strongly connected components (SCCs). It detects the
 * following entities:
 *
 * 1. Entry edges: All edges from a node outside the SCC pointing to a node in the SCC.
 * 2. Entry nodes: All nodes that are the target of one or more entry edges.
 * 3. Exit edges: All edges from a node inside the SCC pointing to a node outside the SCC.
 * 4. Exit nodes: All nodes that are the target of one or more exit edges.
 * 5. Repetition edges: All edges from a node inside the SCC to an entry node.
 */
class StronglyConnectedComponentStructure final
{
  using CfgEdgeConstIterator = util::HashSet<ControlFlowGraphEdge *>::ItemConstIterator;
  using CfgNodeConstIterator = util::HashSet<ControlFlowGraphNode *>::ItemConstIterator;

  using EdgeIteratorRange = util::IteratorRange<CfgEdgeConstIterator>;
  using NodeIteratorRange = util::IteratorRange<CfgNodeConstIterator>;

public:
  size_t
  NumEntryNodes() const noexcept
  {
    return EntryNodes_.Size();
  }

  size_t
  NumExitNodes() const noexcept
  {
    return ExitNodes_.Size();
  }

  size_t
  NumEntryEdges() const noexcept
  {
    return EntryEdges_.Size();
  }

  size_t
  NumRepetitionEdges() const noexcept
  {
    return RepetitionEdges_.Size();
  }

  size_t
  NumExitEdges() const noexcept
  {
    return ExitEdges_.Size();
  }

  NodeIteratorRange
  EntryNodes() const
  {
    return { EntryNodes_.Items().begin(), EntryNodes_.Items().end() };
  }

  NodeIteratorRange
  ExitNodes() const
  {
    return { ExitNodes_.Items().begin(), ExitNodes_.Items().end() };
  }

  EdgeIteratorRange
  EntryEdges() const
  {
    return { EntryEdges_.Items().begin(), EntryEdges_.Items().end() };
  }

  EdgeIteratorRange
  RepetitionEdges() const
  {
    return { RepetitionEdges_.Items().begin(), RepetitionEdges_.Items().end() };
  }

  EdgeIteratorRange
  ExitEdges() const
  {
    return { ExitEdges_.Items().begin(), ExitEdges_.Items().end() };
  }

  /**
   * Creates an SCC structure from \p scc.
   */
  static std::unique_ptr<StronglyConnectedComponentStructure>
  Create(const StronglyConnectedComponent & scc);

  /**
   * Checks if the SCC structure is a tail-controlled loop. A tail-controlled loop is defined as an
   * SSC with a single entry node, as well as a single repetition and exit edge. Both these edges
   * must have the same CFG node as origin.
   */
  bool
  IsTailControlledLoop() const noexcept;

private:
  util::HashSet<ControlFlowGraphNode *> EntryNodes_{};
  util::HashSet<ControlFlowGraphNode *> ExitNodes_{};
  util::HashSet<ControlFlowGraphEdge *> EntryEdges_{};
  util::HashSet<ControlFlowGraphEdge *> RepetitionEdges_{};
  util::HashSet<ControlFlowGraphEdge *> ExitEdges_{};
};

bool
is_valid(const ControlFlowGraph & cfg);

bool
is_closed(const ControlFlowGraph & cfg);

bool
is_linear(const ControlFlowGraph & cfg);

/**
 * Compute a Control Flow Graph's Strongly Connected Components.
 */
std::vector<StronglyConnectedComponent>
find_sccs(const ControlFlowGraph & cfg);

/**
 * Compute all Strongly Connected Components of a single-entry/single-exit region.
 * The \p entry parameter must dominate the \p exit parameter.
 */
std::vector<StronglyConnectedComponent>
find_sccs(ControlFlowGraphNode * entry, ControlFlowGraphNode * exit);

static inline bool
is_acyclic(const ControlFlowGraph & cfg)
{
  auto sccs = find_sccs(cfg);
  return sccs.size() == 0;
}

bool
is_structured(const ControlFlowGraph & cfg);

bool
is_proper_structured(const ControlFlowGraph & cfg);

bool
is_reducible(const ControlFlowGraph & cfg);

/**
 * Finds all pairs of basic blocks A, B where the edge
 *  A -> B
 * is A's only out-edge, and B's only in-edge.
 *
 * B may not have any Phi operations.
 *
 * For each such pair, A and B are merged into a single basic block.
 *
 * @param cfg the control flow graph for a function
 */
void
straighten(ControlFlowGraph & cfg);

/** \brief Remove all basic blocks without instructions
 */
void
purge(ControlFlowGraph & cfg);

/**
 * Removes unreachable nodes from the control flow graph.
 * @param cfg the control flow graph of a function
 */
void
prune(ControlFlowGraph & cfg);

}

#endif
