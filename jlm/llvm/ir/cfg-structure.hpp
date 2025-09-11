/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_CFG_STRUCTURE_HPP
#define JLM_LLVM_IR_CFG_STRUCTURE_HPP

#include <jlm/util/HashSet.hpp>
#include <jlm/util/iterator_range.hpp>
#include <jlm/util/IteratorWrapper.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

namespace jlm::llvm
{

class ControlFlowGraph;
class ControlFlowGraphEdge;
class ControlFlowGraphNode;

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
