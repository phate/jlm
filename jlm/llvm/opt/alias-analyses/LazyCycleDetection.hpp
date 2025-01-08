/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_LAZYCYCLEDETECTION_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_LAZYCYCLEDETECTION_HPP

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/util/HashSet.hpp>

#include <limits>
#include <optional>
#include <stack>
#include <vector>

namespace jlm::llvm::aa
{

/**
 * Implements Lazy Cycle Detection, as described by
 *   Hardekopf and Lin, 2007: "The And and the Grasshopper"
 * @tparam GetSuccessorsFunctor is a function returning the superset edge successors of a given node
 * @tparam UnifyPointerObjectsFunctor the functor to be called to unify a cycle, when found
 */
template<typename GetSuccessorsFunctor, typename UnifyPointerObjectsFunctor>
class LazyCycleDetector
{

public:
  LazyCycleDetector(
      PointerObjectSet & set,
      const GetSuccessorsFunctor & GetSuccessors,
      const UnifyPointerObjectsFunctor & unifyPointerObjects)
      : Set_(set),
        GetSuccessors_(GetSuccessors),
        UnifyPointerObjects_(unifyPointerObjects)
  {}

  /**
   * Call before calling any other method
   */
  void
  Initialize()
  {
    NodeStates_.resize(Set_.NumPointerObjects(), NodeStateNotVisited);
  }

  bool
  IsInitialized() const noexcept
  {
    return NodeStates_.size() == Set_.NumPointerObjects();
  }

  /**
   * Call when an edge subset -> superset was visited, and zero pointees had to be propagated.
   * Only call if subset has at least one new pointee.
   * If a path from superset to subset is found, there is a cycle, that gets unified.
   * @param subset the tail of the added edge, must be unification root
   * @param superset the head of the added edge, must be unification root
   * @return the root of the unification if unification happened, otherwise nullopt
   */
  std::optional<PointerObjectIndex>
  OnPropagatedNothing(PointerObjectIndex subset, PointerObjectIndex superset)
  {
    JLM_ASSERT(IsInitialized());
    JLM_ASSERT(Set_.IsUnificationRoot(subset));
    JLM_ASSERT(Set_.IsUnificationRoot(superset));

    // Add this edge to the list of checked edges, or return if it was already there
    if (!CheckedEdges_.Insert({ subset, superset }))
      return std::nullopt;

    NumCycleDetectAttempts_++;

    JLM_ASSERT(DfsStack_.empty());
    DfsStack_.push(superset);

    // Reset all node states
    std::fill(NodeStates_.begin(), NodeStates_.end(), NodeStateNotVisited);

    while (!DfsStack_.empty())
    {
      auto node = DfsStack_.top();
      if (NodeStates_[node] == NodeStateNotVisited)
      {
        NodeStates_[node] = NodeStateVisited;
        // Make sure all successors get visited
        for (auto successor : GetSuccessors_(node))
        {
          auto successorRoot = Set_.GetUnificationRoot(successor);

          // Cycle found! Do not add the subset to the dfs stack
          if (successorRoot == subset)
            continue;

          if (NodeStates_[successorRoot] != NodeStateNotVisited)
            continue;

          DfsStack_.push(successorRoot);
        }
      }
      else if (NodeStates_[node] == NodeStateVisited)
      {
        DfsStack_.pop();
        NodeStates_[node] = NodeStatePopped;

        // Check if any successors are unified with the subset. If so, join it!
        for (auto successor : GetSuccessors_(node))
        {
          auto successorRoot = Set_.GetUnificationRoot(successor);
          if (successorRoot == subset)
          {
            subset = UnifyPointerObjects_(node, subset);
            NumCycleUnifications_++;
            break;
          }
        }
      }
      else
      {
        // The node has already been visited for a second time
        DfsStack_.pop();
      }
    }

    JLM_ASSERT(Set_.IsUnificationRoot(subset));
    superset = Set_.GetUnificationRoot(superset);
    if (subset == superset)
    {
      NumCyclesDetected_++;
      return subset;
    }
    return std::nullopt;
  }

  /**
   * @return the number of DFSs performed to look for cycles
   */
  [[nodiscard]] size_t
  NumCycleDetectionAttempts() const noexcept
  {
    return NumCycleDetectAttempts_;
  }

  /**
   * @return the number of cycles detected by Lazy cycle detection
   */
  [[nodiscard]] size_t
  NumCyclesDetected() const noexcept
  {
    return NumCyclesDetected_;
  }

  /**
   * @return the number of unifications made while eliminating found cycles
   */
  [[nodiscard]] size_t
  NumCycleUnifications() const noexcept
  {
    return NumCycleUnifications_;
  }

private:
  PointerObjectSet & Set_;
  const GetSuccessorsFunctor & GetSuccessors_;
  const UnifyPointerObjectsFunctor & UnifyPointerObjects_;

  // A set of all checked simple edges first -> second, to avoid checking again
  util::HashSet<std::pair<PointerObjectIndex, PointerObjectIndex>> CheckedEdges_;

  // The dfs stack, which may contain the same node multiple times
  std::stack<PointerObjectIndex> DfsStack_;
  // Possible states of nodes during the DFS
  static constexpr uint8_t NodeStateNotVisited = 0;
  static constexpr uint8_t NodeStateVisited = 1;
  static constexpr uint8_t NodeStatePopped = 2;
  std::vector<uint8_t> NodeStates_;

  size_t NumCycleDetectAttempts_ = 0;
  size_t NumCyclesDetected_ = 0;
  size_t NumCycleUnifications_ = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_LAZYCYCLEDETECTION_HPP
