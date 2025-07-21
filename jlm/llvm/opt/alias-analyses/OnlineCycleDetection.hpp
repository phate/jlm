/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ONLINECYCLEDETECTION_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ONLINECYCLEDETECTION_HPP

#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>
#include <jlm/util/TarjanScc.hpp>

#include <stack>
#include <vector>

namespace jlm::llvm::aa
{

/**
 * Online Cycle Detection maintains a topological ordering of all unification roots, with respect
 * to superset edges. If this is not possible, there must be a cycle, which gets eliminated.
 * Online Cycle Detection can not be combined with other cycle detection schemes,
 * but that is OK, since OCD finds every cycle.
 * @tparam GetSuccessorsFunctor is a function returning the superset edge successors of a given node
 * @tparam UnifyPointerObjectsFunctor is a function that unifies other unification roots.
 */
template<typename GetSuccessorsFunctor, typename UnifyPointerObjectsFunctor>
class OnlineCycleDetector
{

public:
  OnlineCycleDetector(
      PointerObjectSet & set,
      const GetSuccessorsFunctor & GetSuccessors,
      const UnifyPointerObjectsFunctor & UnifyPointerObjects)
      : Set_(set),
        GetSuccessors_(GetSuccessors),
        UnifyPointerObjects_(UnifyPointerObjects)
  {}

  /**
   * Assigns each unification root a position in a topological ordering.
   * If any cycles are detected, they are unified.
   * This function assumes that GetSuccessors returns only unification roots.
   */
  void
  InitializeTopologicalOrdering()
  {
    // Ensure no pointer object has the sentinel value as its index
    JLM_ASSERT(Set_.NumPointerObjects() <= EmptyTopologicalSlot);

    // Initialize Online Cycle Detection using a full topological sort of the nodes
    ObjectToTopoOrder_.resize(Set_.NumPointerObjects(), EmptyTopologicalSlot);

    // At this point, all subsetEdges are between unification roots only
    // Use TarjanSCC to find our starting topological ordering
    std::vector<PointerObjectIndex> sccIndex;
    std::vector<PointerObjectIndex> reverseTopologicalOrder;

    // Used by Tarjan to avoid traversing non-roots
    const auto GetUnificationRoot = [&](PointerObjectIndex node)
    {
      return Set_.GetUnificationRoot(node);
    };

    util::FindStronglyConnectedComponents<PointerObjectIndex>(
        Set_.NumPointerObjects(),
        GetUnificationRoot,
        GetSuccessors_,
        sccIndex,
        reverseTopologicalOrder);

    // Go through the topological ordering and add all unification roots to OCD_ObjectToTopoOrder
    // Also, if we find any new SCCs while doing this, perform unification right now
    for (auto it = reverseTopologicalOrder.rbegin(); it != reverseTopologicalOrder.rend(); ++it)
    {
      JLM_ASSERT(Set_.IsUnificationRoot(*it));

      // Check if we can unify node with the next node in the topological order
      if (const auto nextIt = it + 1;
          nextIt != reverseTopologicalOrder.rend() && sccIndex[*it] == sccIndex[*nextIt])
      {
        // We know that the SCC consists of only roots
        JLM_ASSERT(Set_.IsUnificationRoot(*nextIt));
        // Make the next object the root, to keep the unification going
        *nextIt = UnifyPointerObjects_(*it, *nextIt);
      }
      else
      {
        // Add this root to the next index in the topological order
        ObjectToTopoOrder_[*it] = TopoOrderToObject_.size();
        TopoOrderToObject_.push_back(*it);
      }
    }

    JLM_ASSERT(HasValidTopologicalOrder());
  }

  /** Call this function after adding a superset edge subset -> superset.
   * Both must be unification roots.
   * If subset has a higher topological index than superset, the topological order is re-arranged.
   * If a cycle is detected, then it will be unified away, and the unification root is returned.
   * @param subset the tail of the just added superset edge
   * @param superset the head of the just added superset edge
   * @return the root of the new unification, if any, otherwise nullopt.
   */
  std::optional<PointerObjectIndex>
  MaintainTopologicalOrder(PointerObjectIndex subset, PointerObjectIndex superset)
  {
    JLM_ASSERT(Set_.IsUnificationRoot(subset) && Set_.IsUnificationRoot(superset));
    // All unification roots should have positions in the topological order
    JLM_ASSERT(ObjectIsInTopologicalOrder(subset) && ObjectIsInTopologicalOrder(superset));

    // If adding the simple edge subset -> superset does not break the invariant, return
    if (ObjectToTopoOrder_[subset] <= ObjectToTopoOrder_[superset])
      return std::nullopt;

    // Otherwise, we need to do reordering in this range to fix the invariant
    const auto lowerTopo = ObjectToTopoOrder_[superset];
    const auto upperTopo = ObjectToTopoOrder_[subset];

    // Perform DFS to find all roots reachable from superset
    // We only care about nodes where with its topo index <= upperTopo
    // These nodes will all need to be moved after subset in the new topological order
    ClearDfsState();

    // Start DFS from the superset. false means first visit
    DfsStack_.emplace(superset, false);

    while (!DfsStack_.empty())
    {
      const auto node = DfsStack_.top().first;
      const bool isFirstVisit = !DfsStack_.top().second;
      DfsStack_.pop();
      JLM_ASSERT(Set_.IsUnificationRoot(node));

      if (isFirstVisit)
      {
        // If a "first visit" of this node has already occurred, skip it.
        // This can happen if the same node is pushed multiple times before being popped.
        if (!DfsNodesVisited_.Insert(node))
          continue;

        // Push node again to visit it a second time on the way back
        DfsStack_.emplace(node, true);

        // Push all non-pushed successors, as long as their position in the topological order is
        // between [lowerTopo, upperTopo]. Nodes outside this range can stay where they are in the
        // topological order, and will never be part of a path from superset to subset.
        for (auto successor : GetSuccessors_(node))
        {
          auto successorParent = Set_.GetUnificationRoot(successor);

          // Only push nodes that have yet to be visited
          if (DfsNodesVisited_.Contains(successorParent))
            continue;

          // Ensure that the topological order invariants are satisfied
          JLM_ASSERT(ComesAfter(successorParent, node));

          // Only care about nodes within the topological range currently being rearranged
          const auto topo = ObjectToTopoOrder_[successorParent];
          if (topo < lowerTopo || topo > upperTopo)
            continue;

          // Visit the successor
          DfsStack_.emplace(successorParent, false);
        }
      }
      else
      {
        // Node is being visited for the second time, on the way back in the dfs
        // If this node is the subset, then there is a cycle
        if (node == subset)
        {
          DfsCycleNodes_.Insert(node);
          continue;
        }

        // Check if any of node's successors reach subset
        for (auto successor : GetSuccessors_(node))
        {
          auto successorParent = Set_.GetUnificationRoot(successor);
          if (DfsCycleNodes_.Contains(successorParent))
          {
            DfsCycleNodes_.Insert(node);
            break;
          }
        }
      }
    }

    // The root of the unified cycle, if any
    std::optional<PointerObjectIndex> optUnificationRoot = std::nullopt;
    if (!DfsCycleNodes_.IsEmpty())
    {
      // If there is a cycle, both the subset and superset must be included
      JLM_ASSERT(DfsCycleNodes_.Contains(subset));
      JLM_ASSERT(DfsCycleNodes_.Contains(superset));

      // Track this cycle in the statistics
      NumOnlineCyclesDetected_++;
      NumOnlineCycleUnifications_ += DfsCycleNodes_.Size() - 1;

      // Merge all entries on the merge list
      for (const auto node : DfsCycleNodes_.Items())
      {
        // Remove node from the topological order, only the final merge result belongs there
        ObjectToTopoOrder_[node] = EmptyTopologicalSlot;
        optUnificationRoot =
            optUnificationRoot ? UnifyPointerObjects_(*optUnificationRoot, node) : node;
      }
      // After the nodes not found during the dfs, the merged node should be next, topologically
      DfsSupersetAndBeyond_.push_back(*optUnificationRoot);
    }

    // Now go through the topological order from lowerTopo to upperTopo.
    // All nodes not visited by the DFS are compacted to the left,
    // All other nodes are added in order in the supersetAndBeyond list
    // Nodes that are part of the cycle, if any, are ignored
    auto nextTopologicalIndex = lowerTopo;
    for (auto i = lowerTopo; i <= upperTopo; i++)
    {
      // Skip all topological indices that do not contain a node
      if (!IsTopologicalOrderIndexFilled(i))
        continue;

      PointerObjectIndex node = TopoOrderToObject_[i];
      if (DfsCycleNodes_.Contains(node))
        continue;

      JLM_ASSERT(ObjectToTopoOrder_[node] == i);

      if (DfsNodesVisited_.Contains(node))
      {
        DfsSupersetAndBeyond_.push_back(node);
        continue;
      }

      // Add node to its new position
      ObjectToTopoOrder_[node] = nextTopologicalIndex;
      TopoOrderToObject_[nextTopologicalIndex] = node;
      nextTopologicalIndex++;
    }

    // Next, add all elements from OCD_supersetAndBeyond
    for (auto node : DfsSupersetAndBeyond_)
    {
      JLM_ASSERT(Set_.IsUnificationRoot(node));

      ObjectToTopoOrder_[node] = nextTopologicalIndex;
      TopoOrderToObject_[nextTopologicalIndex] = node;
      nextTopologicalIndex++;
    }

    // Any leftover positions in the topological order should now become unoccupied
    while (nextTopologicalIndex <= upperTopo)
    {
      TopoOrderToObject_[nextTopologicalIndex] = EmptyTopologicalSlot;
      nextTopologicalIndex++;
    }

    JLM_ASSERT(HasValidTopologicalOrder());

    // return the root of the unification if a cycle was detected, otherwise nullopt
    return optUnificationRoot;
  };

  /**
   * @return how many cycles have been detected and eliminated by OCD
   */
  [[nodiscard]] size_t
  NumOnlineCyclesDetected() const noexcept
  {
    return NumOnlineCyclesDetected_;
  }

  /**
   * @return how many pairwise unifications have been made while eliminating cycles
   */
  [[nodiscard]] size_t
  NumOnlineCycleUnifications() const noexcept
  {
    return NumOnlineCycleUnifications_;
  }

private:
  /**
   * Function for validating that all invariants of the topological ordering are maintained.
   * Nodes should have a position in the topological order iff they are unification roots.
   * All successor roots should come later in the topological order.
   * The bi-directional lookup, node <-> topological index, is maintained.
   * @return true if all invariants are satisfied, false otherwise
   */
  [[nodiscard]] bool
  HasValidTopologicalOrder() const
  {
    for (PointerObjectIndex node = 0; node < Set_.NumPointerObjects(); node++)
    {
      // Non-unification roots should not have a place in the topological order
      if (!Set_.IsUnificationRoot(node))
      {
        if (ObjectIsInTopologicalOrder(node))
          return false;
        continue;
      }

      // Ensure we have a position in the topological order
      if (!ObjectIsInTopologicalOrder(node))
        return false;
      const auto topo = ObjectToTopoOrder_[node];
      if (TopoOrderToObject_[topo] != node)
        return false;

      // Ensure all outgoing edges go to unifications with higher topological index
      for (auto successor : GetSuccessors_(node))
      {
        const auto successorRoot = Set_.GetUnificationRoot(successor);
        if (successorRoot == node) // Ignore self-edges
          continue;

        if (!ComesAfter(successorRoot, node))
          return false;
      }
    }

    // Ensure TopoOrderToObject is a perfect inverse of ObjectToTopoOrder
    for (size_t i = 0; i < TopoOrderToObject_.size(); i++)
    {
      if (!IsTopologicalOrderIndexFilled(i))
        continue;
      size_t backReference = ObjectToTopoOrder_[TopoOrderToObject_[i]];
      if (backReference != i)
        return false;
    }

    return true;
  };

  /**
   * The topological order contains all unification roots, and only unification roots.
   * @return true if \p object has a position in the topological order, false otherwise.
   */
  [[nodiscard]] bool
  ObjectIsInTopologicalOrder(PointerObjectIndex object) const
  {
    return ObjectToTopoOrder_[object] != EmptyTopologicalSlot;
  }

  /**
   * The topological order contains all unification roots, and only unification roots.
   * @return true if the given \p index in the topological order is occupied, false otherwise.
   */
  [[nodiscard]] bool
  IsTopologicalOrderIndexFilled(PointerObjectIndex topologicalIndex) const
  {
    return TopoOrderToObject_[topologicalIndex] != EmptyTopologicalSlot;
  }

  /**
   * @return true if the nodes \p after and \p before both have positions in the topological index,
   * and \p after comes strictly after \p before. Otherwise false.
   */
  [[nodiscard]] bool
  ComesAfter(PointerObjectIndex after, PointerObjectIndex before) const
  {
    if (!ObjectIsInTopologicalOrder(after) || !ObjectIsInTopologicalOrder(before))
      return false;
    return ObjectToTopoOrder_[after] > ObjectToTopoOrder_[before];
  }

  /**
   * Prepares all internal state for a new DFS
   */
  void
  ClearDfsState()
  {
    // As we are not currently in the middle of a DFS, the DFS stack should already be empty
    JLM_ASSERT(DfsStack_.empty());
    DfsNodesVisited_.Clear();
    DfsCycleNodes_.Clear();
    DfsSupersetAndBeyond_.clear();
  }

  // The PointerObjectSet being operated on
  PointerObjectSet & Set_;

  // Functions used to operate on the subset graph while the worklist solver is running
  const GetSuccessorsFunctor & GetSuccessors_;
  const UnifyPointerObjectsFunctor & UnifyPointerObjects_;

  // This sentinel value is used to indicate that
  //  - An object doesn't have a position in the topological order
  //  - A slot in the topological order is not occupied by an object
  static constexpr PointerObjectIndex EmptyTopologicalSlot =
      std::numeric_limits<PointerObjectIndex>::max();

  // Invariant:
  // ObjectToTopoOrder_[PO] == EmptyTopologicalSlot iff PO is not a unification root.
  // ObjectToTopoOrder_[PO] is smaller than all successors of PO's roots.
  std::vector<PointerObjectIndex> ObjectToTopoOrder_;

  // TopoOrderToObject_[topo] == EmptyTopologicalSlot iff no PointerObject has the position.
  // TopoOrderToObject_ is the inverse to ObjectToTopoOrder_
  std::vector<PointerObjectIndex> TopoOrderToObject_;

  // Data structures used to store state during DFS passes are kept here,
  // to avoid re-allocating their backing storage every time a new DFS is performed.

  // Stack used for dfs. The boolean is true iff the node is being visited on the way back.
  // Note: a node can be added to this stack many times, but only once with second=true.
  std::stack<std::pair<PointerObjectIndex, bool>> DfsStack_;

  // Nodes are added to this set when they are actually visited.
  util::HashSet<PointerObjectIndex> DfsNodesVisited_;

  // Nodes that are part of the discovered cycle (reachable from superset, can reach subset)
  // If a cycle is detected, subset and superset will both be in this list
  util::HashSet<PointerObjectIndex> DfsCycleNodes_;

  // List used to store the nodes that should come after the subset in the new topological order
  std::vector<PointerObjectIndex> DfsSupersetAndBeyond_;

  // Statistics measurements
  size_t NumOnlineCyclesDetected_ = 0;
  size_t NumOnlineCycleUnifications_ = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ONLINECYCLEDETECTION_HPP
