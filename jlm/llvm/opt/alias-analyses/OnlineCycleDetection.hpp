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
 * @tparam GetSuccessorsFunctor
 * @tparam UnifyPointerObjectsFunctor
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
   * A assigns each unification root a position in a topological ordering.
   * If any cycles are detected, they are unified, but nothing is placed on the worklist.
   * This function assumes that GetSuccessors returns only unification roots.
   */
  void
  InitializeTopologicalOrdering()
  {
    // Initialize Online Cycle Detection using a full topological sort of the nodes
    ObjectToTopoOrder_.resize(Set_.NumPointerObjects(), -1);

    // At this point, all subsetEdges are between unification roots only
    // Use TarjanSCC to find our starting topological ordering
    std::vector<size_t> sccIndex;
    std::vector<size_t> topologicalOrder;

    util::FindStronglyConnectedComponents(
        Set_.NumPointerObjects(),
        GetSuccessors_,
        sccIndex,
        topologicalOrder);

    // Go through the topological ordering and add all unification roots to OCD_ObjectToTopoOrder
    // Also, if we find any new SCCs while doing this, perform unification right now
    for (size_t i = 0; i < topologicalOrder.size(); i++)
    {
      PointerObjectIndex node = topologicalOrder[i];
      if (!Set_.IsUnificationRoot(node))
        continue; // We only care about unification roots

      // If this root belongs to the same scc as the next root, they should be unified
      if (i + 1 < topologicalOrder.size() && sccIndex[node] == sccIndex[topologicalOrder[i + 1]])
      {
        // We know that the SCC consists of only roots, since GetSuccessors_ returns only roots
        JLM_ASSERT(Set_.IsUnificationRoot(topologicalOrder[i + 1]));
        // Make the next object the root, to keep the unification going
        topologicalOrder[i + 1] = UnifyPointerObjects_(node, topologicalOrder[i + 1]);
      }
      else
      {
        // Add this root to the next index in the topological order
        ObjectToTopoOrder_[node] = TopoOrderToObject_.size();
        TopoOrderToObject_.push_back(node);
      }
    }
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
    JLM_ASSERT(ObjectToTopoOrder_[subset] != -1 && ObjectToTopoOrder_[superset] != -1);

    // If adding the simple edge subset -> superset does not break the invariant, return
    if (ObjectToTopoOrder_[subset] <= ObjectToTopoOrder_[superset])
      return std::nullopt;

    // Otherwise, we need to do reordering in this range to fix the invariant
    const auto lowerTopo = ObjectToTopoOrder_[superset];
    const auto upperTopo = ObjectToTopoOrder_[subset];

    // Perform DFS to find all roots reachable from superset
    // We only care about nodes where with its topo index <= upperTopo
    // These nodes will all need to be moved after subset in the new topological order
    JLM_ASSERT(DfsStack_.empty());
    // All nodes that have been pushed to the dfs stack
    DfsNodesVisited_.Clear();
    // All nodes that could reach subset and thus are part of a subset -> superset -> subset cycle
    DfsCycleNodes_.Clear();

    // Start DFS from the superset
    DfsStack_.emplace(superset, false);

    while (!DfsStack_.empty())
    {
      const auto node = DfsStack_.top().first;
      const bool firstVisit = !DfsStack_.top().second;
      DfsStack_.pop();
      JLM_ASSERT(Set_.IsUnificationRoot(node));

      if (firstVisit)
      {
        // If this node has already been visited, skip it
        if (!DfsNodesVisited_.Insert(node))
          continue;

        // Push node again to visit it a second time on the way back
        DfsStack_.emplace(node, true);

        // Push all non-pushed successors with topo within [lowerTopo, upperTopo]
        for (auto successor : GetSuccessors_(node))
        {
          auto successorParent = Set_.GetUnificationRoot(successor);

          // Only push nodes that have yet to be visited
          if (DfsNodesVisited_.Contains(successorParent))
            continue;

          // Ensure that the topological order invariants are satisfied
          JLM_ASSERT(ObjectToTopoOrder_[successorParent] != -1);
          JLM_ASSERT(ObjectToTopoOrder_[successorParent] > ObjectToTopoOrder_[node]);

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
        // If this node is the subset, there is a cycle
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

    // This list is used to store all roots that should come after the subset in the new ordering
    DfsSupersetAndBeyond_.clear();

    // The root of the unified cycle, if any
    std::optional<PointerObjectIndex> optUnificationRoot = std::nullopt;
    if (!DfsCycleNodes_.IsEmpty())
    {
      // If there is a cycle, both the subset and superset must be included
      JLM_ASSERT(DfsCycleNodes_.Contains(subset));
      JLM_ASSERT(DfsCycleNodes_.Contains(superset));

      // Merge all entries on the merge list
      for (const auto node : DfsCycleNodes_.Items())
      {
        // Remove node from the topological order, only the final merge result belongs there
        ObjectToTopoOrder_[node] = -1;
        if (optUnificationRoot)
          optUnificationRoot = UnifyPointerObjects_(*optUnificationRoot, node);
        else
          optUnificationRoot = node;
      }
      // After the nodes not found during the dfs, the merged node should be next, topologically
      DfsSupersetAndBeyond_.push_back(*optUnificationRoot);
    }

    // Now go through the topological order from lowerTopo to upperTopo.
    // All nodes not visited by the DFS are compacted to the left,
    // All other nodes are added in order in the supersetAndBeyond list
    // Nodes that are part of the cycle, if any, are ignored
    int64_t nextTopologicalIndex = lowerTopo;
    for (int64_t i = lowerTopo; i <= upperTopo; i++)
    {
      // Skip all topological indices that do not contain a node
      if (TopoOrderToObject_[i] == -1)
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
      TopoOrderToObject_[nextTopologicalIndex] = -1;
      nextTopologicalIndex++;
    }

#ifdef JLM_ENABLE_ASSERTS
    ValidateTopologicalOrder();
#endif

    // return the root of the unification if a cycle was detected, otherwise nullopt
    return optUnificationRoot;
  };

  /**
   * Function performing asserts to ensure all invariants are maintained.
   * Nodes should have a position in the topological order iff they are unification roots.
   * All successor roots should come later in the topological order.
   * The bi-directional lookup, node <-> topological index, is maintained.
   */
  void
  ValidateTopologicalOrder()
  {
    for (PointerObjectIndex node = 0; node < Set_.NumPointerObjects(); node++)
    {
      // Non-unification roots should not have a place in the topological order
      if (!Set_.IsUnificationRoot(node))
      {
        JLM_ASSERT(ObjectToTopoOrder_[node] == -1);
        continue;
      }

      // Ensure we have a position in the topological order
      const auto topo = ObjectToTopoOrder_[node];
      JLM_ASSERT(topo != -1 && TopoOrderToObject_[topo] == node);

      // Ensure all outgoing edges go to unifications with higher topological index
      for (auto successor : GetSuccessors_(node))
      {
        const auto successorRoot = Set_.GetUnificationRoot(successor);
        if (successorRoot == node) // Ignore self-edges
          continue;
        JLM_ASSERT(ObjectToTopoOrder_[successorRoot] != -1);
        JLM_ASSERT(ObjectToTopoOrder_[successorRoot] > topo);
      }
    }

    // Ensure all back-references are correct
    for (size_t i = 0; i < TopoOrderToObject_.size(); i++)
    {
      if (TopoOrderToObject_[i] == -1)
        continue;
      size_t backReference = ObjectToTopoOrder_[TopoOrderToObject_[i]];
      JLM_ASSERT(backReference == i);
    }
  };

private:
  // The PointerObjectSet being operated on
  PointerObjectSet & Set_;
  // Functions used to operate on the subset graph while the worklist solver is running
  const GetSuccessorsFunctor & GetSuccessors_;
  const UnifyPointerObjectsFunctor & UnifyPointerObjects_;

  // Invariant:
  // ObjectToTopoOrder_[PO] = -1 iff PO is not a unification root.
  // ObjectToTopoOrder_[PO] is smaller than all successors of PO's roots.
  std::vector<int64_t> ObjectToTopoOrder_;

  // TopoOrderToObject_[topo] = -1 iff no PointerObject has the position
  // TopoOrderToObject_ is the inverse to ObjectToTopoOrder_
  std::vector<int64_t> TopoOrderToObject_;

  // Data structures used by to store state during DFS passes are kept here,
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
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ONLINECYCLEDETECTION_HPP
