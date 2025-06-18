/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_TARJANSCC_HPP
#define JLM_UTIL_TARJANSCC_HPP

#include <jlm/util/common.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stack>
#include <vector>

namespace jlm::util
{

/**
 * Implementation of Tarjan's algorithm for finding strongly connected components in linear time.
 * Each node is assigned a strongly connected component (SCC), with SCC indices starting at 0.
 * SCC indices are assigned such that, for each edge A -> B in the original graph,
 * sccIndex[A] >= sccIndex[B]. They are equal iff A and B belong to the same SCC, i.e.
 * the original graph contains a path from A to B, and from B to A.
 *
 * In addition to assigning SCCs, a partial reverse topological ordering of nodes is returned.
 * The ordering is a list of all root nodes, ordered by non-descending SCC index.
 * Within a single SCC, the ordering of nodes is arbitrary.
 *
 * This function also supports graphs where nodes have been unified.
 * Within such a unification, one node is the root, while all other nodes are aliases for the root.
 * The given \p unificationRoot function should return the root for any node.
 * Only root nodes will be queried about their successors.
 * Nodes that are not roots will not be given an sccIndex, and not be included in topological order.
 *
 * @tparam NodeType the integer type used to index nodes
 * @tparam UnificationRootFunctor a functor with the signature (NodeType) -> NodeType
 * @tparam SuccessorFunctor a functor with the signature (NodeType) -> iterable<NodeType>
 * @param numNodes the total number of nodes, including non-root nodes in unifications.
 *        Nodes are indexed from 0 to numNodes-1
 * @param unificationRoot an instance of the UnificationRootFunctor
 * @param successors an instance of the SuccessorFunctor, returning outgoing edges from a root node.
 * @param sccIndex output vector to be filled with the index of the SCC each node ends up in.
 *        Only nodes that are roots will be given an sccIndex.
 * @param reverseTopologicalOrder output vector filled with root nodes in reverse topological order.
 *        In other words, a list of all root nodes, ordered with non-descending sccIndex.
 * @return the number of SCCs in the graph. One more than the largest SCC index
 */
template<typename NodeType, typename UnificationRootFunctor, typename SuccessorFunctor>
NodeType
FindStronglyConnectedComponents(
    NodeType numNodes,
    UnificationRootFunctor unificationRoot,
    SuccessorFunctor & successors,
    std::vector<NodeType> & sccIndex,
    std::vector<NodeType> & reverseTopologicalOrder)
{
  NodeType sccsFinished = 0;

  // What SCC each node ends up in
  sccIndex.resize(numNodes);
  // Partially ordered topological order
  reverseTopologicalOrder.resize(0);

  // Use the max values of the NodeType as sentinel values
  const NodeType NOT_VISITED = std::numeric_limits<NodeType>::max();
  const NodeType SCC_FINISHED = NOT_VISITED - 1;
  JLM_ASSERT(numNodes <= SCC_FINISHED);

  // Non-recursive implementation of Tarjan's SCC
  std::vector<NodeType> order(numNodes, NOT_VISITED);
  NodeType nextOrder = 0;
  // The lowest order seen through any path of edges that do not enter finished SCCs
  std::vector<NodeType> lowLink(numNodes);

  // The DFS stack is a regular DFS traversal stack.
  // Note that the same element can be pushed many times, but will only be visited twice
  std::stack<NodeType> dfsStack;
  // The SCC stack is only popped once an SCC is found
  std::stack<NodeType> sccStack;

  // Find SCCs
  for (NodeType startNode = 0; startNode < numNodes; startNode++)
  {
    // Skip nodes that are not roots
    if (unificationRoot(startNode) != startNode)
      continue;

    // Only start DFSs at unvisited nodes
    if (order[startNode] != NOT_VISITED)
      continue;

    dfsStack.push(startNode);
    while (!dfsStack.empty())
    {
      auto node = dfsStack.top();

      if (order[node] == NOT_VISITED) // This is the first time node is visited
      {
        sccStack.push(node);
        order[node] = nextOrder++;
        lowLink[node] = order[node];

        for (auto next : successors(node))
        {
          next = unificationRoot(next);

          // Visit nodes that have not been visited before
          if (order[next] == NOT_VISITED)
            dfsStack.push(next);
        }
      }
      else if (order[node] == SCC_FINISHED) // This node has already been fully processed
      {
        dfsStack.pop();
      }
      else // This node is being visited for the second time, i.e., the dfs post-visit
      {
        dfsStack.pop();

        for (auto next : successors(node))
        {
          next = unificationRoot(next);

          // Ignore edges to nodes that are already part of a finished SCC
          if (order[next] == SCC_FINISHED)
            continue;
          lowLink[node] = std::min(lowLink[node], lowLink[next]);
        }

        // This node is the root of an SCC
        if (lowLink[node] == order[node])
        {
          const auto thisSccIndex = sccsFinished;

          while (true)
          {
            auto top = sccStack.top();
            sccStack.pop();
            order[top] = SCC_FINISHED;
            sccIndex[top] = thisSccIndex;
            reverseTopologicalOrder.push_back(top);

            if (top == node)
              break;
          }

          sccsFinished++;
        }
      }
    }
  }
  JLM_ASSERT(sccStack.empty());

  return sccsFinished;
}

/**
 * Implementation of Tarjan's algorithm for finding strongly connected components in linear time.
 * This overload is for use cases where node unification is not used (every node is its own root).
 * This means every node will be given an sccIndex,
 * and all nodes will be included in the reverse topological order.
 *
 * @see FindStronglyConnectedComponents for details.
 *
 * @tparam NodeType the integer type used to index nodes
 * @tparam SuccessorFunctor a functor with the signature (NodeType) -> iterable<NodeType>
 * @param sccIndex output vector to be filled with the index of the SCC each node ends up in.
 * @param reverseTopologicalOrder output vector filled with nodes in reverse topological order.
 * @return the number of SCCs in the graph. One more than the largest SCC index
 */
template<typename NodeType, typename SuccessorFunctor>
NodeType
FindStronglyConnectedComponents(
    NodeType numNodes,
    SuccessorFunctor & successors,
    std::vector<NodeType> & sccIndex,
    std::vector<NodeType> & reverseTopologicalOrder)
{
  const auto identity = [](NodeType n)
  {
    return n;
  };
  return FindStronglyConnectedComponents(
      numNodes,
      identity,
      successors,
      sccIndex,
      reverseTopologicalOrder);
}

}

#endif // JLM_UTIL_TARJANSCC_HPP
