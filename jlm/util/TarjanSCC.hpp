/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_TARJANSCC_HPP
#define JLM_UTIL_TARJANSCC_HPP

#include <jlm/util/common.hpp>

#include <cstddef>
#include <cstdint>
#include <stack>
#include <vector>

namespace jlm::util
{

/**
 * Implementation of Tarjan's algorithm for finding strongly connected components in linear time.
 * Each node is assigned an SCC, with scc indices starting at 0.
 * An SCC has a lower index than all its predecessor SCCs, and a larger index than its successors.
 *
 * In addition to assigning SCCs, a partial topological ordering of nodes is returned.
 * All nodes belonging to a given SCC are in a continuous subsequence.
 * All nodes belonging to predecessor SCCs come earlier in the list,
 * and all nodes belonging to successor SCCs come later.
 *
 * @tparam SuccessorFunctor a functor with the signature (size_t) -> iterable<size_t>
 * @param numNodes the number of nodes. Nodes are indexed from 0 to numNodes-1
 * @param successors an instance of the SucessorFunctor
 * @param sccIndex output vector to be filled with the index of the SCC each node ends up in
 * @param topologicalOrder output vector to be filled with the nodes in a weak topological order
 * @return the number of SCCs in the graph. One more than the largest SCC index
 */
template<typename SuccessorFunctor>
size_t
FindStronglyConnectedComponents(
    size_t numNodes,
    SuccessorFunctor & successors,
    std::vector<size_t> & sccIndex,
    std::vector<size_t> & topologicalOrder)
{
  size_t sccsFinished = 0;
  // The number of nodes that have been assigned to a finished SCC
  size_t nodesFinished = 0;

  // What SCC each node ends up in
  sccIndex.resize(numNodes);
  // Partially ordered topological order
  topologicalOrder.resize(numNodes);

  // Non-recursive implementation of Tarjan's SCC
  const int64_t NOT_VISITED = -1;
  const int64_t SCC_FINISHED = -2;
  std::vector<int64_t> order(numNodes, NOT_VISITED);
  int64_t nextOrder = 0;
  // The lowest order seen through any path of edges that do not enter finished SCCs
  std::vector<int64_t> lowLink(numNodes);

  // The DFS stack is a regular DFS traversal stack
  std::stack<size_t> dfsStack;
  // The SCC stack is only popped once an SCC is found
  std::stack<size_t> sccStack;

  // Find SCCs
  for (size_t startNode = 0; startNode < numNodes; startNode++)
  {
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
          // Visit nodes that have not been visited before
          if (order[next] == NOT_VISITED)
            dfsStack.push(next);
        }
      }
      else // This is the second time node is visited, all children have been processed
      {
        dfsStack.pop();
        for (auto next : successors(node))
        {
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
            topologicalOrder[numNodes - 1 - nodesFinished] = top;
            nodesFinished++;

            if (top == node)
              break;
          }

          sccsFinished++;
        }
      }
    }
  }
  JLM_ASSERT(nodesFinished == numNodes);
  JLM_ASSERT(sccStack.empty());

  return sccsFinished;
}

}

#endif // JLM_UTIL_TARJANSCC_HPP
