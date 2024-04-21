/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/TarjanSCC.hpp>

#include <cassert>
#include <cstdint>
#include <vector>

/**
 * Validation function checking that the found SCCs correspond to the \p successors:
 *  - All successor edges in the graph point to an \p sccIndex which is equal or lower
 *  - All successors are nodes in the same SCC, or a node appearing later in \p topologicalOrder
 */
template<typename SuccessorFunctor>
static void
ValidateTopologicalOrderAndSccIndices(
    size_t numNodes,
    SuccessorFunctor & successors,
    std::vector<size_t> & sccIndex,
    std::vector<size_t> & topologicalOrder)
{
  assert(numNodes == topologicalOrder.size());
  assert(numNodes == sccIndex.size());

  // Lookup of a node's position in the topological order
  std::vector<size_t> topologicalPosition(numNodes);
  for (size_t i = 0; i < numNodes; i++)
    topologicalPosition[topologicalOrder[i]] = i;

  for (size_t i = 0; i < numNodes; i++)
  {
    for (auto next : successors(i))
    {
      // Intra-scc edges are ignored
      if (sccIndex[i] == sccIndex[next])
        continue;

      // successor SCCs must have lower scc index
      assert(sccIndex[next] < sccIndex[i]);
      // the successor node must come after this node in the topological order
      assert(topologicalPosition[i] < topologicalPosition[next]);
    }
  }
}

static void
TestDAG()
{
  // Create a DAG, where each node is its own SCC
  const size_t numNodes = 8;
  std::vector<std::vector<size_t>> successors{
    { 2 },    // 0's successors
    { 2 },    // 1's successors
    { 3, 4 }, // 2's successors
    { 5, 6 }, // 3's successors
    { 5 },    // 4's successors
    { 6 },    // 5's successors
    {},       // 6's successors
    {},       // 7's successors (fully disconnected)
  };

  auto GetSuccessors = [&](size_t node)
  {
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> topologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      GetSuccessors,
      sccIndex,
      topologicalOrder);
  ValidateTopologicalOrderAndSccIndices(numNodes, GetSuccessors, sccIndex, topologicalOrder);

  assert(numSccs == numNodes);
}

// Test a graph with some cycles, ensuring they become SCCs
static void
TestCycles()
{
  const size_t numNodes = 7;
  std::vector<std::vector<size_t>> successors{
    { 1, 3 },    // 0's successors
    { 2 },       // 1's successors
    { 0 },       // 2's successors
    { 4, 5 },    // 3's successors
    { 0, 5 },    // 4's successors
    {},          // 5's successors
    { 2, 4, 5 }, // 6's successors
  };

  auto GetSuccessors = [&](size_t node)
  {
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> topologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      GetSuccessors,
      sccIndex,
      topologicalOrder);
  ValidateTopologicalOrderAndSccIndices(numNodes, GetSuccessors, sccIndex, topologicalOrder);

  assert(numSccs == 3);
  // 5 has to be at the end
  assert(sccIndex[5] == 0);
  assert(topologicalOrder[numNodes - 1] == 5);
  // 6 has to be at the beginning
  assert(sccIndex[6] == 2);
  assert(topologicalOrder[0] == 6);
  // The rest belong to the middle SCC
  for (size_t i = 0; i < 5; i++)
    assert(sccIndex[i] == 1);
}

// Creates a chain of diamonds, possibly with a single back-edge
// When there is a back-edge, there should be a single SCC with order > 1
static void
TestDiamondChain(bool withBackEdge)
{
  /* Should be 3x+1, where x+1 is the number of "knots", the bottlenecks in the diamonds
     O   O   etc.
    / \ / \ /
   O   O   O   <- these are the knots
    \ / \ / \
     O   O   etc.
  */
  const size_t numNodes = 97;
  std::vector<std::vector<size_t>> successors(numNodes);

  // Use a permutation to "randomize" the node indices
  std::vector<size_t> perm(numNodes);
  for (size_t i = 0; i < numNodes; i++)
  {
    // 50 and 97 are coprime, so this will assign each node a unique new index
    perm[i] = (i * 50) % numNodes;
  }

  for (size_t knot = 0; knot + 3 < numNodes; knot += 3)
  {
    successors[perm[knot]].push_back(perm[knot + 1]);
    successors[perm[knot]].push_back(perm[knot + 2]);
    successors[perm[knot + 1]].push_back(perm[knot + 3]);
    successors[perm[knot + 2]].push_back(perm[knot + 3]);
  }

  if (withBackEdge)
  {
    // Leave 5 nodes on the end, and 5 at the beginning, out of the big loop
    successors[perm[numNodes - 6]].push_back(perm[5]);
  }

  auto GetSuccessors = [&](size_t node)
  {
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> topologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      GetSuccessors,
      sccIndex,
      topologicalOrder);
  ValidateTopologicalOrderAndSccIndices(numNodes, GetSuccessors, sccIndex, topologicalOrder);

  if (withBackEdge)
  {
    // With a back edge, only 5 nodes at the beginning and 5 nodes and the end are not in the loop
    assert(numSccs == 5 + 5 + 1);
    // All the middle nodes belong to one huge SCC
    auto loopScc = sccIndex[perm[5]];
    for (size_t i = 6; i < numNodes - 5; i++)
    {
      assert(sccIndex[perm[i]] == loopScc);
    }
  }
  else
  {
    // Without any back-edges, every node should be its own SCC
    assert(numSccs == numNodes);
  }
}

static int
TestTarjanSCC()
{
  TestDAG();
  TestCycles();
  TestDiamondChain(false);
  TestDiamondChain(true);
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestTarjanSCC", TestTarjanSCC)
