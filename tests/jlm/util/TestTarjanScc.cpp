/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/util/TarjanScc.hpp>

#include <cassert>
#include <cstdint>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <iostream>

// Used to represent graphs with no node unification
static size_t
Identity(size_t i)
{
  return i;
}

/**
 * Validation function checking that the found SCCs correspond to the \p successors:
 *  - All sccIndex are between 0 and numSccs-1
 *  - All SCCs contain at least one node
 *  - All edges in the graph point to an \p sccIndex which is equal or lower
 *  - The reverseTopologicalOrder contains all nodes, sorted by descending sccIndex
 */
template<typename SuccessorFunctor>
static void
ValidateTopologicalOrderAndSccIndices(
    size_t numNodes,
    SuccessorFunctor & successors,
    size_t numSccs,
    const std::vector<size_t> & sccIndex,
    const std::vector<size_t> & reverseTopologicalOrder)
{
  assert(numNodes == sccIndex.size());

  // Check that all sccIndex are valid, and each SCC has at least one node
  std::vector<size_t> numNodesInScc(numSccs, 0);
  for (size_t i = 0; i < numNodes; i++)
  {
    assert(sccIndex[i] < numSccs);
    numNodesInScc[sccIndex[i]]++;
  }
  for (size_t i = 0; i < numSccs; i++)
    assert(numNodesInScc[i] > 0);

  // Check that no edge in the graph points to an earlier SCC
  for (size_t i = 0; i < numNodes; i++)
  {
    for (auto next : successors(i))
    {
      // Intra-scc edges are ignored
      if (sccIndex[i] == sccIndex[next])
        continue;

      // successor SCCs must have lower scc index
      assert(sccIndex[next] < sccIndex[i]);
    }
  }

  // Check that all nodes appear once in the topological order
  std::unordered_set<size_t> nodeInTopologicalOrder;
  assert(numNodes == reverseTopologicalOrder.size());
  for (size_t i = 0; i < numNodes; i++)
  {
    assert(reverseTopologicalOrder[i] < numNodes);
    nodeInTopologicalOrder.insert(reverseTopologicalOrder[i]);
  }
  assert(numNodes == nodeInTopologicalOrder.size());

  // Check that the topological order contains nodes sorted by ascending sccIndex
  for (size_t i = 1; i < numNodes; i++)
  {
    assert(sccIndex[reverseTopologicalOrder[i - 1]] <= sccIndex[reverseTopologicalOrder[i]]);
  }
}

static int
TestDag()
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
    { 7 },    // 7's successors (fully disconnected, just a self-loop)
  };

  auto GetSuccessors = [&](size_t node)
  {
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      Identity,
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);

  assert(numSccs == numNodes);
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestTarjanScc-TestDag", TestDag);

// Test a graph with some cycles, ensuring they become SCCs
static int
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
  std::vector<size_t> reverseTopologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      Identity,
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);

  assert(numSccs == 3);
  // 5 has to be at the end
  assert(sccIndex[5] == 0);
  assert(reverseTopologicalOrder[0] == 5);
  // 6 has to be at the beginning
  assert(sccIndex[6] == 2);
  assert(reverseTopologicalOrder[numNodes - 1] == 6);
  // The rest belong to the middle SCC
  for (size_t i = 0; i < 5; i++)
    assert(sccIndex[i] == 1);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestTarjanScc-TestCycles", TestCycles);

/**
 * Creates a chain of diamonds, possibly with an extra edge. Performs SCC on the graph.
 * The node indices are shuffled before being sent to the SCC algorithm.
 *
 * @param knots
 * The knots are the bottlenecks in the diamonds.
 * All edges go that way ==>
 *    1   4   etc.
 *   / \ / \ /
 *  0   3   6   <- these are the knots
 *   \ / \ / \
 *    2   5   etc.
 *
 * @param extraEdge an optional extra edge going from first to second.
 * If the extra edge is a forward edge (first < second), no cycles exist.
 * If the extra edge is a back edge, there should be a single SCC with order > 1.
 *
 * @return a pair: the number of nodes, the number of SCCs,
 * and the sccIndex of each node, after un-shuffling
 */
static std::tuple<size_t, size_t, std::vector<size_t>>
CreateDiamondChain(size_t knots, std::optional<std::pair<size_t, size_t>> extraEdge)
{
  assert(knots >= 2);
  const size_t numNodes = 3 * knots - 2;
  std::vector<std::vector<size_t>> successors(numNodes);

  // Use a permutation to "randomize" the node indices
  std::vector<size_t> perm(numNodes);
  for (size_t i = 0; i < numNodes; i++)
  {
    // 3 and numNodes is always coprime, so this will assign each node a unique new index
    perm[i] = (i * 3) % numNodes;
  }

  for (size_t knot = 0; knot + 3 < numNodes; knot += 3)
  {
    successors[perm[knot]].push_back(perm[knot + 1]);
    successors[perm[knot]].push_back(perm[knot + 2]);
    successors[perm[knot + 1]].push_back(perm[knot + 3]);
    successors[perm[knot + 2]].push_back(perm[knot + 3]);
  }

  if (extraEdge)
  {
    assert(extraEdge->first < numNodes && extraEdge->second < numNodes);
    // Leave 5 nodes on the end, and 5 at the beginning, out of the big loop
    successors[perm[extraEdge->first]].push_back(perm[extraEdge->second]);
  }

  auto GetSuccessors = [&](size_t node)
  {
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      Identity,
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);

  std::vector<size_t> unshuffledNodeIndex(numNodes);
  for (size_t i = 0; i < numNodes; i++)
  {
    unshuffledNodeIndex[i] = sccIndex[perm[i]];
  }

  return { numNodes, numSccs, std::move(unshuffledNodeIndex) };
}

static int
TestSimpleDiamondChain()
{
  auto [numNodes, numSccs, sccIndex] = CreateDiamondChain(100, std::nullopt);
  assert(numNodes == numSccs);
  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestTarjanScc-TestSimpleDiamondChain", TestSimpleDiamondChain);

static int
TestDiamondChainWithForwardEdge()
{
  // Forward edges do not create any cycles
  std::pair<size_t, size_t> forwardEdge{ 5, 200 };
  auto [numNodes, numSccs, sccIndex] = CreateDiamondChain(100, forwardEdge);
  assert(numNodes == numSccs);
  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/TestTarjanScc-TestDiamondChainWithForwardEdge",
    TestDiamondChainWithForwardEdge);

static int
TestDiamondChainWithBackEdge()
{
  // Back edges create one big SCC, the rest are single node SCCs
  std::pair<size_t, size_t> backEdge{ 3 * 50, 3 * 3 };
  auto [numNodes, numSccs, sccIndex] = CreateDiamondChain(100, backEdge);

  std::cerr << "numSccs: " << numSccs << std::endl;
  std::cerr << "numNodes: " << numNodes << std::endl;
  assert(numSccs == backEdge.second + numNodes - backEdge.first);

  auto largeScc = sccIndex[backEdge.second];

  // All nodes before the back edge head have a higher SCC index
  for (size_t i = 0; i < backEdge.second; i++)
    assert(sccIndex[i] > largeScc);

  // All nodes after the back edge tail have a lower SCC index
  for (size_t i = backEdge.first + 1; i < numNodes; i++)
    assert(sccIndex[i] < largeScc);

  // All nodes between the back edge tail and head have same SCC index
  for (size_t i = backEdge.second; i <= backEdge.first; i++)
    assert(sccIndex[i] == largeScc);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/util/TestTarjanScc-TestDiamondChainWithBackEdge",
    TestDiamondChainWithBackEdge);

// During SCC creation, the function should query a node for its successors at most twice
static int
TestVisitEachNodeTwice()
{
  const size_t numNodes = 5;
  std::vector<std::vector<size_t>> successors{
    { 1, 2 }, // 0's successors
    { 2, 3 }, // 1's successors
    { 1, 3 }, // 2's successors
    {},       // 3's successors
    { 4 }     // 4's successors
  };
  // The graph looks like a 0->(12)->3 diamond with edges between 1 and 2, and a lone node 4

  std::vector<size_t> successorsQueried(numNodes, 0);
  auto GetSuccessors = [&](size_t node)
  {
    JLM_ASSERT(node < numNodes);
    successorsQueried[node]++;
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      Identity,
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);

  JLM_ASSERT(numSccs == 4);
  for (size_t timesQueried : successorsQueried)
    JLM_ASSERT(timesQueried <= 2);

  // Validate the produced SCC DAG as well, but do it last, as this function calls GetSuccessors.
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/util/TestTarjanScc-TestVisitEachNodeTwice", TestVisitEachNodeTwice);
