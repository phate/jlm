/*
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/util/HashSet.hpp>
#include <jlm/util/TarjanScc.hpp>

#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

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
template<typename UnificationRootFunctor, typename SuccessorFunctor>
static void
ValidateTopologicalOrderAndSccIndices(
    size_t numNodes,
    UnificationRootFunctor & unificationRoot,
    SuccessorFunctor & successors,
    size_t numSccs,
    const std::vector<size_t> & sccIndex,
    const std::vector<size_t> & reverseTopologicalOrder)
{
  EXPECT_EQ(numNodes, sccIndex.size());

  // Check that all sccIndex are valid, and each SCC has at least one node
  std::vector<size_t> numNodesInScc(numSccs, 0);
  for (size_t i = 0; i < numNodes; i++)
  {
    auto node = unificationRoot(i);
    EXPECT_LT(sccIndex[node], numSccs);
    numNodesInScc[sccIndex[node]]++;
  }
  for (size_t i = 0; i < numSccs; i++)
    EXPECT_GT(numNodesInScc[i], 0);

  // Check that no edge in the graph points to an earlier SCC
  for (size_t i = 0; i < numNodes; i++)
  {
    // Only consider unification roots
    if (unificationRoot(i) != i)
      continue;

    for (auto next : successors(i))
    {
      next = unificationRoot(next);

      // successor SCCs must have lower scc index
      EXPECT_LE(sccIndex[next], sccIndex[i]);
    }
  }

  // Check that all unification roots appear exactly once in the topological order
  jlm::util::HashSet<size_t> nodeInTopologicalOrder(
      reverseTopologicalOrder.begin(),
      reverseTopologicalOrder.end());
  EXPECT_EQ(nodeInTopologicalOrder.Size(), reverseTopologicalOrder.size());

  for (size_t i = 0; i < numNodes; i++)
  {
    if (unificationRoot(i) == i)
      EXPECT_TRUE(nodeInTopologicalOrder.Contains(i));
    else
      EXPECT_TRUE(!nodeInTopologicalOrder.Contains(i));
  }

  // Check that the reverse topological order contains nodes with ascending sccIndex
  for (size_t i = 1; i < reverseTopologicalOrder.size(); i++)
  {
    EXPECT_LE(sccIndex[reverseTopologicalOrder[i - 1]], sccIndex[reverseTopologicalOrder[i]]);
  }
}

TEST(TarjanSccTests, TestDag)
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
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      Identity,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);

  EXPECT_EQ(numSccs, numNodes);
}

// Test a graph with some cycles, ensuring they become SCCs
TEST(TarjanSccTests, TestCycles)
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
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      Identity,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);

  EXPECT_EQ(numSccs, 3);
  // 5 has to be at the end
  EXPECT_EQ(sccIndex[5], 0);
  EXPECT_EQ(reverseTopologicalOrder[0], 5);
  // 6 has to be at the beginning
  EXPECT_EQ(sccIndex[6], 2);
  EXPECT_EQ(reverseTopologicalOrder[numNodes - 1], 6);
  // The rest belong to the middle SCC
  for (size_t i = 0; i < 5; i++)
    EXPECT_EQ(sccIndex[i], 1);
}

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
  EXPECT_GE(knots, 2);
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
    EXPECT_LT(extraEdge->first, numNodes);
    EXPECT_LT(extraEdge->second, numNodes);
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
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      Identity,
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

TEST(TarjanSccTests, TestSimpleDiamondChain)
{
  auto [numNodes, numSccs, sccIndex] = CreateDiamondChain(100, std::nullopt);
  EXPECT_EQ(numNodes, numSccs);
}

TEST(TarjanSccTests, TestDiamondChainWithForwardEdge)
{
  // Forward edges do not create any cycles
  std::pair<size_t, size_t> forwardEdge{ 5, 200 };
  auto [numNodes, numSccs, sccIndex] = CreateDiamondChain(100, forwardEdge);
  EXPECT_EQ(numNodes, numSccs);
}

TEST(TarjanSccTests, TestDiamondChainWithBackEdge)
{
  // Back edges create one big SCC, the rest are single node SCCs
  std::pair<size_t, size_t> backEdge{ 3 * 50, 3 * 3 };
  auto [numNodes, numSccs, sccIndex] = CreateDiamondChain(100, backEdge);

  std::cerr << "numSccs: " << numSccs << std::endl;
  std::cerr << "numNodes: " << numNodes << std::endl;
  EXPECT_EQ(numSccs, backEdge.second + numNodes - backEdge.first);

  auto largeScc = sccIndex[backEdge.second];

  // All nodes before the back edge head have a higher SCC index
  for (size_t i = 0; i < backEdge.second; i++)
    EXPECT_GT(sccIndex[i], largeScc);

  // All nodes after the back edge tail have a lower SCC index
  for (size_t i = backEdge.first + 1; i < numNodes; i++)
    EXPECT_LT(sccIndex[i], largeScc);

  // All nodes between the back edge tail and head have same SCC index
  for (size_t i = backEdge.second; i <= backEdge.first; i++)
    EXPECT_EQ(sccIndex[i], largeScc);
}

// During SCC creation, the function should query a node for its successors at most twice
TEST(TarjanSccTests, TestVisitEachNodeTwice)
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
    EXPECT_LT(node, numNodes);
    successorsQueried[node]++;
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);

  EXPECT_EQ(numSccs, 4);
  for (size_t timesQueried : successorsQueried)
    EXPECT_LE(timesQueried, 2);

  // Validate the produced SCC DAG as well, but do it last, as this function calls GetSuccessors.
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      Identity,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);
}

TEST(TarjanSccTests, TestUnifiedNodes)
{
  // Each node with index >= 5 has a unification root equal to index - 5
  const size_t numNodes = 10;
  std::vector<std::vector<size_t>> successors{
    { 1, 1 + 5, 2 },  // 0's successors
    { 2 + 5, 3 },     // 1's successors
    { 1 + 5, 3 + 5 }, // 2's successors
    {},               // 3's successors
    { 4, 4 + 5 }      // 4's successors
  };
  // The graph looks like a 0->(12)->3 diamond with edges between 1 and 2, and a lone node 4

  auto GetUnificationRoot = [&](size_t node)
  {
    if (node >= 5)
      return node - 5;
    return node;
  };

  auto GetSuccessors = [&](size_t node)
  {
    EXPECT_LT(node, 5);
    return successors[node];
  };

  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSccs = jlm::util::FindStronglyConnectedComponents(
      numNodes,
      GetUnificationRoot,
      GetSuccessors,
      sccIndex,
      reverseTopologicalOrder);

  EXPECT_EQ(numSccs, 4);

  // Validate the produced SCC DAG as well, but do it last, as this function calls GetSuccessors.
  ValidateTopologicalOrderAndSccIndices(
      numNodes,
      GetUnificationRoot,
      GetSuccessors,
      numSccs,
      sccIndex,
      reverseTopologicalOrder);
}
