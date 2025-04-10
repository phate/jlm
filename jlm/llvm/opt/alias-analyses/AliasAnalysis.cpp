/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>

namespace jlm::llvm::aa
{

AliasAnalysis::AliasAnalysis() = default;

AliasAnalysis::~AliasAnalysis() = default;

PointsToGraphAliasAnalysis::PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph)
    : PointsToGraph_(pointsToGraph)
{}

PointsToGraphAliasAnalysis::~PointsToGraphAliasAnalysis() = default;

AliasAnalysis::AliasQueryResponse
PointsToGraphAliasAnalysis::Query(
    const rvsdg::output & p1,
    [[maybe_unused]] size_t s1,
    const rvsdg::output & p2,
    [[maybe_unused]] size_t s2)
{
  // If the two pointers are the same value, they must alias
  // This is the only situation where this analysis gives MustAlias,
  // as no offset information is stored in the PointsToGraph.
  if (&p1 == &p2)
    return MustAlias;

  // Assume that all pointers actually exist in the PointsToGraph
  auto & p1RegisterNode = PointsToGraph_.GetRegisterNode(p1);
  auto & p2RegisterNode = PointsToGraph_.GetRegisterNode(p2);

  // If the registers are represented by the same node, they may alias
  if (&p1RegisterNode == &p2RegisterNode)
    return MayAlias;

  // Check if both pointers may target the external node, to avoid iterating over large sets
  const auto & externalNode = PointsToGraph_.GetExternalMemoryNode();
  if (p1RegisterNode.HasTarget(externalNode) && p2RegisterNode.HasTarget(externalNode))
    return MayAlias;

  // Check if p1 and p2 share any target memory nodes
  for (auto & target : p1RegisterNode.Targets())
  {
    if (p2RegisterNode.HasTarget(target))
      return MayAlias;
  }

  return NoAlias;
}

std::string
PointsToGraphAliasAnalysis::ToString() const
{
  return "PointsToGraphAA";
}

AliasAnalysis::AliasQueryResponse
ChainedAliasAnalysis::Query(
    const rvsdg::output & p1,
    size_t s1,
    const rvsdg::output & p2,
    size_t s2)
{
  const auto firstResponse = First_.Query(p1, s1, p2, s2);

  // Anything other than MayAlias is precise, and can be returned right away
  if (firstResponse != MayAlias)
    return firstResponse;

  return Second_.Query(p1, s1, p2, s2);
}

std::string ChainedAliasAnalysis::ToString() const
{
  return util::strfmt("ChainedAA(", First_.ToString(), ", ", Second_.ToString(), ")");
}

}
