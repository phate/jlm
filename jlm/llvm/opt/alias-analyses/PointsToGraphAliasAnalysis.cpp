/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/PointsToGraphAliasAnalysis.hpp>

namespace jlm::llvm::aa
{
PointsToGraphAliasAnalysis::PointsToGraphAliasAnalysis(const PointsToGraph & pointsToGraph)
    : pointsToGraph_(pointsToGraph)
{}

PointsToGraphAliasAnalysis::~PointsToGraphAliasAnalysis() noexcept = default;

std::string
PointsToGraphAliasAnalysis::ToString() const
{
  return "PointsToGraphAA";
}

AliasAnalysis::AliasQueryResponse
PointsToGraphAliasAnalysis::Query(
    const rvsdg::Output & p1,
    const size_t s1,
    const rvsdg::Output & p2,
    const size_t s2)
{
  // If the two pointers are the same value, they must alias
  if (&p1 == &p2)
    return MustAlias;

  // Assume that all pointers actually exist in the PointsToGraph
  const auto p1RegisterNode = pointsToGraph_.getNodeForRegister(p1);
  const auto p2RegisterNode = pointsToGraph_.getNodeForRegister(p2);

  // Check if both pointers may target the external node, to avoid iterating over large sets
  const bool p1TargetsExternal = pointsToGraph_.isTargetingAllExternallyAvailable(p1RegisterNode);
  const bool p2TargetsExternal = pointsToGraph_.isTargetingAllExternallyAvailable(p2RegisterNode);
  if (p1TargetsExternal && p2TargetsExternal)
    return MayAlias;

  // If both p1 and p2 have exactly one possible target, which is the same for both,
  // and is the same size as the operations, and only represents one concrete memory location,
  // we can respond with MustAlias
  if (s1 == s2)
  {
    const auto p1SingleTarget = TryGetSingleTarget(p1RegisterNode, s1);
    const auto p2SingleTarget = TryGetSingleTarget(p2RegisterNode, s2);
    if (p1SingleTarget.has_value() && p2SingleTarget.has_value()
        && *p1SingleTarget == *p2SingleTarget && IsRepresentingSingleMemoryLocation(*p1SingleTarget)
        && pointsToGraph_.tryGetNodeSize(*p1SingleTarget) == s1)
    {
      return MustAlias;
    }
  }

  // For a memory location to be the target of both p1 and p2, it needs to be large enough
  // to represent both [p1, p1+s1) and [p2, p2+s2)
  const auto neededSize = std::max(s1, s2);

  // At least one of the nodes only has explicit targets
  PointsToGraph::NodeIndex onlyExplicitTargetsNode = p1RegisterNode;
  PointsToGraph::NodeIndex otherNode = p2RegisterNode;
  if (p1TargetsExternal)
  {
    JLM_ASSERT(!p2TargetsExternal);
    std::swap(onlyExplicitTargetsNode, otherNode);
  }

  for (const auto target : pointsToGraph_.getExplicitTargets(onlyExplicitTargetsNode).Items())
  {
    // Skip memory locations that are too small
    const auto targetSize = pointsToGraph_.tryGetNodeSize(target);

    if (targetSize.has_value() && *targetSize < neededSize)
      continue;

    if (pointsToGraph_.isTargeting(otherNode, target))
      return MayAlias;
  }

  return NoAlias;
}

std::optional<PointsToGraph::NodeIndex>
PointsToGraphAliasAnalysis::TryGetSingleTarget(PointsToGraph::NodeIndex node, size_t size) const
{
  // Nodes that target everything external always have "infinite" targets
  if (pointsToGraph_.isTargetingAllExternallyAvailable(node))
    return std::nullopt;

  std::optional<PointsToGraph::NodeIndex> singleTarget = std::nullopt;

  for (const auto target : pointsToGraph_.getExplicitTargets(node).Items())
  {
    // Skip memory locations that are too small to hold size
    const auto targetSize = pointsToGraph_.tryGetNodeSize(target);
    if (targetSize.has_value() && *targetSize < size)
      continue;

    // If we already have a "single target", there is more than one target
    if (singleTarget.has_value())
      return std::nullopt;

    singleTarget = target;
  }

  return singleTarget;
}

bool
PointsToGraphAliasAnalysis::IsRepresentingSingleMemoryLocation(PointsToGraph::NodeIndex node) const
{
  switch (pointsToGraph_.getNodeKind(node))
  {
  case PointsToGraph::NodeKind::AllocaNode:
  case PointsToGraph::NodeKind::MallocNode:
  case PointsToGraph::NodeKind::RegisterNode:
    return false;
  case PointsToGraph::NodeKind::DeltaNode:
  case PointsToGraph::NodeKind::LambdaNode:
  case PointsToGraph::NodeKind::ImportNode:
    return true;
  default:
    throw std::logic_error("Unknown node kind");
  }
}

}
