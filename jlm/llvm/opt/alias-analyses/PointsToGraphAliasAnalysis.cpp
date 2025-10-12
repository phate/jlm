/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/PointsToGraphAliasAnalysis.hpp>

namespace jlm::llvm::aa
{
PointsToGraphAliasAnalysis::PointsToGraphAliasAnalysis(const PointsToGraph & pointsToGraph)
    : PointsToGraph_(pointsToGraph)
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
  auto & p1RegisterNode = PointsToGraph_.GetRegisterNode(p1);
  auto & p2RegisterNode = PointsToGraph_.GetRegisterNode(p2);

  // Check if both pointers may target the external node, to avoid iterating over large sets
  const auto & externalNode = PointsToGraph_.GetExternalMemoryNode();
  if (p1RegisterNode.HasTarget(externalNode) && p2RegisterNode.HasTarget(externalNode))
    return MayAlias;

  // If both p1 and p2 have exactly one possible target, which is the same for both,
  // and is the same size as the operations, and only represents one concrete memory location,
  // we can respond with MustAlias
  if (s1 == s2)
  {
    const auto p1SingleTarget = TryGetSingleTarget(p1RegisterNode, s1);
    const auto p2SingleTarget = TryGetSingleTarget(p2RegisterNode, s2);
    if (p1SingleTarget && p1SingleTarget == p2SingleTarget
        && IsRepresentingSingleMemoryLocation(*p1SingleTarget)
        && GetMemoryNodeSize(*p1SingleTarget) == s1)
    {
      return MustAlias;
    }
  }

  // For a memory location to be the target of both p1 and p2, it needs to be large enough
  // to represent both [p1, p1+s1) and [p2, p2+s2)
  const auto neededSize = std::max(s1, s2);

  for (auto & target : p1RegisterNode.Targets())
  {
    // Skip memory locations that are too small
    const auto targetSize = GetMemoryNodeSize(target);
    if (targetSize.has_value() && *targetSize < neededSize)
      continue;

    if (p2RegisterNode.HasTarget(target))
      return MayAlias;
  }

  return NoAlias;
}

const PointsToGraph::MemoryNode *
PointsToGraphAliasAnalysis::TryGetSingleTarget(
    const PointsToGraph::RegisterNode & node,
    size_t size)
{
  const PointsToGraph::MemoryNode * singleTarget = nullptr;
  for (auto & target : node.Targets())
  {
    // Skip memory locations that are too small to hold size
    const auto targetSize = GetMemoryNodeSize(target);
    if (targetSize.has_value() && *targetSize < size)
      continue;

    // If we already have a "single target", there is more than one target
    if (singleTarget)
      return nullptr;

    singleTarget = &target;
  }

  return singleTarget;
}

std::optional<size_t>
PointsToGraphAliasAnalysis::GetMemoryNodeSize(const PointsToGraph::MemoryNode & node)
{
  if (auto delta = dynamic_cast<const PointsToGraph::DeltaNode *>(&node))
    return GetTypeSize(*delta->GetDeltaNode().GetOperation().Type());
  if (auto import = dynamic_cast<const PointsToGraph::ImportNode *>(&node))
  {
    auto size = GetTypeSize(*import->GetArgument().ValueType());
    // Workaround for imported incomplete types appearing to have size 0 in the LLVM IR
    if (size == 0)
      return std::nullopt;

    return size;
  }
  if (auto alloca = dynamic_cast<const PointsToGraph::AllocaNode *>(&node))
  {
    const auto & allocaNode = alloca->GetAllocaNode();
    const auto allocaOp = util::assertedCast<const AllocaOperation>(&allocaNode.GetOperation());

    // An alloca has a count parameter, which on rare occasions is not just the constant 1.
    const auto elementCount = TryGetConstantSignedInteger(*allocaNode.input(0)->origin());
    if (elementCount.has_value())
      return *elementCount * GetTypeSize(*allocaOp->ValueType());
  }
  if (auto malloc = dynamic_cast<const PointsToGraph::MallocNode *>(&node))
  {
    const auto & mallocNode = malloc->GetMallocNode();

    const auto mallocSize = TryGetConstantSignedInteger(*mallocNode.input(0)->origin());
    if (mallocSize.has_value())
      return *mallocSize;
  }

  return std::nullopt;
}

bool
PointsToGraphAliasAnalysis::IsRepresentingSingleMemoryLocation(
    const PointsToGraph::MemoryNode & node)
{
  return PointsToGraph::Node::Is<PointsToGraph::DeltaNode>(node)
      || PointsToGraph::Node::Is<PointsToGraph::ImportNode>(node)
      || PointsToGraph::Node::Is<PointsToGraph::LambdaNode>(node);
}

}
