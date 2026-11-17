/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/RegionPredicateTrace.hpp>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/MatchVariant.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/util/common.hpp>

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace jlm::rvsdg
{

AlternativeRegionPredicateTracer::AlternativeRegionPredicateTracer() = default;

PredicateValueRange &
AlternativeRegionPredicateTracer::getPossibleValues(rvsdg::Output & output)
{
  auto & tracedOutput = rvsdg::traceOutputIntraProcedurally(output, false);
  if (auto it = predicateValueRanges_.find(&tracedOutput); it != predicateValueRanges_.end())
  {
    return it->second;
  }

  PredicateValueRange range = getPossibleValuesInternal(tracedOutput);
  auto [it, inserted] = predicateValueRanges_.emplace(&tracedOutput, std::move(range));
  JLM_ASSERT(inserted);
  return it->second;
}

PredicateValueRange
AlternativeRegionPredicateTracer::getPossibleValuesInternal(rvsdg::Output & output)
{
  auto & controlType = *util::assertedCast<const ControlType>(output.Type().get());

  // Control constants provide a known value for the predicate
  if (auto [_, ctrlOp] = TryGetSimpleNodeAndOptionalOp<ControlConstantOperation>(output); ctrlOp)
  {
    return PredicateValueRange::CreateSingleValue(ctrlOp->value());
  }

  // handle gamma outputs by taking the union of the possible values of each subregion
  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    auto exitVar = gamma->MapOutputExitVar(output);

    auto range = PredicateValueRange::CreateEmpty(controlType);
    for (auto result : exitVar.branchResult)
    {
      range.UpdateUnion(getPossibleValues(*result->origin()));
    }

    return range;
  }

  // This function is only called on traced outputs, so it never stops at a gamma argument
  JLM_ASSERT(!rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output));

  // handle theta outputs by continuing from the loop var post
  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    auto loopVar = theta->MapOutputLoopVar(output);
    return getPossibleValues(*loopVar.post->origin());
  }

  // Theta arguments belonging to invariant loop variables have already been traced.
  // Other theta arguments would require following back-edges.

  // Otherwise we are unable to provide a set
  return PredicateValueRange::CreateUnknown(controlType);
}

bool
AlternativeRegionPredicateTracer::markRequiredPredicateValue(
    Output & output,
    size_t value,
    std::vector<Region *> & impossibleRegions)
{
  bool satisfiable = markRequiredPredicateValueInternal(output, value, impossibleRegions);

  // If the output is never able to provide the required value, its region can not be an origin
  if (!satisfiable)
  {
    auto originRegion = output.region();

    // Multiple origins from the same region will never be considered,
    // so we are sure we never add duplicate regions to the list
    JLM_ASSERT(std::count(impossibleRegions.begin(), impossibleRegions.end(), originRegion) == 0);

    impossibleRegions.push_back(originRegion);
  }

  return satisfiable;
}

bool
AlternativeRegionPredicateTracer::markRequiredPredicateValueInternal(
    rvsdg::Output & output,
    size_t value,
    std::vector<Region *> & impossibleRegions)
{
  // handle gamma outputs by recursively propagating the requirement to each subregion
  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    auto exitVar = gamma->MapOutputExitVar(output);

    bool anyReachable = false;
    for (auto result : exitVar.branchResult)
    {
      anyReachable |= markRequiredPredicateValue(*result->origin(), value, impossibleRegions);
    }

    // If none of the gamma subregions can provide the required value,
    // its parent region is also not able to satisfy the requirement.
    return anyReachable;
  }

  // handle theta outputs by making the same requirement inside the theta
  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    auto loopVar = theta->MapOutputLoopVar(output);

    // This call can mark certain subregions within the theta region as impossible origins.
    // If it also returns false, it means the theta will never provide the required value,
    // which also makes the surrounding region an impossible origin.
    bool canLoopSatisfyValue =
        markRequiredPredicateValue(*loopVar.post->origin(), value, impossibleRegions);
    return canLoopSatisfyValue;
  }

  // The predicate output is not the output of a structural node.
  // It can still be the input of a structural node, but we can not continue
  // calling setRequiredPredicateValue out of the structural nodes.
  // The input may for example be used in only some subregions of a gamma,
  // or only the first iteration of a theta.
  // We can therefore not be sure that the value is actually required.
  // Also, RVSDG (pretty much) never routes ControlType values into structural nodes.

  // Use regular tracing to see if we are able to determine a fixed value for the output.
  auto & possibleValues = getPossibleValues(output);
  return possibleValues.AllowsValue(value);
}

bool
AlternativeRegionPredicateTracer::canCurrentOriginSatisfyRequirement(Output & output, size_t value)
{
  auto key = std::make_pair(&output, value);

  // If the (output, value) pair has already been processed, use the cached result
  auto [it, inserted] = impossibleOriginRegions_.insert({ key, {} });

  if (inserted)
  {
    // The requirement has not been processed, find regions that cannot satisfy it
    markRequiredPredicateValue(output, value, it->second);
  }

  // Go through all regions that have been marked as unable to satisfy the requirement,
  // and check if any of them are the origin region, or one of its ancestors.
  for (auto & impossibleOrigin : it->second)
  {
    if (currentOriginRegionAncestors_.Contains(impossibleOrigin))
      return false;
  }

  return true;
}

bool
AlternativeRegionPredicateTracer::isReachableFromRegion(
    Region & targetRegion,
    Region & originRegion)
{
  // find the common ancestor of the origin and target regions
  auto targetAncestor = &targetRegion;
  auto originAncestor = &originRegion;

  // While traversing, add all ancestors of the origin region to this set
  currentOriginRegionAncestors_.Clear();
  currentOriginRegionAncestors_.insert(originAncestor);

  while (targetAncestor != originAncestor)
  {
    const auto targetDepth = targetAncestor->getDepth();
    const auto originDepth = originAncestor->getDepth();

    // Move one region up along the target region ancestors
    if (targetDepth >= originDepth)
    {
      targetAncestor = targetAncestor->node()->region();
    }

    // Move one region up along the origin region ancestors
    if (originDepth >= targetDepth)
    {
      // If the origin ancestor region we are leaving is a theta subregion,
      // we can add the fact that the theta predicate must be 0 in order to leave the region
      auto node = originAncestor->node();
      if (auto theta = dynamic_cast<rvsdg::ThetaNode *>(node))
      {
        if (!canCurrentOriginSatisfyRequirement(*theta->predicate()->origin(), 0))
          return false;
      }

      // move one region up and add it to the set of origin region ancestors
      originAncestor = node->region();
      currentOriginRegionAncestors_.insert(originAncestor);
    }
  }
  // Lowest common ancestor found
  JLM_ASSERT(targetAncestor == originAncestor);
  const auto commonAncestor = targetAncestor;

  // Go through the ancestors of the target region and check if any of them have requirements
  // that can not be satisfied by the origin region or one of its ancestors
  targetAncestor = &targetRegion;
  while (targetAncestor != commonAncestor)
  {
    // when the target region is in a gamma subregion,
    // the origin must be able to provide the correct gamma predicate value
    auto node = targetAncestor->node();
    if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(node))
    {
      if (!canCurrentOriginSatisfyRequirement(
              *gamma->predicate()->origin(),
              targetAncestor->index()))
        return false;
    }
    targetAncestor = node->region();
  }

  // No proof of unreachability was found
  return true;
}

void
AlternativeRegionPredicateTracer::clearCaches()
{
  predicateValueRanges_.clear();
  impossibleOriginRegions_.clear();
}

}
