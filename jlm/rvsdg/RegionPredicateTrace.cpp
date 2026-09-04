/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/RegionPredicateTrace.hpp>

#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
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

// Observe changes to region that may invalidate the cached computation
// of predicate assignments / satisfiability constraints.
class RegionPredicateTrace::Observer : public RegionObserver
{
public:
  ~Observer() override
  {}

  Observer(const Region & region, RegionPredicateTrace * tracer)
      : RegionObserver(region),
        tracer_(tracer)
  {}

  void
  onNodeCreate(Node * node) override
  {}

  void
  onNodeDestroy(Node * node) override
  {
    // If a structural node is destroyed, then we may now
    // refer to region that does not exist any longer.
    // Just invalidate.
    MatchType(
        *node,
        [&](const StructuralNode &)
        {
          tracer_->Clear();
        });
  }

  void
  onInputCreate(Input * input) override
  {}

  void
  onInputChange(Input * input, Output * /* old_origin */, Output * /* new_origin */) override
  {
    // This is really the only operation we care about: One edge has been
    // changed.
    // We can constrain this to changes of control edges -- no
    // recomputation needed otherwise.
    if (std::dynamic_pointer_cast<const ControlType>(input->Type()))
    {
      tracer_->Clear();
    }
  }

  void
  onInputDestroy(Input * input) override
  {}

private:
  RegionPredicateTrace * tracer_;
};

RegionPredicateTrace::~RegionPredicateTrace()
{}

RegionPredicateTrace::RegionPredicateTrace()
{}

void
RegionPredicateTrace::Clear()
{
  // This is the cache invalidation signal: Some control
  // edge assignment has changed. Just to be safe,
  // invalidate all computations.
  predAssignment_.clear();
  predSat_.clear();

  // Note: cannot clear observers here (might be within
  // observer callback), so we will keep observing all
  // regions that have been registered at least once.
  // That is a slight over-approximation, but since we
  // constrain to observing only "control" defs/uses,
  // any change should rarely trigger, if at all.
}

void
RegionPredicateTrace::ObserveRegion(Region & region)
{
  if (observers_.find(&region) == observers_.end())
  {
    observers_.emplace(&region, std::make_unique<Observer>(region, this));
  }
}

// This function recurses through the "definition tree" of
// predicate outputs / inputs. It records observations per region.
const PredicateValueRange &
RegionPredicateTrace::ComputeAndRecord(
    RegionPredRange & regionPredRange,
    Input & input,
    std::unordered_map<Input *, PredicateValueRange> & visitedInputs,
    const ControlType & type)
{
  if (auto it = visitedInputs.find(&input); it != visitedInputs.end())
    return it->second;

  auto range = Compute(regionPredRange, input, visitedInputs, type);

  // If the input is a region result, add its value range to the region
  if (TryGetRegionParentNode<rvsdg::StructuralNode>(input))
  {
    auto it = regionPredRange.find(input.region());

    if (it != regionPredRange.end())
    {
      it->second.UpdateUnion(range);
    }
    else
    {
      regionPredRange.emplace(input.region(), range);
      ObserveRegion(*input.region());
    }
  }

  auto [it, inserted] = visitedInputs.emplace(&input, std::move(range));
  JLM_ASSERT(inserted);

  return it->second;
}

// Second part of the recursion, helper to the function above:
// performs actual recursion, and computes (but without recording,
// which is done by the controller function ComputeAndRecord above).
PredicateValueRange
RegionPredicateTrace::Compute(
    RegionPredRange & regionPredRange,
    Input & input,
    std::unordered_map<Input *, PredicateValueRange> & visitedInputs,
    const ControlType & type)
{
  // Given a predicate use site, record the predicate definition
  // value(s) that occur in this region, or passed as unchanged
  // values in this region.

  // Formal "definition site" of this predicate.
  auto origin = input.origin();
  if (auto node = TryGetOwnerNode<Node>(*origin))
  {
    return MatchTypeWithDefault(
        *node,
        [&](const rvsdg::SimpleNode & node) -> PredicateValueRange
        {
          // Is this a definite value assignment in this region?
          // Then record and terminate the recursion here.
          return MatchTypeWithDefault(
              node.GetOperation(),
              [&](const ControlConstantOperation & op)
              {
                return PredicateValueRange::CreateSingleValue(op.value());
              },
              [&]()
              {
                return PredicateValueRange::CreateUnknown(type);
              });
        },
        [&](const rvsdg::GammaNode & node) -> PredicateValueRange
        {
          // Is this predicate defined as output of gamma?
          // Then accumulate all values obtainable from the
          // different gamma branches into this region.
          auto exitVar = node.MapOutputExitVar(*origin);

          auto range = PredicateValueRange::CreateEmpty(type);
          for (auto res : exitVar.branchResult)
          {
            range.UpdateUnion(ComputeAndRecord(regionPredRange, *res, visitedInputs, type));
          }

          return range;
        },
        [&](const rvsdg::ThetaNode & node) -> PredicateValueRange
        {
          // For theta, check if it is a pass-through -- use
          // the value passed through, if applicable, or
          // declare "indeterminate value".
          auto loopVar = node.MapOutputLoopVar(*origin);
          if (loopVar.post->origin() == loopVar.pre)
          {
            return ComputeAndRecord(regionPredRange, *loopVar.input, visitedInputs, type);
          }
          else
          {
            return ComputeAndRecord(regionPredRange, *loopVar.post, visitedInputs, type);
          }
        },
        [&]()
        {
          return PredicateValueRange::CreateUnknown(type);
        });
  }
  else if (auto node = TryGetRegionParentNode<Node>(*origin))
  {
    // The predicate value is "defined" as input into this region.
    // Trace out of this region, and record possible values
    // entering this region.
    return MatchTypeWithDefault(
        *node,
        [&](const rvsdg::GammaNode & node) -> PredicateValueRange
        {
          auto argVar = node.MapBranchArgument(*origin);

          if (auto entry = std::get_if<GammaNode::EntryVar>(&argVar))
          {
            return ComputeAndRecord(regionPredRange, *entry->input, visitedInputs, type);
          }
          else
          {
            return PredicateValueRange::CreateUnknown(type);
          }
        },
        [&](rvsdg::ThetaNode & node) -> PredicateValueRange
        {
          auto loopVar = node.MapPreLoopVar(*origin);
          if (loopVar.post->origin() == loopVar.pre)
          {
            return ComputeAndRecord(regionPredRange, *loopVar.input, visitedInputs, type);
          }
          else
          {
            return PredicateValueRange::CreateUnknown(type);
          }
        },
        [&]() -> PredicateValueRange
        {
          return PredicateValueRange::CreateUnknown(type);
        });
  }
  else
  {
    return PredicateValueRange::CreateUnknown(type);
  }
}

PredicateValueRange
RegionPredicateTrace::GetRegionPredicateAssignConstraints(Region & region, Input & predUse)
{
  // Check for control type, ignore if wrong type.
  auto controlType = std::dynamic_pointer_cast<const ControlType>(predUse.Type());
  if (!controlType)
  {
    return PredicateValueRange::CreateEmpty(ControlType{ 0 });
  }

  auto i = predAssignment_.find(&predUse);
  if (i == predAssignment_.end())
  {
    // Recursively trace from the predicate use site to its
    // definition sites in different regions. Record predicate
    // assignments per region.
    RegionPredRange range;
    std::unordered_map<Input *, PredicateValueRange> visitedInputs;
    ComputeAndRecord(range, predUse, visitedInputs, *controlType);
    i = predAssignment_.emplace(&predUse, std::move(range)).first;
  }

  const RegionPredRange & regionRange = i->second;
  auto j = regionRange.find(&region);

  return j != regionRange.end() ? j->second : PredicateValueRange::CreateUnknown(*controlType);
}

PredicateSatRequired
RegionPredicateTrace::GetRegionSatRequired(Region & region)
{
  ObserveRegion(region);
  auto i = predSat_.find(&region);
  if (i == predSat_.end())
  {
    if (region.node())
    {
      // Recursively check all regions that this region is nested in.
      // Accumulate all predicates.
      PredicateSatRequired req = GetRegionSatRequired(*region.node()->region());

      // If this region is owned by a gamma node itself, then it is
      // entered conditionally based on the predicate.
      MatchType(
          *region.node(),
          [&](const rvsdg::GammaNode & node)
          {
            req.push_back(std::make_pair(node.predicate(), region.index()));
          });
      i = predSat_.emplace(&region, std::move(req)).first;
    }
    else
    {
      i = predSat_.emplace(&region, PredicateSatRequired{}).first;
    }
  }

  return i->second;
}

bool
RegionPredicateTrace::CheckPredicatesSatisfiable(Region & originRegion, Region & targetRegion)
{
  // Compute "required" predicates + values to enter this region.
  for (auto [pred, value] : GetRegionSatRequired(targetRegion))
  {
    // Check which predicate values the origin region would
    // necessarily assign.
    auto assigned = GetRegionPredicateAssignConstraints(originRegion, *pred);
    if (!assigned.AllowsValue(value))
    {
      // Unsatisfiable, coming from "originRegion", we can never enter
      // "targetRegion".
      return false;
    }
  }

  return true;
}

AlternativeRegionPredicateTracer::AlternativeRegionPredicateTracer() = default;

PredicateValueRange &
AlternativeRegionPredicateTracer::getPossibleValues(rvsdg::Output & output)
{
  auto & tracedOutput = rvsdg::traceOutputIntraProcedurally(output);
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
    // output is the output of a gamma exit variable
    // continue inside each of the gamma subregions
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
    // output is the output of a theta loop variable
    // continue from the loop variable post
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

  // If the target region has already been processed, use the cached result
  auto [it, inserted] = impossibleOriginRegions_.insert({ key, {} });

  if (inserted)
  {
    // The requirement has not been processed, find regions that cannot satisfy it
    markRequiredPredicateValue(output, value, it->second);
  }

  // Go through all regions that have been marked as unable to satisfy the requirement
  // of the target region, and check if any of them are ancestors of the origin region
  for (auto & impossibleOrigin : it->second)
  {
    if (currentOriginRegionAncestors_.Contains(impossibleOrigin))
      return false;
  }

  return true;
}

bool
AlternativeRegionPredicateTracer::canRegionReachRegion(Region & originRegion, Region & targetRegion)
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
  auto commonAncestor = targetAncestor;

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
