/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include "jlm/rvsdg/Trace.hpp"
#include "jlm/rvsdg/control.hpp"
#include "jlm/rvsdg/node.hpp"
#include "jlm/rvsdg/region.hpp"
#include "jlm/rvsdg/simple-node.hpp"
#include "jlm/util/common.hpp"
#include <jlm/rvsdg/RegionPredicateTrace.hpp>

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <unordered_map>

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

void AlternativeRegionPredicateTracer::setTargetRegion(Region & targetRegion)
{
  if (targetRegion_ == &targetRegion)
    return;

  targetRegion_ = &targetRegion;
  targetRegionAncestors_.Clear();
  targetRegionAncestors_.insert(targetRegion_);
  topSeenTargetAncestor_ = targetRegion_;
}

bool AlternativeRegionPredicateTracer::setRequiredPredicateValue(rvsdg::Output & output, size_t value)
{
  // By default assume this output is able to provide the required value
  const auto [it, inserted] = processedOutputs_.emplace(std::make_pair(&output, true));

  // We have already processed this output
  if (!inserted)
    return it->second;

  // gamma output
  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
  {
    // output is the output of a gamma exit variable
    // continue inside each of the gamma subregions
    auto exitVar = gamma->MapOutputExitVar(output);

    bool anyReachable = false;
    for (auto result : exitVar.branchResult)
    {
      anyReachable |= setRequiredPredicateValue(*result->origin(), value);
    }

    if (!anyReachable)
    {
      impossibleOriginRegions_.insert(output.region());
      it->second = false;
      return false;
    }

    return true;
  }

  // theta output
  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
  {
    // output is the output of a theta loop variable
    // continue from the loop variable post
    auto loopVar = theta->MapOutputLoopVar(output);

    bool loopCanTerminate = setRequiredPredicateValue(*loopVar.post->origin(), value);
    if (!loopCanTerminate)
    {
      impossibleOriginRegions_.insert(output.region());
      it->second = false;
      return false;
    }

    return true;
  }

  // The predicate output is not the output of a structural node.
  // It can still be the input of a structural node, but we can not continue
  // calling setRequiredPredicateValue out of the structural nodes.
  // The input may for example be used in only some subregions of a gamma,
  // or only the first iteration of a theta.
  // We can therefore not be sure that the value is actually required.
  // Also, RVSDG (pretty much) never routes ControlType values into structural nodes.

  // Use regular tracing to see if we are able to determine the value of the node
  auto & tracedOutput = rvsdg::traceOutputIntraProcedurally(output);

  // If the value comes from a ControlConstant with the wrong alternative,
  // the output's region is not a possible origin region
  auto [_, ctrlCnstOp] = TryGetSimpleNodeAndOptionalOp<ControlConstantOperation>(tracedOutput);
  if (ctrlCnstOp && ctrlCnstOp->value().alternative() != value)
  {
    impossibleOriginRegions_.insert(output.region());
    it->second = false;
    return false;
  }

  return true;
}

void AlternativeRegionPredicateTracer::visitNextTargetRegionAncestor()
{
  JLM_ASSERT(!topSeenTargetAncestor_->IsRootRegion());

  // When leaving a gamma node, mark the predicate as known
  if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(topSeenTargetAncestor_->node()))
  {
    auto subregionIndex = topSeenTargetAncestor_->index();
    setRequiredPredicateValue(*gamma->predicate()->origin(), subregionIndex);
  }

  // Update top seen target ancestor and add it to the set
  topSeenTargetAncestor_ = topSeenTargetAncestor_->node()->region();
  targetRegionAncestors_.insert(topSeenTargetAncestor_);
}

bool
AlternativeRegionPredicateTracer::canRegionReachRegion(Region & originRegion, Region & targetRegion)
{
  setTargetRegion(targetRegion);

  // If any new region are marked as impossible during traversal,
  // we must re-do traversal again afterwards
  size_t numImpossibleRegions = impossibleOriginRegions_.Size();

  // Move up the region tree to find the lowest common ancestor region
  Region * originRegionAncestor = &originRegion;

  while (true)
  {
    // Make sure the region tree has been tarversed high enough on the target side
    while (topSeenTargetAncestor_->getDepth() > originRegionAncestor->getDepth())
    {
      visitNextTargetRegionAncestor();
    }

    if (impossibleOriginRegions_.Contains(originRegionAncestor))
      return false;

    // If the origin region ancestor has reached a target region ancestor,
    // we have reached the lowest common ancestor
    if (targetRegionAncestors_.Contains(originRegionAncestor))
      break;

    // Otherwise keep traversing to the next ancestor of the origin region
    JLM_ASSERT(!originRegionAncestor->IsRootRegion());

    // If the origin is in a theta subregion, require that the theta predicate is 0
    if (auto theta = dynamic_cast<rvsdg::GammaNode *>(originRegionAncestor->node()))
    {
      setRequiredPredicateValue(*theta->predicate()->origin(), 0);
    }
    originRegionAncestor = originRegionAncestor->node()->region();
  }

  // The target and origin regions have been traced to each other,
  // so if no new regions were determined to be impossible, we are done
  if (impossibleOriginRegions_.Size() == numImpossibleRegions)
    return true;

  // Try checking all regions between the originRegion and the common ancestor again
  Region * retraceOriginRegion = &originRegion;
  while (retraceOriginRegion != originRegionAncestor)
  {
    if (impossibleOriginRegions_.Contains(retraceOriginRegion))
      return false;
    retraceOriginRegion = retraceOriginRegion->node()->region();
  }

  // No proof of unreachability was found
  return true;
}

}
