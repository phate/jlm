/*
 * Copyright 2026 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/RegionPredicateTrace.hpp>

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>

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
// predicate outputs / inputs. It records observations per
// region.
PredicateValueRange
RegionPredicateTrace::ComputeAndRecord(
    RegionPredRange & regionPredRange,
    Input & input,
    const ControlType & type)
{
  auto range = Compute(regionPredRange, input, type);
  regionPredRange.emplace(input.region(), range);
  ObserveRegion(*input.region());
  return range;
}

// Second part of the recursion, helper to the function above:
// performs actual recursion, and computes (but without recording,
// which is done by the controller function ComputeAndRecord above).
PredicateValueRange
RegionPredicateTrace::Compute(
    RegionPredRange & regionPredRange,
    Input & input,
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
            range.UpdateUnion(ComputeAndRecord(regionPredRange, *res, type));
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
            return ComputeAndRecord(regionPredRange, *loopVar.input, type);
          }
          else
          {
            return ComputeAndRecord(regionPredRange, *loopVar.post, type);
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
            return ComputeAndRecord(regionPredRange, *entry->input, type);
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
            return ComputeAndRecord(regionPredRange, *loopVar.input, type);
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
    ComputeAndRecord(range, predUse, *controlType);
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

}
