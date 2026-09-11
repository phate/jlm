/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>

namespace jlm::rvsdg
{
OutputTracer::~OutputTracer() = default;

OutputTracer::OutputTracer() noexcept
{}

Output &
OutputTracer::trace(Output & output)
{
  return trace(output, nullptr);
}

Output &
OutputTracer::trace(Output & output, const rvsdg::Region * withinRegion)
{
  regionPredicateTracer_.clearCaches();
  startingOutput_ = &output;
  return traceInternal(output, false, withinRegion);
}

Output &
OutputTracer::traceInternal(
    Output & output,
    bool loopBackEdgeTaken,
    const rvsdg::Region * withinRegion)
{
  Output * head = &output;

  // Keep tracing until the output stops changing
  while (true)
  {
    Output * prevHead = head;
    head = &traceStep(*head, loopBackEdgeTaken, withinRegion);
    if (head == prevHead)
    {
      return *head;
    }
  }
}

/**
 * Gets the origin of the given gamma subregion argument, outside the gamma node
 * @param gammaNode the gamma node
 * @param output an argument in one of \p gammaNode's subregions
 * @return the origin of the argument outside the gamma node
 */
static Output &
mapGammaArgumentToOrigin(GammaNode & gammaNode, Output & output)
{
  return *gammaNode.mapBranchArgumentToInput(output).origin();
}

Output &
OutputTracer::traceGammaOutput(GammaNode & gammaNode, Output & output, bool loopBackEdgeTaken)
{
  // First check the invariance cache
  if (const auto invariantValueInput = lookupInInvarianceCache(output, loopBackEdgeTaken))
  {
    return *invariantValueInput->origin();
  }

  const auto exitVar = gammaNode.MapOutputExitVar(output);

  // The shared output that is the origin of the entry variable(s) going into the gamma node
  // nullopt means no subregion has been traced yet.
  // nullptr means there can be no shared common outer origin.
  std::optional<Output *> commonOuterOrigin;
  // The gamma input that gets its value from the common outer origin
  Input * commonGammaInput = nullptr;

  // If only a single subregion can provide the value, this is the origin within that subregion.
  // nullopt means no valid subregion been found yet.
  // nullptr means there are multiple valid subregions.
  std::optional<Output *> singleInnerOrigin;

  for (auto branchResult : exitVar.branchResult)
  {
    // Check if region predicate checking is enabled and lets us skip this subregion
    if (!loopBackEdgeTaken && regionPredicateChecking_)
    {
      // If the current gamma subregion can not reach the starting region for the trace,
      // it can not be the region providing the traced value
      if (!regionPredicateTracer_.isReachableFromRegion(
              *startingOutput_->region(),
              *branchResult->region()))
        continue;
    }

    auto tracedInner = branchResult->origin();

    if (isDeepInvarianceCheckingEnabled())
    {
      // Trace the branch result origin, but only within the gamma subregion
      tracedInner = &traceInternal(*tracedInner, loopBackEdgeTaken, tracedInner->region());
    }

    // Set the single inner origin, or clear it if we already had one
    if (!singleInnerOrigin.has_value())
    {
      singleInnerOrigin = tracedInner;
    }
    else
    {
      singleInnerOrigin = nullptr;
    }

    // The traced output must reach a region argument in the gamma subregion
    if (TryGetRegionParentNode<GammaNode>(*tracedInner) != &gammaNode)
    {
      commonOuterOrigin = nullptr;
    }
    else
    {
      // Get the origin of the region argument outside the gamma
      auto & gammaInput = gammaNode.mapBranchArgumentToInput(*tracedInner);
      Output & outerOrigin = *gammaInput.origin();

      // If this is the first outer origin, make it the common outer origin for now
      if (!commonOuterOrigin.has_value())
      {
        commonOuterOrigin = &outerOrigin;
        commonGammaInput = &gammaInput;
      }
      else if (*commonOuterOrigin != &outerOrigin)
      {
        // Mismatching outer origins found
        commonOuterOrigin = nullptr;
      }
    }
  }

  // If we found a common outer origin, continue tracing from there
  if (commonOuterOrigin.has_value() && *commonOuterOrigin != nullptr)
  {
    JLM_ASSERT(commonGammaInput != nullptr);
    return insertInInvarianceCache(output, loopBackEdgeTaken, *commonGammaInput);
  }

  // If only a single gamma subregion provides a possible origin
  if (isTracingIntoSubregionsEnabled() && singleInnerOrigin.has_value()
      && *singleInnerOrigin != nullptr)
  {
    return **singleInnerOrigin;
  }

  // Tracing was unable to make any progress beyond the gamma output
  return output;
}

Output &
OutputTracer::traceThetaOutput(ThetaNode & thetaNode, Output & output, bool loopBackEdgeTaken)
{
  if (const auto invariantValueInput = lookupInInvarianceCache(output, loopBackEdgeTaken))
  {
    return *invariantValueInput->origin();
  }

  const auto loopVar = thetaNode.MapOutputLoopVar(output);

  auto tracedInner = loopVar.post->origin();

  // If invariance detection is enabled, perform tracing inside the subregion
  if (isDeepInvarianceCheckingEnabled())
  {
    // trace the origin within the thetaNode, but only within the theta's subregion
    tracedInner = &traceInternal(*tracedInner, loopBackEdgeTaken, thetaNode.subregion());
  }

  // If tracing reached the pre argument of the same loop variable, it might be invariant
  if (tracedInner == loopVar.pre)
  {
    // If the loop variable was found to be invariant, but loopBackEdgeTaken was false,
    // we must check again without making assumptions about no back-edges being taken

    // We already traced without making assumptions about back-edges
    if (loopBackEdgeTaken)
    {
      return insertInInvarianceCache(output, true, *loopVar.input);
    }

    // Try again with loopBackEdgeTaken=true
    auto tracedInnerAgain = &traceInternal(*tracedInner, true, thetaNode.subregion());
    if (tracedInnerAgain == loopVar.pre)
    {
      // It is still invariant!
      return insertInInvarianceCache(output, true, *loopVar.input);
    }
  }
  else if (TryGetRegionParentNode<ThetaNode>(*tracedInner) == &thetaNode)
  {
    // Tracing from the post result lead to the pre argument of a different loop variable.
    // Check if that loop variable is trivially invariant, and if it is, return its input origin.

    auto originLoopVar = thetaNode.MapPreLoopVar(*tracedInner);
    if (ThetaLoopVarIsInvariant(originLoopVar))
    {
      return insertInInvarianceCache(output, loopBackEdgeTaken, *originLoopVar.input);
    }
  }

  // If we are allowed to return outputs from inside the subregion,
  // return the result from tracing inside the subregion.
  if (isTracingIntoSubregionsEnabled())
  {
    return *tracedInner;
  }

  // Otherwise, we are unable to trace further from the theta output
  return output;
}

Output &
OutputTracer::traceThetaArgument(rvsdg::ThetaNode & thetaNode, Output & output)
{
  // Get the loop variable
  auto loopVar = thetaNode.MapPreLoopVar(output);

  // Trace from the corresponding theta output.
  auto & tracedOutput = traceThetaOutput(thetaNode, *loopVar.output, true);

  // If the loop output is invariant and has the same origin as the loop variable,
  // tracing can continue from outside the theta
  if (&tracedOutput == loopVar.input->origin())
  {
    return tracedOutput;
  }

  // Otherwise tracing stops at the theta argument
  return output;
}

Output &
OutputTracer::traceStep(Output & output, bool loopBackEdgeTaken, const rvsdg::Region * withinRegion)
{
  if (withinRegion && withinRegion == TryGetOwnerRegion(output))
  {
    // We are not allowed to leave this region, and tracing has reached one of its arguments
    return output;
  }

  // Handle gamma node outputs
  if (const auto gammaNode = TryGetOwnerNode<GammaNode>(output))
  {
    return traceGammaOutput(*gammaNode, output, loopBackEdgeTaken);
  }

  // Handle gamma node arguments
  if (const auto gammaNode = TryGetRegionParentNode<GammaNode>(output))
  {
    return mapGammaArgumentToOrigin(*gammaNode, output);
  }

  // Handle theta node outputs
  if (const auto thetaNode = TryGetOwnerNode<ThetaNode>(output))
  {
    return traceThetaOutput(*thetaNode, output, loopBackEdgeTaken);
  }

  // Handle theta node arguments
  if (const auto thetaNode = TryGetRegionParentNode<ThetaNode>(output))
  {
    // When reaching theta arguments this way, it it always because the starting point
    // of the tracing is inside the theta subregion
    JLM_ASSERT(!loopBackEdgeTaken);
    JLM_ASSERT(Region::isAncestorOrSame(*startingOutput_->region(), *thetaNode->subregion()));
    return traceThetaArgument(*thetaNode, output);
  }

  // If we are not doing interprocedural tracing, stop tracing now
  if (!isInterprocedural_)
    return output;

  // Handle lambda context variables
  if (const auto lambda = TryGetRegionParentNode<LambdaNode>(output))
  {
    // If the argument is a contex variable, continue tracing
    if (const auto ctxVar = lambda->MapBinderContextVar(output))
      return *ctxVar->input->origin();

    return output;
  }

  // Handle delta context variables
  if (const auto delta = TryGetRegionParentNode<DeltaNode>(output))
  {
    // If the argument is a contex variable, continue tracing
    const auto ctxVar = delta->MapBinderContextVar(output);
    return *ctxVar.input->origin();
  }

  // Handle phi outputs
  if (const auto phiNode = TryGetOwnerNode<PhiNode>(output))
  {
    if (enterPhiNodes_)
    {
      const auto fixVar = phiNode->MapOutputFixVar(output);
      return *fixVar.result->origin();
    }
    return output;
  }

  // Handle phi region arguments
  if (const auto phiNode = TryGetRegionParentNode<PhiNode>(output))
  {
    // Wo only trace through contex variables.
    // Going through recursion variables would hide the fact that recursion is happening,
    // and risks producing an output that is a successor of the output we started with in the DAG.
    const auto argument = phiNode->MapArgument(output);
    if (const auto ctxVar = std::get_if<PhiNode::ContextVar>(&argument))
    {
      // Follow the context variable to outside the phi
      return *ctxVar->input->origin();
    }
    return output;
  }

  return output;
}

Output &
OutputTracer::insertInInvarianceCache(
    const Output & output,
    bool loopBackEdgeTaken,
    Input & traceResult)
{
  if (enableInvarianceCaching_)
  {
    auto [it, inserted] =
        invariantOutputCache_.emplace(&output, std::make_pair(loopBackEdgeTaken, &traceResult));
    if (!inserted)
    {
      // The only reason we would be inserting into the cache for the same output again,
      // is if the old cache entry is only valid for loopBackEdgeTaken = false,
      // and we now have one where loopBackEdgeTaken = true, which is valid for both.
      // Both the old and new cached value must point to inputs with the same origin.
      JLM_ASSERT(!it->second.first && loopBackEdgeTaken);
      JLM_ASSERT(it->second.second->origin() == traceResult.origin());
      it->second.first = loopBackEdgeTaken;
    }
  }

  return *traceResult.origin();
}

Input *
OutputTracer::lookupInInvarianceCache(const Output & output, bool loopBackEdgeTaken)
{
  if (enableInvarianceCaching_)
  {
    if (const auto it = invariantOutputCache_.find(&output); it != invariantOutputCache_.end())
    {
      // If the query has loopBackEdgeTaken=true but the cached value assumes to loop back-edges
      // have been followed, we must respond that it is not in cache.
      if (loopBackEdgeTaken && !it->second.first)
        return nullptr;

      return it->second.second;
    }
  }

  return nullptr;
}

Output &
traceOutputIntraProcedurally(Output & output, bool mayEnterSubregions)
{
  OutputTracer tracer;
  tracer.setInterprocedural(false);
  tracer.setStructuralNodePolicy(
      mayEnterSubregions ? OutputTracer::StructuralNodePolicy::traceIntoSubregions
                         : OutputTracer::StructuralNodePolicy::traceThroughIfDetectedInvariant);
  return tracer.trace(output);
}

Output &
traceOutput(Output & output, bool mayEnterSubregions, const rvsdg::Region * withinRegion)
{
  OutputTracer tracer;
  tracer.setStructuralNodePolicy(
      mayEnterSubregions ? OutputTracer::StructuralNodePolicy::traceIntoSubregions
                         : OutputTracer::StructuralNodePolicy::traceThroughIfDetectedInvariant);
  tracer.setEnterPhiNodes(mayEnterSubregions);
  return tracer.trace(output, withinRegion);
}

}
