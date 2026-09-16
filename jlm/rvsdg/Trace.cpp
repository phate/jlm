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
OutputTracer::trace(Output & output, const Region * withinRegion)
{
  // FIXME(perf): Querying the region predication checker class causes repeated re-traversal of
  // the same paths through the region hierarchy that the tracer is already taking.
  // Performance could be improved by adding a way of extracting impossible regions directy.
  regionPredicateTracer_.clearCaches();

  // Mark this as the starting output, which becomes the target for region predicate reachability
  startingOutput_ = &output;
  // Since the current output is the starting output, no back-edges have been followed
  BackEdgeState backEdgeState = BackEdgeState::NoBackEdgeTaken;

  // To disable region predicate checking, always assume back-edges have been taken
  if (!isRegionPredicateCheckingEnabled())
    backEdgeState = BackEdgeState::PossiblyBackEdgeTaken;

  return traceInternal(output, backEdgeState, withinRegion);
}

Output &
OutputTracer::traceInternal(
    Output & output,
    BackEdgeState backEdgeState,
    const Region * withinRegion)
{
  Output * head = &output;

  // Keep tracing until the output stops changing
  while (true)
  {
    Output * prevHead = head;
    head = &traceStep(*head, backEdgeState, withinRegion);
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
OutputTracer::traceGammaOutput(GammaNode & gammaNode, Output & output, BackEdgeState backEdgeState)
{
  // First check the invariance cache
  if (const auto invariantValueInput = lookupInInvarianceCache(output))
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
  if (!isTracingIntoSubregionsEnabled())
  {
    singleInnerOrigin = nullptr;
  }

  for (auto branchResult : exitVar.branchResult)
  {
    // Region predication checking requires that no back-edge has been taken around the gamma
    if (backEdgeState == BackEdgeState::NoBackEdgeTaken)
    {
      // If control flow can not go from the gamma subregion to the region of the starting output,
      // it can not be the origin of the traced value.
      if (!regionPredicateTracer_.isReachableFromRegion(
              *startingOutput_->region(),
              *branchResult->region()))
        continue;
    }

    auto tracedInner = branchResult->origin();

    if (isDeepInvarianceCheckingEnabled())
    {
      // Trace the branch result origin, but only within the gamma subregion
      tracedInner = &traceInternal(*tracedInner, backEdgeState, tracedInner->region());
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

    // Check if the subregion result was traced all the way to an argument
    if (TryGetRegionParentNode<GammaNode>(*tracedInner) == &gammaNode)
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
        // Mismatching outer origins found, given up on finding a common outer origin
        commonOuterOrigin = nullptr;
      }
    }
    else
    {
      // The subregion result could not be traced to an outer origin
      commonOuterOrigin = nullptr;
    }

    // Stop looping through subregions if there is neither an inner origin or a common outer origin
    if (commonOuterOrigin == nullptr && singleInnerOrigin == nullptr)
      return output;
  }

  // If we found a common outer origin, continue tracing from there
  if (commonOuterOrigin.has_value() && *commonOuterOrigin != nullptr)
  {
    JLM_ASSERT(commonGammaInput != nullptr);

    // If the gamma was invariant, even with no assumptions about back-edges not being taken
    // around the gamma, the invariance can be added to the cache
    if (backEdgeState == BackEdgeState::PossiblyBackEdgeTaken)
    {
      return insertInInvarianceCache(output, *commonGammaInput);
    }

    return *commonGammaInput->origin();
  }

  // If only a single gamma subregion provides a possible origin, use it
  if (singleInnerOrigin.has_value() && *singleInnerOrigin != nullptr)
  {
    JLM_ASSERT(isTracingIntoSubregionsEnabled());
    return **singleInnerOrigin;
  }

  // Tracing was unable to make any progress beyond the gamma output
  return output;
}

Output &
OutputTracer::traceThetaOutput(ThetaNode & thetaNode, Output & output, BackEdgeState backEdgeState)
{
  // Lookup the output in the invariance cache
  if (const auto invariantValueInput = lookupInInvarianceCache(output))
  {
    return *invariantValueInput->origin();
  }

  const auto loopVar = thetaNode.MapOutputLoopVar(output);

  auto tracedInner = loopVar.post->origin();

  // If invariance detection is enabled, perform tracing inside the subregion
  if (isDeepInvarianceCheckingEnabled())
  {
    // trace the origin within the thetaNode, but only within the theta's subregion
    tracedInner = &traceInternal(*tracedInner, backEdgeState, thetaNode.subregion());
  }

  // If tracing reached the pre argument of the same loop variable, it might be invariant
  if (tracedInner == loopVar.pre)
  {
    // If the loop variable was found to be invariant,
    // but we also made an assumption about not taking any back-edges around the theta subregion,
    // we must check again without making that assumption to be sure it is acutually invariant.

    if (backEdgeState == BackEdgeState::PossiblyBackEdgeTaken)
    {
      // The tracing already made no assumptions about back-edges.
      // The loop variable is definitely invariant
      return insertInInvarianceCache(output, *loopVar.input);
    }

    // Try tracing from the loop var post again, this time with no assumption
    auto tracedInnerAgain = &traceInternal(
        *loopVar.post->origin(),
        BackEdgeState::PossiblyBackEdgeTaken,
        thetaNode.subregion());
    if (tracedInnerAgain == loopVar.pre)
    {
      // The loop variable is in fact invariant, connect the output to the loop variable input
      return insertInInvarianceCache(output, *loopVar.input);
    }

    // If we get here, it means that the loop variable was only found to be invariant in the final
    // iteration of the loop, but not in every iteration
    JLM_ASSERT(!rvsdg::ThetaLoopVarIsInvariant(loopVar));
  }
  else if (TryGetRegionParentNode<ThetaNode>(*tracedInner) == &thetaNode)
  {
    // Tracing from the post result lead to the pre argument of a different loop variable.
    // Check if that loop variable is trivially invariant, and if it is, return its input origin.

    auto originLoopVar = thetaNode.MapPreLoopVar(*tracedInner);
    if (ThetaLoopVarIsInvariant(originLoopVar))
    {
      return insertInInvarianceCache(output, *originLoopVar.input);
    }
  }

  // If we are allowed to return outputs from inside the subregion,
  // return the result from tracing inside the subregion
  if (isTracingIntoSubregionsEnabled())
  {
    return *tracedInner;
  }

  // Otherwise, we are unable to trace further from the theta output
  return output;
}

Output &
OutputTracer::traceThetaArgument(ThetaNode & thetaNode, Output & output)
{
  // Get the loop variable
  auto loopVar = thetaNode.MapPreLoopVar(output);

  // Trace from the corresponding theta output by following the back-edge
  auto & tracedOutput =
      traceThetaOutput(thetaNode, *loopVar.output, BackEdgeState::PossiblyBackEdgeTaken);

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
OutputTracer::traceStep(Output & output, BackEdgeState backEdgeState, const Region * withinRegion)
{
  if (withinRegion && withinRegion == TryGetOwnerRegion(output))
  {
    // We are not allowed to leave this region, and tracing has reached one of its arguments
    return output;
  }

  // Handle gamma node outputs
  if (const auto gammaNode = TryGetOwnerNode<GammaNode>(output))
  {
    return traceGammaOutput(*gammaNode, output, backEdgeState);
  }

  // Handle gamma node arguments
  if (const auto gammaNode = TryGetRegionParentNode<GammaNode>(output))
  {
    return mapGammaArgumentToOrigin(*gammaNode, output);
  }

  // Handle theta node outputs
  if (const auto thetaNode = TryGetOwnerNode<ThetaNode>(output))
  {
    return traceThetaOutput(*thetaNode, output, backEdgeState);
  }

  // Handle theta node arguments
  if (const auto thetaNode = TryGetRegionParentNode<ThetaNode>(output))
  {
    // The backEdgeState is not provided to this function,
    // since it must anyways immediately follow a back-edge to determine loop invariance
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
OutputTracer::insertInInvarianceCache(const Output & output, Input & traceResult)
{
  if (enableInvarianceCaching_)
  {
    const auto [_, inserted] = invariantOutputCache_.emplace(&output, &traceResult);
    JLM_ASSERT(inserted);
  }

  return *traceResult.origin();
}

Input *
OutputTracer::lookupInInvarianceCache(const Output & output)
{
  if (enableInvarianceCaching_)
  {
    if (const auto it = invariantOutputCache_.find(&output); it != invariantOutputCache_.end())
    {
      return it->second;
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
  tracer.setRegionPredicateCheckingEnabled(mayEnterSubregions);
  return tracer.trace(output);
}

Output &
traceOutput(Output & output, bool mayEnterSubregions, const Region * withinRegion)
{
  OutputTracer tracer;
  tracer.setStructuralNodePolicy(
      mayEnterSubregions ? OutputTracer::StructuralNodePolicy::traceIntoSubregions
                         : OutputTracer::StructuralNodePolicy::traceThroughIfDetectedInvariant);
  tracer.setRegionPredicateCheckingEnabled(mayEnterSubregions);
  tracer.setEnterPhiNodes(mayEnterSubregions);
  return tracer.trace(output, withinRegion);
}

}
