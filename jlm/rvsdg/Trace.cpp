/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/util/common.hpp"
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
  regionPredicateTracer_.clearCaches();

  // Region predication checking skips regions from which control flow can not reach the output
  // This can be disabled by not having a known starting region
  const auto startingRegion = isRegionPredicateCheckingEnabled() ? output.region() : nullptr;

  return traceInternal(output, startingRegion, withinRegion);
}

Output &
OutputTracer::traceInternal(
    Output & output,
    const Region * directlyFromRegion,
    const Region * withinRegion)
{
  Output * head = &output;

  // Keep tracing until the output stops changing
  while (true)
  {
    Output * prevHead = head;
    head = &traceStep(*head, directlyFromRegion, withinRegion);
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
OutputTracer::traceGammaOutput(
    GammaNode & gammaNode,
    Output & output,
    const Region * directlyFromRegion)
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

  for (auto branchResult : exitVar.branchResult)
  {
    // Check if region predicate checking is enabled and lets us skip this subregion
    if (directlyFromRegion)
    {
      // If the current gamma subregion can not reach the starting region for the trace,
      // it can not be the region providing the traced value

      // TODO: Fix const correctness in region predicate tracer and then remove const_cast
      if (!regionPredicateTracer_.isReachableFromRegion(
              const_cast<Region &>(*directlyFromRegion),
              *branchResult->region()))
        continue;
    }

    auto tracedInner = branchResult->origin();

    if (isDeepInvarianceCheckingEnabled())
    {
      // Trace the branch result origin, but only within the gamma subregion
      tracedInner = &traceInternal(*tracedInner, directlyFromRegion, tracedInner->region());
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

    // If the gamma was invariant, even with no assumptions about not taking back-edges
    // around the gamma, the invariance can be added to the cache
    if (!directlyFromRegion)
    {
      return insertInInvarianceCache(output, *commonGammaInput);
    }

    return *commonGammaInput->origin();
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
OutputTracer::traceThetaOutput(
    ThetaNode & thetaNode,
    Output & output,
    const Region * directlyFromRegion)
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
    tracedInner = &traceInternal(*tracedInner, directlyFromRegion, thetaNode.subregion());
  }

  // If tracing reached the pre argument of the same loop variable, it might be invariant
  if (tracedInner == loopVar.pre)
  {
    // If the loop variable was found to be invariant,
    // but we also made an assumption about not taking any back-edges around the theta subregion,
    // we must check again without making that assumption to be sure it is acutually invariant.

    if (!directlyFromRegion)
    {
      // The tracing already made no assumptions about back-edges.
      // The loop variable is definitely invariant
      return insertInInvarianceCache(output, *loopVar.input);
    }

    // Try tracing from the loop var post again, this time with no assumption
    auto tracedInnerAgain = &traceInternal(*loopVar.post->origin(), nullptr, thetaNode.subregion());
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
  // return the result from tracing inside the subregion.
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
  auto & tracedOutput = traceThetaOutput(thetaNode, *loopVar.output, nullptr);

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
OutputTracer::traceStep(
    Output & output,
    const Region * directlyFromRegion,
    const Region * withinRegion)
{
  if (withinRegion && withinRegion == TryGetOwnerRegion(output))
  {
    // We are not allowed to leave this region, and tracing has reached one of its arguments
    return output;
  }

  // Handle gamma node outputs
  if (const auto gammaNode = TryGetOwnerNode<GammaNode>(output))
  {
    return traceGammaOutput(*gammaNode, output, directlyFromRegion);
  }

  // Handle gamma node arguments
  if (const auto gammaNode = TryGetRegionParentNode<GammaNode>(output))
  {
    return mapGammaArgumentToOrigin(*gammaNode, output);
  }

  // Handle theta node outputs
  if (const auto thetaNode = TryGetOwnerNode<ThetaNode>(output))
  {
    return traceThetaOutput(*thetaNode, output, directlyFromRegion);
  }

  // Handle theta node arguments
  if (const auto thetaNode = TryGetRegionParentNode<ThetaNode>(output))
  {
    // When reaching theta arguments this way, it it always because the starting point
    // of the tracing is inside the theta subregion
    JLM_ASSERT(directlyFromRegion != nullptr);
    JLM_ASSERT(Region::isAncestorOrSame(*directlyFromRegion, *thetaNode->subregion()));
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
  return tracer.trace(output);
}

Output &
traceOutput(Output & output, bool mayEnterSubregions, const Region * withinRegion)
{
  OutputTracer tracer;
  tracer.setStructuralNodePolicy(
      mayEnterSubregions ? OutputTracer::StructuralNodePolicy::traceIntoSubregions
                         : OutputTracer::StructuralNodePolicy::traceThroughIfDetectedInvariant);
  tracer.setEnterPhiNodes(mayEnterSubregions);
  return tracer.trace(output, withinRegion);
}

}
