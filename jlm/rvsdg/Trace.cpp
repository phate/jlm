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

  auto result = traceInternal(output, backEdgeState, withinRegion);

  // If the output is statically unreachable, it may not have any origins,
  // so return the same output back
  if (result.isImpossibleOrigin())
    return output;

  JLM_ASSERT(result.isFinalOutput());
  return result.getOutput();
}

OutputTracer::TraceStepResult
OutputTracer::traceInternal(
    Output & output,
    BackEdgeState backEdgeState,
    const Region * withinRegion)
{
  Output * head = &output;

  // Keep tracing until a final result is reached
  while (true)
  {
    const auto traceStepResult = traceStep(*head, backEdgeState, withinRegion);

    // If the tracing reached a final output, or determined no origin exists, stop now
    if (traceStepResult.isFinalOutput() || traceStepResult.isImpossibleOrigin())
      return traceStepResult;

    // Otherwise the result is a tracing step, and it must have made progress
    JLM_ASSERT(traceStepResult.isStepOutput());
    JLM_ASSERT(&traceStepResult.getOutput() != head);

    // Keep tracing from the step result
    head = &traceStepResult.getOutput();
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

OutputTracer::TraceStepResult
OutputTracer::traceGammaOutput(GammaNode & gammaNode, Output & output, BackEdgeState backEdgeState)
{
  // First check the invariance cache
  if (const auto invariantValueInput = lookupInInvarianceCache(output))
  {
    return TraceStepResult::createStepOutput(*invariantValueInput->origin());
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

    auto innerOrigin = branchResult->origin();

    if (isDeepInvarianceCheckingEnabled())
    {
      // Trace the branch result origin, but only within the gamma subregion
      auto traceInnerResult = traceInternal(*innerOrigin, backEdgeState, innerOrigin->region());

      // Tracing inside the subregion only leads to regions that can not reach the starting output
      if (traceInnerResult.isImpossibleOrigin())
        continue;

      innerOrigin = &traceInnerResult.getOutput();
    }

    // Set the single inner origin, or clear it if we already had one
    if (!singleInnerOrigin.has_value())
    {
      singleInnerOrigin = innerOrigin;
    }
    else
    {
      singleInnerOrigin = nullptr;
    }

    // Check if the subregion result was traced all the way to an argument
    if (TryGetRegionParentNode<GammaNode>(*innerOrigin) == &gammaNode)
    {
      // Get the origin of the region argument outside the gamma
      auto & gammaInput = gammaNode.mapBranchArgumentToInput(*innerOrigin);
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
      return TraceStepResult::createFinalOutput(output);
  }

  // All subregions either agree with commonOuterOrigin, or set it to nullptr
  // If it is still nullopt, that means none of the subregions were reachable.
  if (!commonOuterOrigin.has_value())
  {
    return TraceStepResult::createImpossibleOrigin();
  }

  // If we found a common outer origin, continue tracing from there
  if (*commonOuterOrigin != nullptr)
  {
    JLM_ASSERT(commonGammaInput != nullptr);

    // If the gamma was invariant, even with no assumptions about back-edges not being taken
    // around the gamma, the invariance can be added to the cache
    if (backEdgeState == BackEdgeState::PossiblyBackEdgeTaken)
    {
      insertInInvarianceCache(output, *commonGammaInput);
    }

    return TraceStepResult::createStepOutput(*commonGammaInput->origin());
  }

  // If only a single gamma subregion provides a possible origin, use it
  if (singleInnerOrigin.has_value() && *singleInnerOrigin != nullptr)
  {
    JLM_ASSERT(isTracingIntoSubregionsEnabled());

    // The origins found inside subregions have already been fully traced, so they are final
    return TraceStepResult::createFinalOutput(**singleInnerOrigin);
  }

  // Tracing was unable to make any progress beyond the gamma output
  return TraceStepResult::createFinalOutput(output);
}

OutputTracer::TraceStepResult
OutputTracer::traceThetaOutput(ThetaNode & thetaNode, Output & output, BackEdgeState backEdgeState)
{
  // Lookup the output in the invariance cache
  if (const auto invariantValueInput = lookupInInvarianceCache(output))
  {
    return TraceStepResult::createStepOutput(*invariantValueInput->origin());
  }

  const auto loopVar = thetaNode.MapOutputLoopVar(output);

  auto innerOrigin = loopVar.post->origin();

  // If invariance detection is enabled, perform tracing inside the subregion
  if (isDeepInvarianceCheckingEnabled())
  {
    // trace the origin within the thetaNode, but only within the theta's subregion
    auto tracedInnerResult = traceInternal(*innerOrigin, backEdgeState, thetaNode.subregion());

    // If tracing in the theta only reaches regions that can not reach the starting output,
    // the theta itself can not be the origin providing values to the starting output.
    if (tracedInnerResult.isImpossibleOrigin())
      return TraceStepResult::createImpossibleOrigin();

    innerOrigin = &tracedInnerResult.getOutput();
  }

  // If tracing reached the pre argument of the same loop variable, it might be invariant
  if (innerOrigin == loopVar.pre)
  {
    // If the loop variable was found to be invariant,
    // but we also made an assumption about not taking any back-edges around the theta subregion,
    // we must check again without making that assumption to be sure it is acutually invariant.

    if (backEdgeState == BackEdgeState::PossiblyBackEdgeTaken)
    {
      // The tracing already made no assumptions about back-edges.
      // The loop variable is definitely invariant
      return TraceStepResult::createStepOutput(insertInInvarianceCache(output, *loopVar.input));
    }

    // Try tracing from the loop var post again, this time with no assumptions about back-edges
    auto tracedInnerAgain = traceInternal(
        *loopVar.post->origin(),
        BackEdgeState::PossiblyBackEdgeTaken,
        thetaNode.subregion());

    // The loop variable has already been traced once without being impossible.
    // Tracing again with weaker assumptions can never fail.
    JLM_ASSERT(!tracedInnerAgain.isImpossibleOrigin());

    if (&tracedInnerAgain.getOutput() == loopVar.pre)
    {
      // The loop variable is in fact invariant, connect the output to the loop variable input
      return TraceStepResult::createStepOutput(insertInInvarianceCache(output, *loopVar.input));
    }

    // If we get here, it means that the loop variable was only found to be invariant in the final
    // iteration of the loop, but not in every iteration
    JLM_ASSERT(!rvsdg::ThetaLoopVarIsInvariant(loopVar));
  }
  else if (TryGetRegionParentNode<ThetaNode>(*innerOrigin) == &thetaNode)
  {
    // Tracing from the post result lead to the pre argument of a different loop variable.
    // Check if that loop variable is trivially invariant, and if it is, return its input origin.

    auto originLoopVar = thetaNode.MapPreLoopVar(*innerOrigin);
    if (ThetaLoopVarIsInvariant(originLoopVar))
    {
      return TraceStepResult::createStepOutput(
          insertInInvarianceCache(output, *originLoopVar.input));
    }
  }

  // If we are allowed to return outputs from inside the subregion,
  // return the result from tracing inside the subregion
  if (isTracingIntoSubregionsEnabled())
  {
    // The origin found inside the theta is already fully traced, so it is final
    return TraceStepResult::createFinalOutput(*innerOrigin);
  }

  // Otherwise, we are unable to trace further from the theta output
  return TraceStepResult::createFinalOutput(output);
}

OutputTracer::TraceStepResult
OutputTracer::traceThetaArgument(ThetaNode & thetaNode, Output & output)
{
  // Get the loop variable
  auto loopVar = thetaNode.MapPreLoopVar(output);

  // Trace from the corresponding theta output by following the back-edge
  auto outputTraceResult =
      traceThetaOutput(thetaNode, *loopVar.output, BackEdgeState::PossiblyBackEdgeTaken);

  // If the loop output is invariant and has the same origin as the loop variable,
  // tracing can continue outside the theta, so we get a step output result
  if (outputTraceResult.isStepOutput() && &outputTraceResult.getOutput() == loopVar.input->origin())
  {
    return outputTraceResult;
  }

  // Otherwise tracing stops at the theta argument
  return TraceStepResult::createFinalOutput(output);
}

OutputTracer::TraceStepResult
OutputTracer::traceStep(Output & output, BackEdgeState backEdgeState, const Region * withinRegion)
{
  if (withinRegion && withinRegion == TryGetOwnerRegion(output))
  {
    // We are not allowed to leave this region, and tracing has reached one of its arguments
    return TraceStepResult::createFinalOutput(output);
  }

  // Handle gamma node outputs
  if (const auto gammaNode = TryGetOwnerNode<GammaNode>(output))
  {
    return traceGammaOutput(*gammaNode, output, backEdgeState);
  }

  // Handle gamma node arguments
  if (const auto gammaNode = TryGetRegionParentNode<GammaNode>(output))
  {
    return TraceStepResult::createStepOutput(mapGammaArgumentToOrigin(*gammaNode, output));
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
    return TraceStepResult::createFinalOutput(output);

  // Handle lambda context variables
  if (const auto lambda = TryGetRegionParentNode<LambdaNode>(output))
  {
    // If the argument is a contex variable, continue tracing
    if (const auto ctxVar = lambda->MapBinderContextVar(output))
      return TraceStepResult::createStepOutput(*ctxVar->input->origin());

    return TraceStepResult::createFinalOutput(output);
  }

  // Handle delta context variables
  if (const auto delta = TryGetRegionParentNode<DeltaNode>(output))
  {
    // If the argument is a contex variable, continue tracing
    const auto ctxVar = delta->MapBinderContextVar(output);
    return TraceStepResult::createStepOutput(*ctxVar.input->origin());
  }

  // Handle phi outputs
  if (const auto phiNode = TryGetOwnerNode<PhiNode>(output))
  {
    if (enterPhiNodes_)
    {
      const auto fixVar = phiNode->MapOutputFixVar(output);
      return TraceStepResult::createStepOutput(*fixVar.result->origin());
    }
    return TraceStepResult::createFinalOutput(output);
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
      return TraceStepResult::createStepOutput(*ctxVar->input->origin());
    }
    return TraceStepResult::createFinalOutput(output);
  }

  return TraceStepResult::createFinalOutput(output);
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
