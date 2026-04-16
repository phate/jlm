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

OutputTracer::OutputTracer(const bool enableCaching) noexcept
    : enableCaching_(enableCaching)
{}

Output &
OutputTracer::trace(Output & output)
{
  return trace(output, true, nullptr);
}

Output &
OutputTracer::trace(Output & output, rvsdg::Region * targetRegion)
{
  return trace(output, true, targetRegion);
}

Output &
OutputTracer::trace(Output & output, bool mayLeaveRegion, rvsdg::Region * targetRegion)
{
  Output * head = &output;

  // Keep tracing until the output stops changing
  while (true)
  {
    Output * prevHead = head;
    head = &traceStep(*head, mayLeaveRegion, targetRegion);
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

Output *
OutputTracer::tryTraceThroughGamma(GammaNode & gammaNode, Output & output)
{
  if (const auto traceResultOpt = lookupInCache(output); traceResultOpt.has_value())
  {
    return traceResultOpt.value();
  }

  const auto exitVar = gammaNode.MapOutputExitVar(output);

  // The shared output that is the origin of the entry variable(s) going into the gamma node
  Output * commonOrigin = nullptr;
  Input * gammaInput = nullptr;

  for (auto branchResult : exitVar.branchResult)
  {
    auto tracedInner = branchResult->origin();

    // If deep tracing is enabled, make a greater effort to trace up to a region argument
    if (traceThroughStrucutalNodes_)
    {
      tracedInner = &trace(*tracedInner, false, output.region());
    }

    // The traced output must reach a region argument in the gamma subregion
    if (TryGetRegionParentNode<GammaNode>(*tracedInner) != &gammaNode)
      return insertInCache(output, nullptr);

    // Get the origin of the region argument outside the gamma
    gammaInput = &gammaNode.mapBranchArgumentToInput(*tracedInner);
    Output & outerOrigin = *gammaInput->origin();

    // Check that the origin matches with all other origins
    if (commonOrigin == nullptr)
    {
      commonOrigin = &outerOrigin;
    }
    else if (commonOrigin != &outerOrigin)
    {
      return insertInCache(output, nullptr);
    }
  }

  JLM_ASSERT(commonOrigin != nullptr);
  JLM_ASSERT(gammaInput != nullptr);
  return insertInCache(output, gammaInput);
}

Output *
OutputTracer::tryTraceThroughTheta(ThetaNode & thetaNode, Output & output)
{
  if (const auto traceResultOpt = lookupInCache(output); traceResultOpt.has_value())
  {
    return traceResultOpt.value();
  }

  const auto loopVar = thetaNode.MapOutputLoopVar(output);

  auto tracedInner = loopVar.post->origin();

  // If deep tracing is enabled, make a greater effort in tracing up to a region argument
  if (traceThroughStrucutalNodes_)
  {
    tracedInner = &trace(*tracedInner, false, output.region());
  }

  // If tracing reached the pre argument of the same loop variable, it is invariant
  if (tracedInner == loopVar.pre)
  {
    return insertInCache(output, loopVar.input);
  }

  // If tracing from the post result lead to the pre argument of a different loop variable,
  // check if that loop variable is trivially invariant, and if it is, return its input origin.
  if (TryGetRegionParentNode<ThetaNode>(*tracedInner) == &thetaNode)
  {
    auto originLoopVar = thetaNode.MapPreLoopVar(*tracedInner);
    if (ThetaLoopVarIsInvariant(originLoopVar))
    {
      return insertInCache(output, originLoopVar.input);
    }
  }

  return insertInCache(output, nullptr);
}

Output &
OutputTracer::traceStep(Output & output, bool mayLeaveRegion, rvsdg::Region * targetRegion)
{
  if (!mayLeaveRegion && TryGetRegionParentNode<Node>(output))
  {
    // We are not allowed to leave the region, so return early on region arguments
    return output;
  }

  if (targetRegion && output.region() == targetRegion)
  {
    return output;
  }

  // Handle gamma node outputs
  if (const auto gammaNode = TryGetOwnerNode<GammaNode>(output))
  {
    if (const auto traced = tryTraceThroughGamma(*gammaNode, output))
      return *traced;

    return output;
  }

  // Handle gamma node arguments
  if (const auto gammaNode = TryGetRegionParentNode<GammaNode>(output))
  {
    return mapGammaArgumentToOrigin(*gammaNode, output);
  }

  // Handle theta node outputs
  if (const auto thetaNode = TryGetOwnerNode<ThetaNode>(output))
  {
    if (const auto traced = tryTraceThroughTheta(*thetaNode, output))
    {
      return *traced;
    }

    return output;
  }

  // Handle theta node arguments
  if (const auto thetaNode = TryGetRegionParentNode<ThetaNode>(output))
  {
    // Tracing from inside a theta to outside it is only valid if the loop variable is invariant.
    // This is determined by tracing from the loop variable's post result,
    // and seeing if it leads to the same input origin as the loop variable's own input.

    // The loop variable whose pre argument is being traced from
    const auto loopVar = thetaNode->MapPreLoopVar(output);

    // The origin of the loop variable's input.
    const auto inputOrigin = loopVar.input->origin();

    // The origin reached when tracing from the loop variable's post result,
    // if it reaches an invariant loop variable and "escapes" the theta.
    // The invariant loop variable found does not have to be the same as the above loopVar.
    // See TraceTests' TestIndirectLoopInvariance.
    const auto postOrigin = tryTraceThroughTheta(*thetaNode, *loopVar.output);

    if (postOrigin == inputOrigin)
    {
      return *inputOrigin;
    }

    return output;
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

Output *
OutputTracer::insertInCache(const Output & output, Input * traceResult)
{
  if (!enableCaching_)
    return traceResult != nullptr ? traceResult->origin() : nullptr;

  JLM_ASSERT(traceCache_.find(&output) == traceCache_.end());
  traceCache_[&output] = traceResult;
  return traceResult != nullptr ? traceResult->origin() : nullptr;
}

std::optional<Output *>
OutputTracer::lookupInCache(const Output & output)
{
  if (!enableCaching_)
    return std::nullopt;

  if (const auto it = traceCache_.find(&output); it != traceCache_.end())
  {
    return it->second != nullptr ? it->second->origin() : nullptr;
  }

  return std::nullopt;
}

Output &
traceOutputIntraProcedurally(Output & output)
{
  constexpr bool enableCaching = false;
  OutputTracer tracer(enableCaching);
  tracer.setInterprocedural(false);
  return tracer.trace(output);
}

Output &
traceOutput(Output & output, rvsdg::Region * targetRegion)
{
  OutputTracer tracer;
  return tracer.trace(output, targetRegion);
}

Output &
traceOutput(Output & output)
{
  constexpr bool enableCaching = false;
  OutputTracer tracer(enableCaching);
  return tracer.trace(output);
}

}
