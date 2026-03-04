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

OutputTracer::OutputTracer() = default;

Output &
OutputTracer::trace(Output & output)
{
  return trace(output, true);
}

Output &
OutputTracer::trace(Output & output, bool mayLeaveRegion)
{
  Output * head = &output;

  // Keep tracing until the output stops changing
  while (true)
  {
    Output * prevHead = head;
    head = &traceStep(*head, mayLeaveRegion);
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
  const auto roleVar = gammaNode.MapBranchArgument(output);
  if (const auto entryVar = std::get_if<GammaNode::EntryVar>(&roleVar))
  {
    return *entryVar->input->origin();
  }
  if (const auto matchVar = std::get_if<GammaNode::MatchVar>(&roleVar))
  {
    return *matchVar->input->origin();
  }
  throw std::logic_error("Unhandled role variable type.");
}

Output *
OutputTracer::tryTraceThroughGamma(GammaNode & gammaNode, Output & output)
{
  const auto exitVar = gammaNode.MapOutputExitVar(output);

  // The shared output that is the origin of the entry variable(s) going into the gamma node
  Output * commonOrigin = nullptr;

  for (auto branchResult : exitVar.branchResult)
  {
    auto tracedInner = branchResult->origin();

    // If deep tracing is enabled, make a greater effort to trace up to a region argument
    if (traceThroughStrucutalNodes_)
    {
      tracedInner = &trace(*tracedInner, false);
    }

    // The traced output must reach a region argument in the gamma subregion
    if (TryGetRegionParentNode<GammaNode>(*tracedInner) != &gammaNode)
      return nullptr;

    // Get the origin of the region argument outside the gamma
    Output & outerOrigin = mapGammaArgumentToOrigin(gammaNode, *tracedInner);

    // Check that the origin matches with all other origins
    if (commonOrigin == nullptr)
    {
      commonOrigin = &outerOrigin;
    }
    else if (commonOrigin != &outerOrigin)
    {
      return nullptr;
    }
  }

  JLM_ASSERT(commonOrigin != nullptr);
  return commonOrigin;
}

Output *
OutputTracer::tryTraceThroughTheta(ThetaNode & thetaNode, Output & output)
{
  const auto loopVar = thetaNode.MapOutputLoopVar(output);

  auto tracedInner = loopVar.post->origin();

  // If deep tracing is enabled, make a greater effort in tracing up to a region argument
  if (traceThroughStrucutalNodes_)
  {
    tracedInner = &trace(*tracedInner, false);
  }

  if (tracedInner == loopVar.pre)
  {
    return loopVar.input->origin();
  }
  return nullptr;
}

Output &
OutputTracer::traceStep(Output & output, bool mayLeaveRegion)
{
  if (!mayLeaveRegion && TryGetRegionParentNode<Node>(output))
  {
    // We are not allowed to leave the region, so return early on region arguments
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
    // Tracing from inside a theta to outside it is only valid if the loop variable is invariant
    const auto loopVar = thetaNode->MapPreLoopVar(output);
    if (const auto traced = tryTraceThroughTheta(*thetaNode, *loopVar.output))
    {
      return *traced;
    }

    return output;
  }

  // If we are not doing interprocedural tracing, stop tracing now
  if (!isInterprocedural_)
    return output;

  // Handle lambda context variables
  if (const auto lambda = rvsdg::TryGetRegionParentNode<LambdaNode>(output))
  {
    // If the argument is a contex variable, continue tracing
    if (const auto ctxVar = lambda->MapBinderContextVar(output))
      return *ctxVar->input->origin();

    return output;
  }

  // Handle delta context variables
  if (const auto delta = rvsdg::TryGetRegionParentNode<DeltaNode>(output))
  {
    // If the argument is a contex variable, continue tracing
    const auto ctxVar = delta->MapBinderContextVar(output);
    return *ctxVar.input->origin();
  }

  // Handle phi outputs
  if (const auto phiNode = rvsdg::TryGetOwnerNode<PhiNode>(output))
  {
    if (enterPhiNodes_)
    {
      const auto fixVar = phiNode->MapOutputFixVar(output);
      return *fixVar.result->origin();
    }
    return output;
  }

  // Handle phi region arguments
  if (const auto phiNode = rvsdg::TryGetRegionParentNode<PhiNode>(output))
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
traceOutputIntraProcedurally(Output & output)
{
  OutputTracer tracer;
  tracer.setInterprocedural(false);
  return tracer.trace(output);
}

Output &
traceOutput(Output & output)
{
  OutputTracer tracer;
  return tracer.trace(output);
}

}
