/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm::aa
{

AliasAnalysis::AliasAnalysis() = default;

AliasAnalysis::~AliasAnalysis() noexcept = default;

bool
IsPointerCompatible(const rvsdg::Output & value)
{
  return IsOrContains<PointerType>(*value.Type());
}

const rvsdg::Output &
NormalizeOutput(const rvsdg::Output & output)
{
  if (const auto [node, ioBarrierOp] = rvsdg::TryGetSimpleNodeAndOp<IOBarrierOperation>(output);
      node && ioBarrierOp)
  {
    return NormalizeOutput(*node->input(0)->origin());
  }

  if (rvsdg::TryGetOwnerNode<rvsdg::StructuralNode>(output))
  {
    // If the output is a phi recursion variable, continue tracing inside the phi
    if (const auto phiNode = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(output))
    {
      const auto fixVar = phiNode->MapOutputFixVar(output);
      return NormalizeOutput(*fixVar.result->origin());
    }

    // If the output is a theta output, check if it is invariant
    if (const auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
    {
      const auto loopVar = theta->MapOutputLoopVar(output);
      if (ThetaLoopVarIsInvariant(loopVar))
        return NormalizeOutput(*loopVar.input->origin());
    }

    return output;
  }

  if (const auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    // Follow the gamma input
    auto input = gamma->MapBranchArgument(output);
    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&input))
    {
      return NormalizeOutput(*entryVar->input->origin());
    }

    return output;
  }

  if (const auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    const auto loopVar = theta->GetLoopVars()[output.index()];

    // If the loop variable is invariant, continue normalizing
    if (ThetaLoopVarIsInvariant(loopVar))
      return NormalizeOutput(*loopVar.input->origin());

    return output;
  }

  if (const auto lambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    // If the argument is a contex variable, continue normalizing
    if (const auto ctxVar = lambda->MapBinderContextVar(output))
      return NormalizeOutput(*ctxVar->input->origin());

    return output;
  }

  if (const auto phiNode = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(output))
  {
    const auto argument = phiNode->MapArgument(output);
    if (const auto ctxVar = std::get_if<rvsdg::PhiNode::ContextVar>(&argument))
    {
      // Follow the context variable to outside the phi
      return NormalizeOutput(*ctxVar->input->origin());
    }
    if (const auto fixVar = std::get_if<rvsdg::PhiNode::FixVar>(&argument))
    {
      // Follow to the recursion variable's definition
      return NormalizeOutput(*fixVar->result->origin());
    }

    JLM_UNREACHABLE("Unknown phi argument type");
  }

  return output;
}

}
