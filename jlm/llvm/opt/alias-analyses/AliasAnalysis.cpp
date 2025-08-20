/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/llvm/ir/operators/IntegerOperations.hpp"
#include "jlm/rvsdg/bitstring/constant.hpp"
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

ChainedAliasAnalysis::ChainedAliasAnalysis(AliasAnalysis & first, AliasAnalysis & second)
    : First_(first),
      Second_(second)
{}

ChainedAliasAnalysis::~ChainedAliasAnalysis() = default;

AliasAnalysis::AliasQueryResponse
ChainedAliasAnalysis::Query(
    const rvsdg::Output & p1,
    size_t s1,
    const rvsdg::Output & p2,
    size_t s2)
{
  const auto firstResponse = First_.Query(p1, s1, p2, s2);

  // Anything other than MayAlias is precise, and can be returned right away
  if (firstResponse != MayAlias)
  {
    [[maybe_unused]] AliasQueryResponse opposite = firstResponse == MustAlias ? NoAlias : MustAlias;
    JLM_ASSERT(Second_.Query(p1, s1, p2, s2) != opposite);
    return firstResponse;
  }

  return Second_.Query(p1, s1, p2, s2);
}

std::string
ChainedAliasAnalysis::ToString() const
{
  return util::strfmt("ChainedAA(", First_.ToString(), ",", Second_.ToString(), ")");
}

bool
IsPointerCompatible(const rvsdg::Output & value)
{
  return IsOrContains<PointerType>(*value.Type());
}

const rvsdg::Output &
NormalizeOutput(const rvsdg::Output & output)
{
  if (const auto [node, ioBarrierOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(output);
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

std::optional<int64_t>
TryGetConstantSignedInteger(const rvsdg::Output & output)
{
  const auto & normalized = NormalizeOutput(output);

  if (const auto [_, constant] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(normalized);
      constant)
  {
    const auto & rep = constant->Representation();
    if (rep.is_known() && rep.nbits() <= 64)
      return rep.to_int();
    return std::nullopt;
  }

  if (const auto [_, constant] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::bitconstant_op>(normalized);
      constant)
  {
    const auto & rep = constant->value();
    if (rep.is_known() && rep.nbits() <= 64)
      return rep.to_int();
    return std::nullopt;
  }

  return std::nullopt;
}

}
