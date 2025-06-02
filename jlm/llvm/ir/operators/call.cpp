/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm
{

/**
 * Support functions
 */

using InvariantOutputMap = std::unordered_map<const rvsdg::output *, rvsdg::Input *>;

static rvsdg::Input *
invariantInput(const rvsdg::output & output, InvariantOutputMap & invariantOutputs);

static rvsdg::StructuralInput *
invariantInput(
    const rvsdg::GammaNode & gamma,
    const rvsdg::output & output,
    InvariantOutputMap & invariantOutputs)
{
  size_t n;
  rvsdg::StructuralInput * input = nullptr;
  auto exitvar = gamma.MapOutputExitVar(output);
  for (n = 0; n < exitvar.branchResult.size(); n++)
  {
    auto origin = exitvar.branchResult[n]->origin();

    bool resultIsInvariant = false;
    while (true)
    {
      if (auto argument = dynamic_cast<const rvsdg::RegionArgument *>(origin))
      {
        resultIsInvariant = true;
        input = argument->input();
        break;
      }

      if (auto input = invariantInput(*origin, invariantOutputs))
      {
        origin = input->origin();
        continue;
      }

      break;
    }

    if (resultIsInvariant == false)
      break;
  }

  if (n == exitvar.branchResult.size())
  {
    invariantOutputs[&output] = input;
    return input;
  }

  invariantOutputs[&output] = nullptr;
  return nullptr;
}

static rvsdg::Input *
invariantInput(
    const rvsdg::ThetaNode & theta,
    const rvsdg::output & output,
    InvariantOutputMap & invariantOutputs)
{
  auto loopvar = theta.MapOutputLoopVar(output);

  auto origin = loopvar.post->origin();

  while (true)
  {
    if (origin == loopvar.pre)
    {
      invariantOutputs[&output] = loopvar.input;
      return loopvar.input;
    }

    if (auto input = invariantInput(*origin, invariantOutputs))
    {
      origin = input->origin();
      continue;
    }

    break;
  }

  invariantOutputs[&output] = nullptr;
  return nullptr;
}

static rvsdg::Input *
invariantInput(const rvsdg::output & output, InvariantOutputMap & invariantOutputs)
{
  /*
    We already have seen the output, just return the corresponding input.
  */
  if (invariantOutputs.find(&output) != invariantOutputs.end())
    return invariantOutputs[&output];

  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
    return invariantInput(*theta, output, invariantOutputs);

  if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    auto loopvar = theta->MapPreLoopVar(output);
    return invariantInput(*loopvar.output, invariantOutputs);
  }

  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
    return invariantInput(*gamma, output, invariantOutputs);

  return nullptr;
}

static rvsdg::Input *
invariantInput(const rvsdg::output & output)
{
  InvariantOutputMap invariantOutputs;
  return invariantInput(output, invariantOutputs);
}

/*
 * CallOperation class
 */

CallOperation::~CallOperation() = default;

bool
CallOperation::operator==(const Operation & other) const noexcept
{
  auto callOperation = dynamic_cast<const CallOperation *>(&other);
  return callOperation && FunctionType_ == callOperation->FunctionType_;
}

std::string
CallOperation::debug_string() const
{
  return "CALL";
}

std::unique_ptr<rvsdg::Operation>
CallOperation::copy() const
{
  return std::make_unique<CallOperation>(*this);
}

rvsdg::output *
CallOperation::TraceFunctionInput(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));
  auto origin = GetFunctionInput(callNode).origin();

  while (true)
  {
    if (rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(*origin))
      return origin;

    if (is<rvsdg::GraphImport>(origin))
      return origin;

    if (rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*origin))
      return origin;

    if (auto lambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(*origin))
    {
      if (auto ctxvar = lambda->MapBinderContextVar(*origin))
      {
        origin = ctxvar->input->origin();
        continue;
      }
      else
      {
        return origin;
      }
    }

    if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*origin))
    {
      if (auto input = invariantInput(*origin))
      {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*origin))
    {
      origin = std::visit(
          [](const auto & rolevar) -> rvsdg::output *
          {
            return rolevar.input->origin();
          },
          gamma->MapBranchArgument(*origin));
      continue;
    }

    if (rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*origin))
    {
      if (auto input = invariantInput(*origin))
      {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*origin))
    {
      if (auto input = invariantInput(*origin))
      {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(*origin))
    {
      auto var = phi->MapArgument(*origin);
      if (auto fix = std::get_if<rvsdg::PhiNode::FixVar>(&var))
      {
        return fix->recref;
      }
      else if (auto ctx = std::get_if<rvsdg::PhiNode::ContextVar>(&var))
      {
        origin = ctx->input->origin();
        continue;
      }
      else
      {
        JLM_UNREACHABLE("Phi argument must be either fixpoint or context variable");
      }
    }

    if (auto phi = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(*origin))
    {
      origin = phi->MapOutputFixVar(*origin).result->origin();
      continue;
    }

    JLM_UNREACHABLE("We should have never reached this statement.");
  }
}

std::unique_ptr<CallTypeClassifier>
CallOperation::ClassifyCall(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));
  const auto output = TraceFunctionInput(callNode);

  if (rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(*output))
  {
    return CallTypeClassifier::CreateNonRecursiveDirectCallClassifier(*output);
  }

  if (auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(*output))
  {
    if (auto fix = phi->MapArgumentFixVar(*output))
    {
      return CallTypeClassifier::CreateRecursiveDirectCallClassifier(*output);
    }
  }

  if (auto argument = dynamic_cast<rvsdg::RegionArgument *>(output))
  {
    if (argument->region() == &argument->region()->graph()->GetRootRegion())
    {
      return CallTypeClassifier::CreateExternalCallClassifier(*argument);
    }
  }

  return CallTypeClassifier::CreateIndirectCallClassifier(*output);
}

}
