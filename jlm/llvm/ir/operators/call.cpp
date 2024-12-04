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

using InvariantOutputMap = std::unordered_map<const rvsdg::output *, rvsdg::input *>;

static rvsdg::input *
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

static rvsdg::ThetaInput *
invariantInput(const rvsdg::ThetaOutput & output, InvariantOutputMap & invariantOutputs)
{
  auto origin = output.result()->origin();

  while (true)
  {
    if (origin == output.argument())
    {
      invariantOutputs[&output] = output.input();
      return output.input();
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

static rvsdg::input *
invariantInput(const rvsdg::output & output, InvariantOutputMap & invariantOutputs)
{
  /*
    We already have seen the output, just return the corresponding input.
  */
  if (invariantOutputs.find(&output) != invariantOutputs.end())
    return invariantOutputs[&output];

  if (auto thetaOutput = dynamic_cast<const rvsdg::ThetaOutput *>(&output))
    return invariantInput(*thetaOutput, invariantOutputs);

  if (auto thetaArgument = dynamic_cast<const rvsdg::ThetaArgument *>(&output))
  {
    auto thetaInput = static_cast<const rvsdg::ThetaInput *>(thetaArgument->input());
    return invariantInput(*thetaInput->output(), invariantOutputs);
  }

  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
    return invariantInput(*gamma, output, invariantOutputs);

  return nullptr;
}

static rvsdg::input *
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

rvsdg::Node *
CallNode::copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const
{
  return &CreateNode(*region, GetOperation(), operands);
}

rvsdg::output *
CallNode::TraceFunctionInput(const CallNode & callNode)
{
  auto origin = callNode.GetFunctionInput()->origin();

  while (true)
  {
    if (rvsdg::TryGetOwnerNode<lambda::node>(*origin))
      return origin;

    if (is<rvsdg::GraphImport>(origin))
      return origin;

    if (is<rvsdg::SimpleOperation>(rvsdg::output::GetNode(*origin)))
      return origin;

    if (is<phi::rvargument>(origin))
    {
      return origin;
    }

    if (auto lambda = rvsdg::TryGetRegionParentNode<lambda::node>(*origin))
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
      origin = gamma->MapBranchArgumentEntryVar(*origin).input->origin();
      continue;
    }

    if (auto thetaOutput = dynamic_cast<const rvsdg::ThetaOutput *>(origin))
    {
      if (auto input = invariantInput(*thetaOutput))
      {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (auto thetaArgument = dynamic_cast<const rvsdg::ThetaArgument *>(origin))
    {
      if (auto input = invariantInput(*thetaArgument))
      {
        origin = input->origin();
        continue;
      }

      return origin;
    }

    if (auto phiInputArgument = dynamic_cast<const phi::cvargument *>(origin))
    {
      origin = phiInputArgument->input()->origin();
      continue;
    }

    if (auto rvoutput = dynamic_cast<const phi::rvoutput *>(origin))
    {
      origin = rvoutput->result()->origin();
      continue;
    }

    JLM_UNREACHABLE("We should have never reached this statement.");
  }
}

std::unique_ptr<CallTypeClassifier>
CallNode::ClassifyCall(const CallNode & callNode)
{
  auto output = CallNode::TraceFunctionInput(callNode);

  if (rvsdg::TryGetOwnerNode<lambda::node>(*output))
  {
    return CallTypeClassifier::CreateNonRecursiveDirectCallClassifier(*output);
  }

  if (auto argument = dynamic_cast<rvsdg::RegionArgument *>(output))
  {
    if (is<phi::rvargument>(argument))
    {
      return CallTypeClassifier::CreateRecursiveDirectCallClassifier(*argument);
    }

    if (argument->region() == argument->region()->graph()->root())
    {
      return CallTypeClassifier::CreateExternalCallClassifier(*argument);
    }
  }

  return CallTypeClassifier::CreateIndirectCallClassifier(*output);
}

}
