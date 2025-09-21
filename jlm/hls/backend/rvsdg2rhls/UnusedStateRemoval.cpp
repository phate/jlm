/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/UnusedStateRemoval.hpp>
#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <algorithm>

namespace jlm::hls
{

static bool
IsPassthroughLoopVar(const rvsdg::ThetaNode::LoopVar & loopVar)
{
  return rvsdg::ThetaLoopVarIsInvariant(loopVar) && loopVar.pre->nusers() == 1;
}

static bool
IsPassthroughArgument(const rvsdg::Output & argument)
{
  if (argument.nusers() != 1)
  {
    return false;
  }

  return rvsdg::is<rvsdg::RegionResult>(*argument.Users().begin());
}

static bool
IsPassthroughResult(const rvsdg::Input & result)
{
  auto argument = dynamic_cast<rvsdg::RegionArgument *>(result.origin());
  return argument != nullptr;
}

static void
RemoveUnusedStatesFromLambda(rvsdg::LambdaNode & lambdaNode)
{
  const auto & op = dynamic_cast<llvm::LlvmLambdaOperation &>(lambdaNode.GetOperation());
  auto & oldFunctionType = op.type();

  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> newArgumentTypes;
  for (size_t i = 0; i < oldFunctionType.NumArguments(); ++i)
  {
    auto argument = lambdaNode.subregion()->argument(i);
    auto argumentType = oldFunctionType.Arguments()[i];
    JLM_ASSERT(*argumentType == *argument->Type());

    if (!IsPassthroughArgument(*argument))
    {
      newArgumentTypes.push_back(argumentType);
    }
  }

  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> newResultTypes;
  for (size_t i = 0; i < oldFunctionType.NumResults(); ++i)
  {
    auto result = lambdaNode.subregion()->result(i);
    auto resultType = oldFunctionType.Results()[i];
    JLM_ASSERT(*resultType == *result->Type());

    if (!IsPassthroughResult(*result))
    {
      newResultTypes.push_back(resultType);
    }
  }

  auto newFunctionType = rvsdg::FunctionType::Create(newArgumentTypes, newResultTypes);
  auto newLambda = rvsdg::LambdaNode::Create(
      *lambdaNode.region(),
      llvm::LlvmLambdaOperation::Create(newFunctionType, op.name(), op.linkage(), op.attributes()));

  rvsdg::SubstitutionMap substitutionMap;
  for (const auto & ctxvar : lambdaNode.GetContextVars())
  {
    auto oldArgument = ctxvar.inner;
    auto origin = ctxvar.input->origin();

    auto newArgument = newLambda->AddContextVar(*origin).inner;
    substitutionMap.insert(oldArgument, newArgument);
  }

  size_t new_i = 0;
  auto newArgs = newLambda->GetFunctionArguments();
  for (auto argument : lambdaNode.GetFunctionArguments())
  {
    if (!IsPassthroughArgument(*argument))
    {
      substitutionMap.insert(argument, newArgs[new_i]);
      new_i++;
    }
  }
  lambdaNode.subregion()->copy(newLambda->subregion(), substitutionMap, false, false);

  std::vector<jlm::rvsdg::Output *> newResults;
  for (auto result : lambdaNode.GetFunctionResults())
  {
    if (!IsPassthroughResult(*result))
    {
      newResults.push_back(substitutionMap.lookup(result->origin()));
    }
  }
  auto newLambdaOutput = newLambda->finalize(newResults);

  // TODO handle functions at other levels?
  JLM_ASSERT(lambdaNode.region() == &lambdaNode.region()->graph()->GetRootRegion());
  JLM_ASSERT(
      (*lambdaNode.output()->Users().begin()).region()
      == &lambdaNode.region()->graph()->GetRootRegion());

  JLM_ASSERT(lambdaNode.output()->nusers() == 1);
  lambdaNode.region()->RemoveResult((*lambdaNode.output()->Users().begin()).index());
  auto oldExport = jlm::llvm::ComputeCallSummary(lambdaNode).GetRvsdgExport();
  rvsdg::GraphExport::Create(*newLambdaOutput, oldExport ? oldExport->Name() : "");
  remove(&lambdaNode);
}

// If this output has a single user and that single user happens to be
// the exit variable of this gamma node, then return it.
static std::optional<rvsdg::GammaNode::ExitVar>
TryGetSingleUserExitVar(rvsdg::GammaNode & gammaNode, rvsdg::Output & argument)
{
  if (argument.nusers() == 1)
  {
    rvsdg::Input * user = &*argument.Users().begin();
    if (rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*user) == &gammaNode)
    {
      return gammaNode.MapBranchResultExitVar(*user);
    }
  }
  return std::nullopt;
}

static void
RemoveUnusedStatesFromGammaNode(rvsdg::GammaNode & gammaNode)
{
  std::vector<rvsdg::GammaNode::EntryVar> deadEntryVars;
  std::vector<rvsdg::GammaNode::ExitVar> deadExitVars;

  for (const auto & entryvar : gammaNode.GetEntryVars())
  {
    std::optional<rvsdg::GammaNode::ExitVar> exitvar0 =
        TryGetSingleUserExitVar(gammaNode, *entryvar.branchArgument[0]);

    bool shouldRemove = exitvar0
                     && std::all_of(
                            entryvar.branchArgument.begin(),
                            entryvar.branchArgument.end(),
                            [&gammaNode, &exitvar0](rvsdg::Output * argument) -> bool
                            {
                              auto exitvar = TryGetSingleUserExitVar(gammaNode, *argument);
                              return exitvar && exitvar->output == exitvar0->output;
                            });

    if (shouldRemove)
    {
      exitvar0->output->divert_users(entryvar.input->origin());
      deadEntryVars.push_back(entryvar);
      deadExitVars.push_back(*exitvar0);
    }
  }

  gammaNode.RemoveExitVars(deadExitVars);
  gammaNode.RemoveEntryVars(deadEntryVars);
}

static void
RemoveUnusedStatesFromThetaNode(rvsdg::ThetaNode & thetaNode)
{
  std::vector<rvsdg::ThetaNode::LoopVar> passthroughLoopVars;
  for (auto & loopVar : thetaNode.GetLoopVars())
  {
    if (IsPassthroughLoopVar(loopVar))
    {
      loopVar.output->divert_users(loopVar.input->origin());
      passthroughLoopVars.emplace_back(loopVar);
    }
  }

  thetaNode.RemoveLoopVars(std::move(passthroughLoopVars));
}

static void
RemoveUnusedStatesInRegion(rvsdg::Region & region);

static void
RemoveUnusedStatesInStructuralNode(rvsdg::StructuralNode & structuralNode)
{
  // Remove unused states from innermost regions first
  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    RemoveUnusedStatesInRegion(*structuralNode.subregion(n));
  }

  if (auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(&structuralNode))
  {
    RemoveUnusedStatesFromGammaNode(*gammaNode);
  }
  else if (auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(&structuralNode))
  {
    RemoveUnusedStatesFromThetaNode(*thetaNode);
  }
  else if (auto lambdaNode = dynamic_cast<rvsdg::LambdaNode *>(&structuralNode))
  {
    RemoveUnusedStatesFromLambda(*lambdaNode);
  }
}

static void
RemoveUnusedStatesInRegion(rvsdg::Region & region)
{
  for (auto & node : rvsdg::TopDownTraverser(&region))
  {
    if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      RemoveUnusedStatesInStructuralNode(*structuralNode);
    }
  }
}

void
RemoveUnusedStates(llvm::RvsdgModule & rvsdgModule)
{
  RemoveUnusedStatesInRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

void
RemoveInvariantLambdaStateEdges(llvm::RvsdgModule & rvsdgModule)
{
  auto & root = rvsdgModule.Rvsdg().GetRootRegion();
  for (auto & node : rvsdg::TopDownTraverser(&root))
  {
    if (auto lambdaNode = dynamic_cast<rvsdg::LambdaNode *>(node))
    {
      RemoveUnusedStatesFromLambda(*lambdaNode);
    }
  }
}

}
