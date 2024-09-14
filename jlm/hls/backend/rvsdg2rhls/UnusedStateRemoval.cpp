/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/UnusedStateRemoval.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static bool
IsPassthroughArgument(const rvsdg::RegionArgument & argument)
{
  if (argument.nusers() != 1)
  {
    return false;
  }

  return rvsdg::is<rvsdg::RegionResult>(**argument.begin());
}

static bool
IsPassthroughResult(const rvsdg::RegionResult & result)
{
  auto argument = dynamic_cast<rvsdg::RegionArgument *>(result.origin());
  return argument != nullptr;
}

static void
RemoveUnusedStatesFromLambda(llvm::lambda::node & lambdaNode)
{
  auto & oldFunctionType = lambdaNode.type();

  std::vector<std::shared_ptr<const jlm::rvsdg::type>> newArgumentTypes;
  for (size_t i = 0; i < oldFunctionType.NumArguments(); ++i)
  {
    auto argument = lambdaNode.subregion()->argument(i);
    auto argumentType = oldFunctionType.Arguments()[i];
    JLM_ASSERT(*argumentType == argument->type());

    if (!IsPassthroughArgument(*argument))
    {
      newArgumentTypes.push_back(argumentType);
    }
  }

  std::vector<std::shared_ptr<const jlm::rvsdg::type>> newResultTypes;
  for (size_t i = 0; i < oldFunctionType.NumResults(); ++i)
  {
    auto result = lambdaNode.subregion()->result(i);
    auto resultType = oldFunctionType.Results()[i];
    JLM_ASSERT(*resultType == result->type());

    if (!IsPassthroughResult(*result))
    {
      newResultTypes.push_back(resultType);
    }
  }

  auto newFunctionType = llvm::FunctionType::Create(newArgumentTypes, newResultTypes);
  auto newLambda = llvm::lambda::node::create(
      lambdaNode.region(),
      newFunctionType,
      lambdaNode.name(),
      lambdaNode.linkage(),
      lambdaNode.attributes());

  jlm::rvsdg::substitution_map substitutionMap;
  for (size_t i = 0; i < lambdaNode.ncvarguments(); ++i)
  {
    auto oldArgument = lambdaNode.cvargument(i);
    auto origin = oldArgument->input()->origin();

    auto newArgument = newLambda->add_ctxvar(origin);
    substitutionMap.insert(oldArgument, newArgument);
  }

  size_t new_i = 0;
  for (size_t i = 0; i < lambdaNode.nfctarguments(); ++i)
  {
    auto argument = lambdaNode.fctargument(i);
    if (!IsPassthroughArgument(*argument))
    {
      substitutionMap.insert(argument, newLambda->fctargument(new_i));
      new_i++;
    }
  }
  lambdaNode.subregion()->copy(newLambda->subregion(), substitutionMap, false, false);

  std::vector<jlm::rvsdg::output *> newResults;
  for (size_t i = 0; i < lambdaNode.nfctresults(); ++i)
  {
    auto result = lambdaNode.fctresult(i);
    if (!IsPassthroughResult(*result))
    {
      newResults.push_back(substitutionMap.lookup(result->origin()));
    }
  }
  auto newLambdaOutput = newLambda->finalize(newResults);

  // TODO handle functions at other levels?
  JLM_ASSERT(lambdaNode.region() == lambdaNode.region()->graph()->root());
  JLM_ASSERT((*lambdaNode.output()->begin())->region() == lambdaNode.region()->graph()->root());

  JLM_ASSERT(lambdaNode.output()->nusers() == 1);
  lambdaNode.region()->RemoveResult((*lambdaNode.output()->begin())->index());
  auto oldExport = lambdaNode.ComputeCallSummary()->GetRvsdgExport();
  jlm::llvm::GraphExport::Create(*newLambdaOutput, oldExport ? oldExport->Name() : "");
  remove(&lambdaNode);
}

static void
RemovePassthroughArgument(const rvsdg::RegionArgument & argument)
{
  auto origin = argument.input()->origin();
  auto result = dynamic_cast<rvsdg::RegionResult *>(*argument.begin());
  argument.region()->node()->output(result->output()->index())->divert_users(origin);

  auto inputIndex = argument.input()->index();
  auto outputIndex = result->output()->index();
  auto region = argument.region();
  region->RemoveResult(result->index());
  region->RemoveArgument(argument.index());
  region->node()->RemoveInput(inputIndex);
  region->node()->RemoveOutput(outputIndex);
}

static void
RemoveUnusedStatesFromGammaNode(rvsdg::GammaNode & gammaNode)
{
  for (int i = gammaNode.nentryvars() - 1; i >= 0; --i)
  {
    size_t resultIndex = 0;
    auto argument = gammaNode.subregion(0)->argument(i);
    if (argument->nusers() == 1)
    {
      auto result = dynamic_cast<rvsdg::RegionResult *>(*argument->begin());
      resultIndex = result ? result->index() : resultIndex;
    }

    bool shouldRemove = true;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto subregion = gammaNode.subregion(n);
      shouldRemove &=
          IsPassthroughArgument(*subregion->argument(i))
          && dynamic_cast<jlm::rvsdg::RegionResult *>(*subregion->argument(i)->begin())->index()
                 == resultIndex;
    }

    if (shouldRemove)
    {
      auto origin = gammaNode.entryvar(i)->origin();
      gammaNode.output(resultIndex)->divert_users(origin);

      for (size_t r = 0; r < gammaNode.nsubregions(); r++)
      {
        gammaNode.subregion(r)->RemoveResult(resultIndex);
      }
      gammaNode.RemoveOutput(resultIndex);

      for (size_t r = 0; r < gammaNode.nsubregions(); r++)
      {
        gammaNode.subregion(r)->RemoveArgument(i);
      }
      gammaNode.RemoveInput(i + 1);
    }
  }
}

static void
RemoveUnusedStatesFromThetaNode(rvsdg::ThetaNode & thetaNode)
{
  auto thetaSubregion = thetaNode.subregion();
  for (int i = thetaSubregion->narguments() - 1; i >= 0; --i)
  {
    auto & argument = *thetaSubregion->argument(i);
    if (IsPassthroughArgument(argument))
    {
      RemovePassthroughArgument(argument);
    }
  }
}

static void
RemoveUnusedStatesInRegion(rvsdg::region & region);

static void
RemoveUnusedStatesInStructuralNode(rvsdg::structural_node & structuralNode)
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
  else if (auto lambdaNode = dynamic_cast<llvm::lambda::node *>(&structuralNode))
  {
    RemoveUnusedStatesFromLambda(*lambdaNode);
  }
}

static void
RemoveUnusedStatesInRegion(rvsdg::region & region)
{
  for (auto & node : rvsdg::topdown_traverser(&region))
  {
    if (auto structuralNode = dynamic_cast<rvsdg::structural_node *>(node))
    {
      RemoveUnusedStatesInStructuralNode(*structuralNode);
    }
  }
}

void
RemoveUnusedStates(llvm::RvsdgModule & rvsdgModule)
{
  RemoveUnusedStatesInRegion(*rvsdgModule.Rvsdg().root());
}

}
