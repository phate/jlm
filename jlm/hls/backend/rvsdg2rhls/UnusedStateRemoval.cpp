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
IsPassthroughArgument(const jlm::rvsdg::argument & argument)
{
  if (argument.nusers() != 1)
  {
    return false;
  }

  return rvsdg::is<rvsdg::result>(**argument.begin());
}

static bool
is_passthrough(const jlm::rvsdg::result * res)
{
  auto arg = dynamic_cast<jlm::rvsdg::argument *>(res->origin());
  if (arg)
  {
    return true;
  }
  return false;
}

static jlm::llvm::lambda::node *
remove_lambda_passthrough(llvm::lambda::node * ln)
{
  auto old_fcttype = ln->type();
  std::vector<const jlm::rvsdg::type *> new_argument_types;
  for (size_t i = 0; i < old_fcttype.NumArguments(); ++i)
  {
    auto arg = ln->subregion()->argument(i);
    auto argtype = &old_fcttype.ArgumentType(i);
    assert(*argtype == arg->type());
    if (!IsPassthroughArgument(*arg))
    {
      new_argument_types.push_back(argtype);
    }
  }
  std::vector<const jlm::rvsdg::type *> new_result_types;
  for (size_t i = 0; i < old_fcttype.NumResults(); ++i)
  {
    auto res = ln->subregion()->result(i);
    auto restype = &old_fcttype.ResultType(i);
    assert(*restype == res->type());
    if (!is_passthrough(res))
    {
      new_result_types.push_back(&old_fcttype.ResultType(i));
    }
  }
  llvm::FunctionType new_fcttype(new_argument_types, new_result_types);
  auto new_lambda = llvm::lambda::node::create(
      ln->region(),
      new_fcttype,
      ln->name(),
      ln->linkage(),
      ln->attributes());

  jlm::rvsdg::substitution_map smap;
  for (size_t i = 0; i < ln->ncvarguments(); ++i)
  {
    // copy over cvarguments
    smap.insert(ln->cvargument(i), new_lambda->add_ctxvar(ln->cvargument(i)->input()->origin()));
  }
  size_t new_i = 0;
  for (size_t i = 0; i < ln->nfctarguments(); ++i)
  {
    auto arg = ln->fctargument(i);
    if (!IsPassthroughArgument(*arg))
    {
      smap.insert(arg, new_lambda->fctargument(new_i));
      new_i++;
    }
  }
  ln->subregion()->copy(new_lambda->subregion(), smap, false, false);

  std::vector<jlm::rvsdg::output *> new_results;
  for (size_t i = 0; i < ln->nfctresults(); ++i)
  {
    auto res = ln->fctresult(i);
    if (!is_passthrough(res))
    {
      new_results.push_back(smap.lookup(res->origin()));
    }
  }
  auto new_out = new_lambda->finalize(new_results);

  // TODO handle functions at other levels?
  assert(ln->region() == ln->region()->graph()->root());
  assert((*ln->output()->begin())->region() == ln->region()->graph()->root());

  //	ln->output()->divert_users(new_out); // can't divert since the type changed
  JLM_ASSERT(ln->output()->nusers() == 1);
  ln->region()->RemoveResult((*ln->output()->begin())->index());
  remove(ln);
  jlm::rvsdg::result::create(new_lambda->region(), new_out, nullptr, new_out->type());
  return new_lambda;
}

static void
RemovePassthroughArgument(const jlm::rvsdg::argument & argument)
{
  auto origin = argument.input()->origin();
  auto result = dynamic_cast<rvsdg::result *>(*argument.begin());
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
RemoveUnusedStatesFromGammaNode(rvsdg::gamma_node & gammaNode)
{
  for (int i = gammaNode.nentryvars() - 1; i >= 0; --i)
  {
    size_t resultIndex = 0;
    auto argument = gammaNode.subregion(0)->argument(i);
    if (argument->nusers() == 1)
    {
      auto result = dynamic_cast<rvsdg::result *>(*argument->begin());
      resultIndex = result ? result->index() : resultIndex;
    }

    bool shouldRemove = true;
    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto subregion = gammaNode.subregion(n);
      shouldRemove &= IsPassthroughArgument(*subregion->argument(i))
                   && dynamic_cast<jlm::rvsdg::result *>(*subregion->argument(i)->begin())->index()
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
RemoveUnusedStatesFromThetaNode(rvsdg::theta_node & thetaNode)
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

  if (auto gammaNode = dynamic_cast<rvsdg::gamma_node *>(&structuralNode))
  {
    RemoveUnusedStatesFromGammaNode(*gammaNode);
  }
  else if (auto thetaNode = dynamic_cast<rvsdg::theta_node *>(&structuralNode))
  {
    RemoveUnusedStatesFromThetaNode(*thetaNode);
  }
  else if (auto lambdaNode = dynamic_cast<llvm::lambda::node *>(&structuralNode))
  {
    remove_lambda_passthrough(lambdaNode);
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
