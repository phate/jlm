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
remove_region_passthrough(const jlm::rvsdg::argument * arg)
{
  auto res = dynamic_cast<jlm::rvsdg::result *>(*arg->begin());
  auto origin = arg->input()->origin();
  // divert users of output to origin of input
  arg->region()->node()->output(res->output()->index())->divert_users(origin);
  // remove result first so argument has no users
  auto inputIndex = arg->input()->index();
  auto outputIndex = res->output()->index();
  auto region = arg->region();
  region->RemoveResult(res->index());
  region->RemoveArgument(arg->index());
  region->node()->RemoveInput(inputIndex);
  region->node()->RemoveOutput(outputIndex);
}

static void
RemoveUnusedStatesInGammaNode(jlm::rvsdg::gamma_node & gammaNode)
{
  for (int i = gammaNode.nentryvars() - 1; i >= 0; --i)
  {
    bool can_remove = true;
    size_t res_index = 0;
    auto arg = gammaNode.subregion(0)->argument(i);
    if (arg->nusers() == 1)
    {
      auto res = dynamic_cast<jlm::rvsdg::result *>(*arg->begin());
      res_index = res ? res->index() : res_index;
    }
    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto sr = gammaNode.subregion(n);
      can_remove &=
          IsPassthroughArgument(*sr->argument(i)) &&
          // check that all subregions pass through to the same result
          dynamic_cast<jlm::rvsdg::result *>(*sr->argument(i)->begin())->index() == res_index;
    }
    if (can_remove)
    {
      auto origin = gammaNode.entryvar(i)->origin();
      // divert users of output to origin of input

      gammaNode.output(res_index)->divert_users(origin);

      for (size_t r = 0; r < gammaNode.nsubregions(); r++)
      {
        gammaNode.subregion(r)->RemoveResult(res_index);
      }
      gammaNode.RemoveOutput(res_index);

      for (size_t r = 0; r < gammaNode.nsubregions(); r++)
      {
        gammaNode.subregion(r)->RemoveArgument(i);
      }
      gammaNode.RemoveInput(i + 1);
    }
  }
}

static void
RemoveUnusedStatesInRegion(rvsdg::region & region);

static void
RemoveUnusedStatesInStructuralNode(rvsdg::structural_node & structuralNode)
{
  // remove unused states from innermost regions first
  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    RemoveUnusedStatesInRegion(*structuralNode.subregion(n));
  }

  if (auto gammaNode = dynamic_cast<rvsdg::gamma_node *>(&structuralNode))
  {
    RemoveUnusedStatesInGammaNode(*gammaNode);
  }
  else if (auto thetaNode = dynamic_cast<rvsdg::theta_node *>(&structuralNode))
  {
    auto thetaSubregion = thetaNode->subregion();
    for (int i = thetaSubregion->narguments() - 1; i >= 0; --i)
    {
      auto arg = thetaSubregion->argument(i);
      if (IsPassthroughArgument(*arg))
      {
        remove_region_passthrough(arg);
      }
    }
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
