/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/remove-unused-state.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
remove_unused_state(jlm::rvsdg::region * region, bool can_remove_arguments)
{
  // process children first so that unnecessary users get removed
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      if (auto gn = dynamic_cast<jlm::rvsdg::gamma_node *>(node))
      {
        // process subnodes first
        for (size_t n = 0; n < gn->nsubregions(); n++)
        {
          remove_unused_state(gn->subregion(n), false);
        }
        remove_gamma_passthrough(gn);
      }
      else if (auto ln = dynamic_cast<llvm::lambda::node *>(node))
      {
        remove_unused_state(structnode->subregion(0), false);
        remove_lambda_passthrough(ln);
      }
      else
      {
        assert(structnode->nsubregions() == 1);
        remove_unused_state(structnode->subregion(0));
      }
    }
  }
  if (can_remove_arguments)
  {
    // check if an input is passed through unnecessarily
    for (int i = region->narguments() - 1; i >= 0; --i)
    {
      auto arg = region->argument(i);
      if (is_passthrough(arg))
      {
        remove_region_passthrough(arg);
      }
    }
  }
}

void
remove_unused_state(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  remove_unused_state(root);
}

void
remove_gamma_passthrough(jlm::rvsdg::gamma_node * gn)
{ // remove inputs in reverse
  for (int i = gn->nentryvars() - 1; i >= 0; --i)
  {
    bool can_remove = true;
    size_t res_index = 0;
    auto arg = gn->subregion(0)->argument(i);
    if (arg->nusers() == 1)
    {
      auto res = dynamic_cast<jlm::rvsdg::result *>(*arg->begin());
      res_index = res ? res->index() : res_index;
    }
    for (size_t n = 0; n < gn->nsubregions(); n++)
    {
      auto sr = gn->subregion(n);
      can_remove &=
          is_passthrough(sr->argument(i)) &&
          // check that all subregions pass through to the same result
          dynamic_cast<jlm::rvsdg::result *>(*sr->argument(i)->begin())->index() == res_index;
    }
    if (can_remove)
    {
      auto origin = gn->entryvar(i)->origin();
      // divert users of output to origin of input

      gn->output(res_index)->divert_users(origin);
      gn->output(res_index)->results.clear();
      gn->RemoveOutput(res_index);
      // remove input
      gn->input(i + 1)->arguments.clear();
      gn->RemoveInput(i + 1);
      for (size_t j = 0; j < gn->nsubregions(); ++j)
      {
        JLM_ASSERT(gn->subregion(j)->result(res_index)->origin() == gn->subregion(j)->argument(i));
        JLM_ASSERT(gn->subregion(j)->argument(i)->nusers() == 1);
        gn->subregion(j)->RemoveResult(res_index);
        JLM_ASSERT(gn->subregion(j)->argument(i)->nusers() == 0);
        gn->subregion(j)->RemoveArgument(i);
      }
    }
  }
}

jlm::llvm::lambda::node *
remove_lambda_passthrough(llvm::lambda::node * ln)
{
  auto old_fcttype = ln->type();
  std::vector<const jlm::rvsdg::type *> new_argument_types;
  for (size_t i = 0; i < old_fcttype.NumArguments(); ++i)
  {
    auto arg = ln->subregion()->argument(i);
    auto argtype = &old_fcttype.ArgumentType(i);
    assert(*argtype == arg->type());
    if (!is_passthrough(arg))
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
    if (!is_passthrough(arg))
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

void
remove_region_passthrough(const jlm::rvsdg::argument * arg)
{
  auto res = dynamic_cast<jlm::rvsdg::result *>(*arg->begin());
  auto origin = arg->input()->origin();
  // divert users of output to origin of input
  arg->region()->node()->output(res->output()->index())->divert_users(origin);
  // remove result first so argument has no users
  arg->region()->RemoveResult(res->index());
  arg->region()->RemoveArgument(arg->index());
  arg->region()->node()->RemoveInput(arg->input()->index());
  arg->region()->node()->RemoveOutput(res->output()->index());
}

bool
is_passthrough(const jlm::rvsdg::result * res)
{
  auto arg = dynamic_cast<jlm::rvsdg::argument *>(res->origin());
  if (arg)
  {
    return true;
  }
  return false;
}

bool
is_passthrough(const jlm::rvsdg::argument * arg)
{
  if (arg->nusers() == 1)
  {
    auto res = dynamic_cast<jlm::rvsdg::result *>(*arg->begin());
    // used only by a result
    if (res)
    {
      return true;
    }
  }
  return false;
}

}
