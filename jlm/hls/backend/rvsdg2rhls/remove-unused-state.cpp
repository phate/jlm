/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/remove-unused-state.hpp>
#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

bool
is_passthrough(const rvsdg::input * res)
{
  auto arg = dynamic_cast<rvsdg::RegionArgument *>(res->origin());
  if (arg)
  {
    return true;
  }
  return false;
}

bool
is_passthrough(const rvsdg::output * arg)
{
  if (arg->nusers() == 1)
  {
    auto res = dynamic_cast<rvsdg::RegionResult *>(*arg->begin());
    // used only by a result
    if (res)
    {
      return true;
    }
  }
  return false;
}

jlm::llvm::lambda::node *
remove_lambda_passthrough(llvm::lambda::node * ln)
{
  auto old_fcttype = ln->type();
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> new_argument_types;
  for (size_t i = 0; i < old_fcttype.NumArguments(); ++i)
  {
    auto arg = ln->subregion()->argument(i);
    auto argtype = old_fcttype.Arguments()[i];
    JLM_ASSERT(*argtype == arg->type());
    if (!is_passthrough(arg))
    {
      new_argument_types.push_back(argtype);
    }
  }
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> new_result_types;
  for (size_t i = 0; i < old_fcttype.NumResults(); ++i)
  {
    auto res = ln->subregion()->result(i);
    auto restype = &old_fcttype.ResultType(i);
    JLM_ASSERT(*restype == res->type());
    if (!is_passthrough(res))
    {
      new_result_types.push_back(old_fcttype.Results()[i]);
    }
  }
  auto new_fcttype = rvsdg::FunctionType::Create(new_argument_types, new_result_types);
  auto new_lambda = llvm::lambda::node::create(
      ln->region(),
      new_fcttype,
      ln->name(),
      ln->linkage(),
      ln->attributes());

  rvsdg::SubstitutionMap smap;
  for (const auto & ctxvar : ln->GetContextVars())
  {
    // copy over context vars
    smap.insert(ctxvar.inner, new_lambda->AddContextVar(*ctxvar.input->origin()).inner);
  }

  size_t new_i = 0;
  auto args = ln->GetFunctionArguments();
  auto new_args = new_lambda->GetFunctionArguments();
  JLM_ASSERT(args.size() >= new_args.size());
  for (size_t i = 0; i < args.size(); ++i)
  {
    auto arg = args[i];
    if (!is_passthrough(arg))
    {
      smap.insert(arg, new_args[new_i]);
      new_i++;
    }
  }
  ln->subregion()->copy(new_lambda->subregion(), smap, false, false);

  std::vector<jlm::rvsdg::output *> new_results;
  for (auto res : ln->GetFunctionResults())
  {
    if (!is_passthrough(res))
    {
      new_results.push_back(smap.lookup(res->origin()));
    }
  }
  auto new_out = new_lambda->finalize(new_results);

  // TODO handle functions at other levels?
  JLM_ASSERT(ln->region() == &ln->region()->graph()->GetRootRegion());
  JLM_ASSERT((*ln->output()->begin())->region() == &ln->region()->graph()->GetRootRegion());

  //	ln->output()->divert_users(new_out); // can't divert since the type changed
  JLM_ASSERT(ln->output()->nusers() == 1);
  ln->region()->RemoveResult((*ln->output()->begin())->index());
  auto oldExport = jlm::llvm::ComputeCallSummary(*ln).GetRvsdgExport();
  jlm::llvm::GraphExport::Create(*new_out, oldExport ? oldExport->Name() : "");
  remove(ln);
  return new_lambda;
}

} // namespace jlm::hls
