/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-triggers.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::hls
{

jlm::rvsdg::output *
get_trigger(rvsdg::Region * region)
{
  for (size_t i = 0; i < region->narguments(); ++i)
  {
    if (region->argument(i)->type() == *hls::triggertype::Create())
    {
      return region->argument(i);
    }
  }
  return nullptr;
}

jlm::llvm::lambda::node *
add_lambda_argument(llvm::lambda::node * ln, std::shared_ptr<const jlm::rvsdg::Type> type)
{
  auto old_fcttype = ln->type();
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> new_argument_types;
  for (size_t i = 0; i < old_fcttype.NumArguments(); ++i)
  {
    new_argument_types.push_back(old_fcttype.Arguments()[i]);
  }
  new_argument_types.push_back(std::move(type));
  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> new_result_types;
  for (size_t i = 0; i < old_fcttype.NumResults(); ++i)
  {
    new_result_types.push_back(old_fcttype.Results()[i]);
  }
  auto new_fcttype = llvm::FunctionType::Create(new_argument_types, new_result_types);
  auto new_lambda = llvm::lambda::node::create(
      ln->region(),
      new_fcttype,
      ln->name(),
      ln->linkage(),
      ln->attributes());

  rvsdg::SubstitutionMap smap;
  for (size_t i = 0; i < ln->ncvarguments(); ++i)
  {
    // copy over cvarguments
    smap.insert(ln->cvargument(i), new_lambda->add_ctxvar(ln->cvargument(i)->input()->origin()));
  }
  for (size_t i = 0; i < ln->nfctarguments(); ++i)
  {
    smap.insert(ln->fctargument(i), new_lambda->fctargument(i));
  }
  //	jlm::rvsdg::view(ln->subregion(), stdout);
  //	jlm::rvsdg::view(new_lambda->subregion(), stdout);
  ln->subregion()->copy(new_lambda->subregion(), smap, false, false);

  std::vector<jlm::rvsdg::output *> new_results;
  for (size_t i = 0; i < ln->nfctresults(); ++i)
  {
    new_results.push_back(smap.lookup(ln->fctresult(i)->origin()));
  }
  auto new_out = new_lambda->finalize(new_results);

  // TODO handle functions at other levels?
  JLM_ASSERT(ln->region() == ln->region()->graph()->root());
  JLM_ASSERT((*ln->output()->begin())->region() == ln->region()->graph()->root());

  //            ln->output()->divert_users(new_out);
  ln->region()->RemoveResult((*ln->output()->begin())->index());
  auto oldExport = ln->ComputeCallSummary()->GetRvsdgExport();
  jlm::llvm::GraphExport::Create(*new_out, oldExport ? oldExport->Name() : "");
  remove(ln);
  return new_lambda;
}

void
add_triggers(rvsdg::Region * region)
{
  auto trigger = get_trigger(region);
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (rvsdg::is<rvsdg::StructuralOperation>(node))
    {
      if (auto ln = dynamic_cast<llvm::lambda::node *>(node))
      {
        // check here in order not to process removed and re-added node twice
        if (!get_trigger(ln->subregion()))
        {
          auto new_lambda = add_lambda_argument(ln, hls::triggertype::Create());
          add_triggers(new_lambda->subregion());
        }
      }
      else if (auto t = dynamic_cast<rvsdg::ThetaNode *>(node))
      {
        JLM_ASSERT(trigger != nullptr);
        JLM_ASSERT(get_trigger(t->subregion()) == nullptr);
        t->add_loopvar(trigger);
        add_triggers(t->subregion());
      }
      else if (auto gn = dynamic_cast<rvsdg::GammaNode *>(node))
      {
        JLM_ASSERT(trigger != nullptr);
        JLM_ASSERT(get_trigger(gn->subregion(0)) == nullptr);
        gn->add_entryvar(trigger);
        for (size_t i = 0; i < gn->nsubregions(); ++i)
        {
          add_triggers(gn->subregion(i));
        }
      }
      else
      {
        throw jlm::util::error("Unexpected node type: " + node->operation().debug_string());
      }
    }
    else if (auto sn = dynamic_cast<jlm::rvsdg::simple_node *>(node))
    {
      JLM_ASSERT(trigger != nullptr);
      if (is_constant(node))
      {
        auto orig_out = sn->output(0);
        std::vector<jlm::rvsdg::input *> previous_users(orig_out->begin(), orig_out->end());
        auto gated = hls::trigger_op::create(*trigger, *orig_out)[0];
        for (auto user : previous_users)
        {
          user->divert_to(gated);
        }
      }
    }
    else
    {
      throw jlm::util::error("Unexpected node type: " + node->operation().debug_string());
    }
  }
}

void
add_triggers(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  add_triggers(root);
}

}
