/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/distribute-constants.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm
{

void
distribute_constant(const rvsdg::simple_op & op, rvsdg::simple_output * out)
{
  JLM_ASSERT(jlm::hls::is_constant(out->node()));
  bool changed = true;
  while (changed)
  {
    changed = false;
    for (auto user : *out)
    {
      if (auto ti = dynamic_cast<rvsdg::theta_input *>(user))
      {
        auto arg = ti->argument();
        auto res = ti->result();
        if (res->origin() == arg)
        {
          // pass-through
          auto arg_replacement = dynamic_cast<rvsdg::simple_output *>(
              rvsdg::simple_node::create_normalized(ti->node()->subregion(), op, {})[0]);
          ti->argument()->divert_users(arg_replacement);
          ti->output()->divert_users(
              rvsdg::simple_node::create_normalized(out->region(), op, {})[0]);
          distribute_constant(op, arg_replacement);
          arg->region()->RemoveResult(res->index());
          arg->region()->RemoveArgument(arg->index());
          arg->region()->node()->RemoveInput(arg->input()->index());
          arg->region()->node()->RemoveOutput(res->output()->index());
          changed = true;
          break;
        }
      }
      if (auto gi = dynamic_cast<rvsdg::GammaInput *>(user))
      {
        if (gi->node()->predicate() == gi)
        {
          continue;
        }
        for (int i = gi->narguments() - 1; i >= 0; --i)
        {
          if (gi->argument(i)->nusers())
          {
            auto arg_replacement = dynamic_cast<rvsdg::simple_output *>(
                rvsdg::simple_node::create_normalized(gi->argument(i)->region(), op, {})[0]);
            gi->argument(i)->divert_users(arg_replacement);
            distribute_constant(op, arg_replacement);
          }
          gi->node()->subregion(i)->RemoveArgument(gi->argument(i)->index());
        }
        gi->node()->RemoveInput(gi->index());
        changed = true;
        break;
      }
    }
  }
}

void
hls::distribute_constants(rvsdg::region * region)
{
  // push constants down as far as possible, since this is cheaper than having forks and potentially
  // buffers for them
  for (auto & node : rvsdg::topdown_traverser(region))
  {
    if (dynamic_cast<rvsdg::structural_node *>(node))
    {
      if (auto ln = dynamic_cast<llvm::lambda::node *>(node))
      {
        distribute_constants(ln->subregion());
      }
      else if (auto t = dynamic_cast<rvsdg::theta_node *>(node))
      {
        distribute_constants(t->subregion());
      }
      else if (auto gn = dynamic_cast<rvsdg::GammaNode *>(node))
      {
        for (size_t i = 0; i < gn->nsubregions(); ++i)
        {
          distribute_constants(gn->subregion(i));
        }
      }
      else
      {
        throw util::error("Unexpected node type: " + node->operation().debug_string());
      }
    }
    else if (auto sn = dynamic_cast<rvsdg::simple_node *>(node))
    {
      if (is_constant(node))
      {
        distribute_constant(sn->operation(), sn->output(0));
      }
    }
    else
    {
      throw util::error("Unexpected node type: " + node->operation().debug_string());
    }
  }
}

void
hls::distribute_constants(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  distribute_constants(root);
}

} // namespace jlm
