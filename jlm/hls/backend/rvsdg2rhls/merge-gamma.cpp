/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/merge-gamma.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

bool
is_output_of(jlm::rvsdg::Output * output, rvsdg::Node * node)
{
  auto no = dynamic_cast<rvsdg::NodeOutput *>(output);
  return no && no->node() == node;
}

bool
depends_on(jlm::rvsdg::Output * output, rvsdg::Node * node)
{
  auto arg = dynamic_cast<rvsdg::RegionArgument *>(output);
  if (arg)
  {
    return false;
  }
  auto no = dynamic_cast<rvsdg::NodeOutput *>(output);
  JLM_ASSERT(no);
  if (no->node() == node)
  {
    return true;
  }
  for (size_t i = 0; i < no->node()->ninputs(); ++i)
  {
    if (depends_on(no->node()->input(i)->origin(), node))
    {
      return true;
    }
  }
  return false;
}

rvsdg::GammaNode::EntryVar
get_entryvar(jlm::rvsdg::Output * origin, rvsdg::GammaNode * gamma)
{
  for (auto & user : origin->Users())
  {
    if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user) == gamma)
    {
      auto rolevar = gamma->MapInput(user);
      if (auto entryvar = std::get_if<rvsdg::GammaNode::EntryVar>(&rolevar))
      {
        return *entryvar;
      }
    }
  }
  return gamma->AddEntryVar(origin);
}

bool
merge_gamma(rvsdg::GammaNode * gamma)
{
  for (auto & user : gamma->predicate()->origin()->Users())
  {
    auto other_gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user);
    if (other_gamma && gamma != other_gamma)
    {
      // other gamma depending on same predicate
      JLM_ASSERT(other_gamma->nsubregions() == gamma->nsubregions());
      bool can_merge = true;
      for (const auto & ev : gamma->GetEntryVars())
      {
        // we only merge gammas whose inputs directly, or not at all, depend on the gamma being
        // merged into
        can_merge &= is_output_of(ev.input->origin(), other_gamma)
                  || !depends_on(ev.input->origin(), other_gamma);
      }
      for (const auto & oev : other_gamma->GetEntryVars())
      {
        // prevent cycles
        can_merge &= !depends_on(oev.input->origin(), gamma);
      }
      if (can_merge)
      {
        std::vector<rvsdg::SubstitutionMap> rmap(gamma->nsubregions());
        // populate argument mappings
        for (const auto & ev : gamma->GetEntryVars())
        {
          if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*ev.input->origin()) == other_gamma)
          {
            auto oex = other_gamma->MapOutputExitVar(*ev.input->origin());
            for (size_t j = 0; j < gamma->nsubregions(); ++j)
            {
              rmap[j].insert(ev.branchArgument[j], oex.branchResult[j]->origin());
            }
          }
          else
          {
            auto oev = get_entryvar(ev.input->origin(), other_gamma);
            for (size_t j = 0; j < gamma->nsubregions(); ++j)
            {
              rmap[j].insert(ev.branchArgument[j], oev.branchArgument[j]);
            }
          }
        }
        // copy subregions
        for (size_t j = 0; j < gamma->nsubregions(); ++j)
        {
          gamma->subregion(j)->copy(other_gamma->subregion(j), rmap[j], false, false);
        }
        // handle exitvars
        for (const auto & ex : gamma->GetExitVars())
        {
          std::vector<jlm::rvsdg::Output *> operands;
          for (size_t j = 0; j < ex.branchResult.size(); j++)
          {
            operands.push_back(rmap[j].lookup(ex.branchResult[j]->origin()));
          }
          auto oex = other_gamma->AddExitVar(operands).output;
          ex.output->divert_users(oex);
        }
        remove(gamma);
        return true;
      }
    }
  }
  return false;
}

bool
eliminate_gamma_eol(rvsdg::GammaNode * gamma)
{
  // eliminates gammas that are only active at the end of the loop and have unused outputs
  // seems to be mostly loop variables
  auto theta = dynamic_cast<rvsdg::ThetaNode *>(gamma->region()->node());
  if (!theta || theta->predicate()->origin() != gamma->predicate()->origin())
  {
    return false;
  }
  if (gamma->nsubregions() != 2)
  {
    return false;
  }
  bool changed = false;
  for (size_t i = 0; i < gamma->noutputs(); ++i)
  {
    auto o = gamma->output(i);
    if (o->nusers() != 1)
    {
      continue;
    }
    auto & user = *o->Users().begin();
    if (auto res = dynamic_cast<rvsdg::RegionResult *>(&user))
    {
      if (res->output() && res->output()->nusers() == 0)
      {
        // continue loop subregion
        if (auto arg =
                dynamic_cast<rvsdg::RegionArgument *>(gamma->subregion(1)->result(i)->origin()))
        {
          // value is just passed through
          if (o->nusers())
          {
            o->divert_users(arg->input()->origin());
            changed = true;
          }
        }
      }
    }
  }
  return changed;
}

void
merge_gamma(rvsdg::Region * region)
{
  bool changed = true;
  while (changed)
  {
    changed = false;
    for (auto & node : rvsdg::TopDownTraverser(region))
    {
      if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
      {
        for (size_t n = 0; n < structnode->nsubregions(); n++)
          merge_gamma(structnode->subregion(n));
        if (auto gamma = dynamic_cast<rvsdg::GammaNode *>(node))
        {
          if (eliminate_gamma_eol(gamma) || merge_gamma(gamma))
          {
            changed = true;
            break;
          }
        }
      }
    }
  }
}

GammaMerge::~GammaMerge() noexcept = default;

GammaMerge::GammaMerge()
    : Transformation("GammaMerge")
{}

void
GammaMerge::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector &)
{
  merge_gamma(&rvsdgModule.Rvsdg().GetRootRegion());
}

} // namespace jlm::hls
