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

void
merge_gamma(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  merge_gamma(root);
}

bool
eliminate_gamma_ctl(rvsdg::GammaNode * gamma)
{
  // eliminates gammas that just replicate the ctl input
  bool changed = false;
  for (size_t i = 0; i < gamma->noutputs(); ++i)
  {
    auto o = gamma->output(i);
    if (rvsdg::is<rvsdg::ControlType>(o->Type()))
    {
      bool eliminate = true;
      for (size_t j = 0; j < gamma->nsubregions(); ++j)
      {
        auto r = gamma->subregion(j)->result(i);
        if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*r->origin()))
        {
          if (auto ctl = dynamic_cast<const rvsdg::ctlconstant_op *>(&simpleNode->GetOperation()))
          {
            if (j == ctl->value().alternative())
            {
              continue;
            }
          }
        }
        eliminate = false;
      }
      if (eliminate)
      {
        if (o->nusers())
        {
          o->divert_users(gamma->predicate()->origin());
          changed = true;
        }
      }
    }
  }
  return changed;
}

bool
bit_type_to_ctl_type(rvsdg::GammaNode * old_gamma)
{
  // for some reason some gamma nodes seem to have bittypes followed by a match instead of ctltypes
  for (size_t i = 0; i < old_gamma->noutputs(); ++i)
  {
    auto o = old_gamma->output(i);
    if (!std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(o->Type()))
      continue;
    if (o->nusers() != 1)
      continue;
    auto & user = *o->Users().begin();
    auto [_, matchOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::MatchOperation>(user);
    if (!matchOperation)
      continue;
    // output is only used by match
    bool all_bittype = true;
    for (size_t j = 0; j < old_gamma->nsubregions(); ++j)
    {
      auto origin = old_gamma->subregion(j)->result(i)->origin();
      if (auto [_, op] =
              rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::IntegerConstantOperation>(*origin);
          !op)
      {
        all_bittype = false;
        break;
      }
    }
    if (!all_bittype)
      continue;
    // actual conversion - instead of copying we just add a new output
    std::vector<rvsdg::Output *> new_outputs;
    for (size_t j = 0; j < old_gamma->nsubregions(); ++j)
    {
      auto origin = old_gamma->subregion(j)->result(i)->origin();
      auto [_, constantOperation] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<llvm::IntegerConstantOperation>(*origin);
      auto ctl_value = matchOperation->alternative(constantOperation->Representation().to_uint());
      auto no = rvsdg::ctlconstant_op::create(
          origin->region(),
          { ctl_value, matchOperation->nalternatives() });
      new_outputs.push_back(no);
    }
    auto match_replacement = old_gamma->AddExitVar(new_outputs).output;
    auto match_node = rvsdg::TryGetOwnerNode<rvsdg::Node>(user);
    match_node->output(0)->divert_users(match_replacement);
    // TODO: divert match users
    remove(match_node);
    return true;
  }
  return false;
}

bool
fix_match_inversion(rvsdg::GammaNode * old_gamma)
{
  // inverts match and swaps regions for gammas that contain swapped control constants
  if (old_gamma->nsubregions() != 2)
  {
    return false;
  }
  bool swapped = false;
  size_t ctl_cnt = 0;
  for (size_t i = 0; i < old_gamma->noutputs(); ++i)
  {
    auto o = old_gamma->output(i);
    if (rvsdg::is<rvsdg::ControlType>(o->Type()))
    {
      ctl_cnt++;
      swapped = true;
      for (size_t j = 0; j < old_gamma->nsubregions(); ++j)
      {
        auto r = old_gamma->subregion(j)->result(i);
        if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*r->origin()))
        {
          if (auto ctl = dynamic_cast<const rvsdg::ctlconstant_op *>(&simpleNode->GetOperation()))
          {
            if (j != ctl->value().alternative())
            {
              continue;
            }
          }
        }
        swapped = false;
      }
    }
  }
  if (ctl_cnt != 1 || !swapped)
  {
    return false;
  }
  if (auto pred_node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*old_gamma->predicate()->origin()))
  {
    if (old_gamma->predicate()->origin()->nusers() != 1)
    {
      return false;
    }
    if (auto match = dynamic_cast<const rvsdg::MatchOperation *>(&pred_node->GetOperation()))
    {
      if (match->nalternatives() == 2)
      {
        uint64_t default_alternative = match->default_alternative() ? 0 : 1;
        auto new_match = rvsdg::MatchOperation::Create(
            *pred_node->input(0)->origin(),
            { { 0, match->alternative(1) }, { 1, match->alternative(0) } },
            default_alternative,
            match->nalternatives());
        auto new_gamma = rvsdg::GammaNode::create(new_match, match->nalternatives());
        rvsdg::SubstitutionMap rmap0; // subregion 0 of the new gamma - 1 of the old
        rvsdg::SubstitutionMap rmap1;
        for (const auto & oev : old_gamma->GetEntryVars())
        {
          auto nev = new_gamma->AddEntryVar(oev.input->origin());
          rmap0.insert(oev.branchArgument[1], nev.branchArgument[0]);
          rmap1.insert(oev.branchArgument[0], nev.branchArgument[1]);
        }
        /* copy subregions */
        old_gamma->subregion(0)->copy(new_gamma->subregion(1), rmap1, false, false);
        old_gamma->subregion(1)->copy(new_gamma->subregion(0), rmap0, false, false);

        for (auto oex : old_gamma->GetExitVars())
        {
          std::vector<rvsdg::Output *> operands;
          operands.push_back(rmap0.lookup(oex.branchResult[1]->origin()));
          operands.push_back(rmap1.lookup(oex.branchResult[0]->origin()));
          auto nex = new_gamma->AddExitVar(operands).output;
          oex.output->divert_users(nex);
        }
        remove(old_gamma);
        remove(pred_node);
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
          if (fix_match_inversion(gamma) || eliminate_gamma_ctl(gamma) || eliminate_gamma_eol(gamma)
              || merge_gamma(gamma) || bit_type_to_ctl_type(gamma))
          {
            changed = true;
            break;
          }
        }
      }
    }
  }
}

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

} // namespace jlm::hls
