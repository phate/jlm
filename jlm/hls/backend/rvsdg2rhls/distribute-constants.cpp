/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/distribute-constants.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
distribute_constant(const rvsdg::SimpleOperation & op, rvsdg::Output * out)
{
  JLM_ASSERT(hls::is_constant(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*out)));

  bool changed = true;
  while (changed)
  {
    changed = false;
    for (auto & user : out->Users())
    {
      if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(user))
      {
        auto loopvar = theta->MapInputLoopVar(user);
        if (loopvar.post->origin() == loopvar.pre)
        {
          // pass-through
          auto arg_replacement =
              rvsdg::SimpleNode::Create(*theta->subregion(), op.copy(), {}).output(0);
          loopvar.pre->divert_users(arg_replacement);
          loopvar.output->divert_users(out);
          distribute_constant(op, arg_replacement);
          theta->RemoveLoopVars({ loopvar });
          changed = true;
          break;
        }
      }
      // push constants that are returned by loops out of them
      if (auto res = dynamic_cast<rvsdg::RegionResult *>(&user))
      {
        auto out = res->output();
        if (out && rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*out))
        {
          if (out->nusers())
          {
            auto out_replacement =
                rvsdg::SimpleNode::Create(*out->node()->region(), op.copy(), {}).output(0);
            out->divert_users(out_replacement);
            distribute_constant(op, out_replacement);
            changed = true;
            break;
          }
        }
      }
      if (auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user))
      {
        auto rolevar = gammaNode->MapInput(user);
        if (std::get_if<rvsdg::GammaNode::MatchVar>(&rolevar))
        {
          // ignore predicate
          continue;
        }
        else if (auto entryvar = std::get_if<rvsdg::GammaNode::EntryVar>(&rolevar))
        {
          for (auto argument : entryvar->branchArgument)
          {
            if (argument->nusers())
            {
              auto arg_replacement =
                  rvsdg::SimpleNode::Create(*argument->region(), op.copy(), {}).output(0);
              argument->divert_users(arg_replacement);
              distribute_constant(op, arg_replacement);
            }
          }
          gammaNode->RemoveEntryVars({ *entryvar });
          changed = true;
          break;
        }
        else
        {
          JLM_UNREACHABLE("Gamma input must either by MatchVar or EntryVar");
        }
      }
    }
  }
}

void
distribute_constants(rvsdg::Region * region)
{
  // push constants down as far as possible, since this is cheaper than having forks and potentially
  // buffers for them
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (rvsdg::is<rvsdg::StructuralOperation>(node))
    {
      if (auto ln = dynamic_cast<rvsdg::LambdaNode *>(node))
      {
        distribute_constants(ln->subregion());
      }
      else if (auto t = dynamic_cast<rvsdg::ThetaNode *>(node))
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
        throw util::Error("Unexpected node type: " + node->DebugString());
      }
    }
    else if (auto sn = dynamic_cast<rvsdg::SimpleNode *>(node))
    {
      if (is_constant(node))
      {
        distribute_constant(sn->GetOperation(), sn->output(0));
      }
    }
    else
    {
      throw util::Error("Unexpected node type: " + node->DebugString());
    }
  }
}

ConstantDistribution::~ConstantDistribution() noexcept = default;

ConstantDistribution::ConstantDistribution()
    : Transformation("ConstantDistribution")
{}

void
ConstantDistribution::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector &)
{
  distribute_constants(&rvsdgModule.Rvsdg().GetRootRegion());
}

} // namespace jlm
