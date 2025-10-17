/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/distribute-constants.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
ConstantDistribution::distributeConstantsInRootRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          distributeConstantsInLambda(lambdaNode);
        },
        [&](rvsdg::PhiNode & phiNode)
        {
          distributeConstantsInRootRegion(*phiNode.subregion());
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        });
  }
}

void
ConstantDistribution::distributeConstantsInLambda(const rvsdg::LambdaNode & lambdaNode)
{
  const auto constants = collectConstants(*lambdaNode.subregion());
  for (const auto constant : constants.Items())
  {
    auto outputs = collectOutputs(*constant);

    std::unordered_map<rvsdg::Region *, rvsdg::Node *> distributedConstants;
    distributedConstants[constant->region()] = constant;

    auto insertAndDivertToNewConstant =
        [&distributedConstants](rvsdg::Output & output, const rvsdg::Node & oldConstant)
    {
      rvsdg::Node * newConstant = nullptr;
      const auto region = output.region();

      const auto it = distributedConstants.find(region);
      if (it == distributedConstants.end())
      {
        newConstant = oldConstant.copy(region, {});
        distributedConstants[region] = newConstant;
      }
      else
      {
        newConstant = it->second;
      }

      output.divert_users(newConstant->output(0));
    };

    for (const auto output : outputs.Items())
    {
      if (rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*output)
          || rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*output)
          || rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*output))
      {
        insertAndDivertToNewConstant(*output, *constant);
      }
      else if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*output))
      {
        // We would like to create constants in every gamma subregion
        auto roleVar = gammaNode->MapBranchArgument(*output);
        if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
        {
          for (const auto argument : entryVar->branchArgument)
          {
            insertAndDivertToNewConstant(*argument, *constant);
          }
        }
      }
      else
      {
        throw std::logic_error("Unhandled output type");
      }
    }
  }
}

util::HashSet<rvsdg::SimpleNode *>
ConstantDistribution::collectConstants(rvsdg::Region & region)
{
  std::function<void(rvsdg::Region &, util::HashSet<rvsdg::SimpleNode *> &)> collect =
      [&collect](rvsdg::Region & region, util::HashSet<rvsdg::SimpleNode *> & constants)
  {
    for (auto & node : region.Nodes())
    {
      rvsdg::MatchTypeOrFail(
          node,
          [&](rvsdg::GammaNode & gammaNode)
          {
            for (auto & subregion : gammaNode.Subregions())
            {
              collect(subregion, constants);
            }
          },
          [&](rvsdg::ThetaNode & thetaNode)
          {
            collect(*thetaNode.subregion(), constants);
          },
          [&constants](rvsdg::SimpleNode & simpleNode)
          {
            if (is_constant(&simpleNode))
            {
              constants.insert(&simpleNode);
            }
          });
    }
  };

  util::HashSet<rvsdg::SimpleNode *> constants;
  collect(region, constants);
  return constants;
}

util::HashSet<rvsdg::Output *>
ConstantDistribution::collectOutputs(const rvsdg::SimpleNode & simpleNode)
{
  JLM_ASSERT(is_constant(&simpleNode));
  JLM_ASSERT(simpleNode.noutputs() == 1);

  std::function<void(rvsdg::Output &, util::HashSet<rvsdg::Output *> &)> collectOutputs =
      [&collectOutputs](rvsdg::Output & output, util::HashSet<rvsdg::Output *> & outputs)
  {
    for (auto & user : output.Users())
    {
      if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(user))
      {
        const auto loopVar = thetaNode->MapInputLoopVar(user);
        if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
        {
          collectOutputs(*loopVar.pre, outputs);
        }
      }
      else if (const auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(user))
      {
        if (&user != thetaNode->predicate())
        {
          const auto loopVar = thetaNode->MapPostLoopVar(user);
          collectOutputs(*loopVar.output, outputs);
        }
      }
      else if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user))
      {
        auto roleVar = gammaNode->MapInput(user);
        if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&roleVar))
        {
          for (const auto argument : entryVar->branchArgument)
          {
            collectOutputs(*argument, outputs);
          }
        }
      }
      else if (const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(user))
      {
        const auto exitVar = gammaNode->MapBranchResultExitVar(user);
        if (rvsdg::GetGammaInvariantOrigin(*gammaNode, exitVar))
        {
          collectOutputs(*exitVar.output, outputs);
        }
      }
      else if (
          rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user)
          || rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(user))
      {
        outputs.insert(&output);
      }
      else
      {
        throw std::logic_error("Unexpected node type");
      }
    }
  };

  util::HashSet<rvsdg::Output *> outputs;
  collectOutputs(*simpleNode.output(0), outputs);
  return outputs;
}

ConstantDistribution::~ConstantDistribution() noexcept = default;

ConstantDistribution::ConstantDistribution()
    : Transformation("ConstantDistribution")
{}

void
ConstantDistribution::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector &)
{
  distributeConstantsInRootRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

}
