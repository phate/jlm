/*
 * Copyright 2026 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/opt/IOBarrierRedirection.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm
{

IOBarrierRedirection::~IOBarrierRedirection() = default;

IOBarrierRedirection::IOBarrierRedirection()
    : Transformation("IOBarrierRedirection")
{}

// FIXME: copied from NodeSinking pass
static util::HashSet<rvsdg::Node *>
collectDependentNodes(const rvsdg::Node & node)
{
  std::function<void(const rvsdg::Node &, util::HashSet<rvsdg::Node *> &)> collect =
      [&collect](const rvsdg::Node & node, util::HashSet<rvsdg::Node *> & dependentNodes)
  {
    for (auto & output : node.Outputs())
    {
      for (auto & user : output.Users())
      {
        if (const auto userNode = rvsdg::TryGetOwnerNode<rvsdg::Node>(user))
        {
          if (dependentNodes.insert(userNode))
          {
            collect(*userNode, dependentNodes);
          }
        }
      }
    }
  };

  util::HashSet<rvsdg::Node *> dependentNodes;
  collect(node, dependentNodes);
  return dependentNodes;
}

void
IOBarrierRedirection::redirectIOBarrierNode(rvsdg::SimpleNode & ioBarrierNode)
{
  JLM_ASSERT(rvsdg::is<IOBarrierOperation>(ioBarrierNode.GetOperation()));

  auto & barredOperand = *IOBarrierOperation::BarredInput(ioBarrierNode).origin();
  auto & ioStateOperand = *IOBarrierOperation::getIOStateInput(ioBarrierNode).origin();

  const auto gammaNode = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(barredOperand);
  if (!gammaNode)
    return;
  if (!rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(ioStateOperand))
    return;

  auto & barredGammaInput = gammaNode->mapBranchArgumentToInput(barredOperand);
  auto & ioStateGammaInput = gammaNode->mapBranchArgumentToInput(ioStateOperand);

  if (!rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*barredGammaInput.origin()))
    return;
  if (!rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*ioStateGammaInput.origin()))
    return;

  rvsdg::Node * outerIOBarrierNode = nullptr;
  for (auto & user : barredGammaInput.origin()->Users())
  {
    auto [node, ioBarrierOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(user);
    if (ioBarrierOp)
    {
      outerIOBarrierNode = node;
      break;
    }
  }

  if (!outerIOBarrierNode)
    return;

  if (const auto dependentNodes = collectDependentNodes(*outerIOBarrierNode);
      !dependentNodes.Contains(gammaNode))
    return;

  ioBarrierNode.output(0)->divert_users(&barredOperand);
  barredGammaInput.divert_to(outerIOBarrierNode->output(0));
}

void
IOBarrierRedirection::redirectInRegion(rvsdg::Region & region)
{
  std::vector<rvsdg::SimpleNode *> ioBarrierNodes;
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeWithDefault(
        node,
        [&](rvsdg::PhiNode & phiNode)
        {
          redirectInRegion(*phiNode.subregion());
        },
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          redirectInRegion(*lambdaNode.subregion());
        },
        [](rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [&](rvsdg::GammaNode & gammaNode)
        {
          for (auto & subregion : gammaNode.Subregions())
            redirectInRegion(subregion);
        },
        [&](rvsdg::ThetaNode & thetaNode)
        {
          redirectInRegion(*thetaNode.subregion());
        },
        [&](rvsdg::SimpleNode & simpleNode)
        {
          if (rvsdg::is<IOBarrierOperation>(simpleNode.GetOperation()))
          {
            ioBarrierNodes.push_back(&simpleNode);
          }
        },
        []()
        {
          throw std::logic_error("Unhandled node type");
        });
  }

  for (auto & ioBarrierNode : ioBarrierNodes)
    redirectIOBarrierNode(*ioBarrierNode);

  region.prune(false);
}

void
IOBarrierRedirection::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  redirectInRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

}
