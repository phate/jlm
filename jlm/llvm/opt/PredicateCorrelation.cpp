/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/PredicateCorrelation.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

PredicateCorrelation::~PredicateCorrelation() noexcept = default;

void
PredicateCorrelation::correlatePredicatesInRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [](rvsdg::LambdaNode & lambdaNode)
        {
          correlatePredicatesInRegion(*lambdaNode.subregion());
        },
        [](rvsdg::PhiNode & phiNode)
        {
          correlatePredicatesInRegion(*phiNode.subregion());
        },
        [](const rvsdg::DeltaNode &)
        {
          // Nothing needs to be done
        },
        [](rvsdg::ThetaNode & thetaNode)
        {
          // Handle innermost subregions first
          correlatePredicatesInRegion(*thetaNode.subregion());

          correlatePredicatesInTheta(thetaNode);
        },
        [](rvsdg::GammaNode & gammaNode)
        {
          for (auto & subregion : gammaNode.Subregions())
          {
            correlatePredicatesInRegion(subregion);
          }
        },
        [](rvsdg::SimpleNode &)
        {
          // Nothing needs to be done
        });
  }
}

void
PredicateCorrelation::correlatePredicatesInTheta(rvsdg::ThetaNode & thetaNode)
{
  const auto & thetaPredicateOperand = *thetaNode.predicate()->origin();
  const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(thetaPredicateOperand);
  if (!gammaNode)
  {
    return;
  }

  const auto controlAlternativesOpt = extractControlConstantAlternatives(thetaPredicateOperand);
  if (!controlAlternativesOpt.has_value())
  {
    return;
  }
  const auto controlAlternatives = controlAlternativesOpt.value();

  if (controlAlternatives.size() != 2)
  {
    return;
  }

  if (controlAlternatives[0] != 0 || controlAlternatives[1] != 1)
  {
    return;
  }

  thetaNode.predicate()->divert_to(gammaNode->predicate()->origin());
}

std::optional<std::vector<size_t>>
PredicateCorrelation::extractControlConstantAlternatives(const rvsdg::Output & gammaOutput)
{
  const auto & gammaNode = rvsdg::AssertGetOwnerNode<rvsdg::GammaNode>(gammaOutput);

  std::vector<size_t> controlAlternatives;
  auto [branchResults, _] = gammaNode.MapOutputExitVar(gammaOutput);
  for (const auto branchResult : branchResults)
  {
    auto [constantNode, constantOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::ControlConstantOperation>(
            *branchResult->origin());
    if (constantOperation == nullptr)
    {
      return std::nullopt;
    }

    controlAlternatives.push_back(constantOperation->value().alternative());
  }

  return controlAlternatives;
}

void
PredicateCorrelation::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector &)
{
  correlatePredicatesInRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

}
