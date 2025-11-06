/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/PredicateCorrelation.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
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

/**
 * Takes the output of a gamma node and if the output's respective branch results in every
 * subregion originate from a constant, then it returns a vector of the constant
 * alternatives.
 *
 * @param gammaOutput The output of a gamma node.
 * @return The constant alternatives for each of the gamma node's subregion, or
 * std::nullopt;
 */
static std::optional<std::vector<uint64_t>>
extractConstantAlternatives(const rvsdg::Output & gammaOutput)
{
  const auto & gammaNode = rvsdg::AssertGetOwnerNode<rvsdg::GammaNode>(gammaOutput);

  std::vector<uint64_t> alternatives;
  auto [branchResults, _] = gammaNode.MapOutputExitVar(gammaOutput);
  for (const auto branchResult : branchResults)
  {
    {
      auto [constantNode, constantOperation] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::ControlConstantOperation>(
              *branchResult->origin());
      if (constantOperation)
      {
        alternatives.push_back(constantOperation->value().alternative());
        continue;
      }
    }

    {
      auto [constantNode, constantOperation] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::BitConstantOperation>(
              *branchResult->origin());
      if (constantOperation)
      {
        alternatives.push_back(constantOperation->value().to_uint());
        continue;
      }
    }

    {
      auto [constantNode, constantOperation] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(*branchResult->origin());
      if (constantOperation)
      {
        alternatives.push_back(constantOperation->Representation().to_uint());
        continue;
      }
    }

    return std::nullopt;
  }

  return alternatives;
}

static std::optional<std::unique_ptr<ThetaGammaPredicateCorrelation>>
computeControlConstantCorrelation(rvsdg::ThetaNode & thetaNode)
{
  const auto & thetaPredicateOperand = *thetaNode.predicate()->origin();
  auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(thetaPredicateOperand);
  if (!gammaNode)
  {
    return std::nullopt;
  }

  const auto controlAlternativesOpt = extractConstantAlternatives(thetaPredicateOperand);
  if (!controlAlternativesOpt.has_value())
  {
    return std::nullopt;
  }
  const auto controlAlternatives = controlAlternativesOpt.value();

  return ThetaGammaPredicateCorrelation::CreateControlConstantCorrelation(
      thetaNode,
      *gammaNode,
      controlAlternatives);
}

static std::optional<std::unique_ptr<ThetaGammaPredicateCorrelation>>
computeMatchConstantCorrelation(rvsdg::ThetaNode & thetaNode)
{
  auto [matchNode, matchOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::MatchOperation>(*thetaNode.predicate()->origin());
  if (!matchOperation)
  {
    return std::nullopt;
  }

  const auto & gammaOutput = *matchNode->input(0)->origin();
  const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(gammaOutput);
  if (!gammaNode)
  {
    return std::nullopt;
  }

  const auto alternativesOpt = extractConstantAlternatives(gammaOutput);
  if (!alternativesOpt.has_value())
  {
    return std::nullopt;
  }
  const auto alternatives = alternativesOpt.value();

  return ThetaGammaPredicateCorrelation::CreateMatchConstantCorrelation(
      thetaNode,
      *gammaNode,
      { matchNode, alternatives });
}

static std::optional<std::unique_ptr<ThetaGammaPredicateCorrelation>>
computeMatchCorrelation(rvsdg::ThetaNode & thetaNode)
{
  auto [matchNode, matchOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::MatchOperation>(*thetaNode.predicate()->origin());
  if (!matchOperation)
  {
    return std::nullopt;
  }

  for (auto & user : matchNode->output(0)->Users())
  {
    if (const auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user))
    {
      return ThetaGammaPredicateCorrelation::CreateMatchCorrelation(
          thetaNode,
          *gammaNode,
          { matchNode });
    }
  }

  return std::nullopt;
}

std::optional<std::unique_ptr<ThetaGammaPredicateCorrelation>>
computeThetaGammaPredicateCorrelation(rvsdg::ThetaNode & thetaNode)
{
  if (auto correlationOpt = computeControlConstantCorrelation(thetaNode))
  {
    return correlationOpt;
  }

  if (auto correlationOpt = computeMatchConstantCorrelation(thetaNode))
  {
    return correlationOpt;
  }

  if (auto correlationOpt = computeMatchCorrelation(thetaNode))
  {
    return correlationOpt;
  }

  return std::nullopt;
}

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
  // FIXME: Reevaluate the fix-point computation after we introduced gamma-gamma predicate
  // correlation. The pattern is a strict top-down pattern, which means that once we resolved the
  // gamma-gamma predicate correlations, there should only be a single theta-gamma predicate
  // correlation left, if any. Thus, it might be that the fix-point computation is unnecessary.
  bool predicateWasRedirected = false;
  do
  {
    predicateWasRedirected = false;

    const auto correlationOpt = computeThetaGammaPredicateCorrelation(thetaNode);
    if (!correlationOpt.has_value())
    {
      return;
    }
    const auto & correlation = correlationOpt.value();

    switch (correlation->type())
    {
    case CorrelationType::ControlConstantCorrelation:
      predicateWasRedirected = handleControlConstantCorrelation(*correlation);
      break;
    case CorrelationType::MatchConstantCorrelation:
      predicateWasRedirected = handleMatchConstantCorrelation(*correlation);
      break;
    case CorrelationType::MatchCorrelation:
      predicateWasRedirected = false;
      break;
    default:
      throw std::logic_error("Unhandled theta-gamma predicate correlation.");
    }
  } while (predicateWasRedirected);
}

bool
PredicateCorrelation::handleControlConstantCorrelation(
    const ThetaGammaPredicateCorrelation & correlation)
{
  JLM_ASSERT(correlation.type() == CorrelationType::ControlConstantCorrelation);
  const auto & gammaNode = correlation.gammaNode();
  const auto & thetaNode = correlation.thetaNode();

  const auto controlAlternatives =
      std::get<ThetaGammaPredicateCorrelation::ControlConstantCorrelationData>(correlation.data());
  if (controlAlternatives.size() != 2 || controlAlternatives[0] != 0 || controlAlternatives[1] != 1)
  {
    return false;
  }

  thetaNode.predicate()->divert_to(gammaNode.predicate()->origin());
  return true;
}

bool
PredicateCorrelation::handleMatchConstantCorrelation(
    const ThetaGammaPredicateCorrelation & correlation)
{
  JLM_ASSERT(correlation.type() == CorrelationType::MatchConstantCorrelation);
  const auto & gammaNode = correlation.gammaNode();
  const auto & thetaNode = correlation.thetaNode();

  const auto [matchNode, alternatives] =
      std::get<ThetaGammaPredicateCorrelation::MatchConstantCorrelationData>(correlation.data());

  if (alternatives.size() != 2)
  {
    return false;
  }

  const auto matchOperation =
      util::assertedCast<const rvsdg::MatchOperation>(&matchNode->GetOperation());
  if (matchOperation->alternative(alternatives[0]) != 0
      || matchOperation->alternative(alternatives[1]) != 1)
  {
    return false;
  }

  thetaNode.predicate()->divert_to(gammaNode.predicate()->origin());
  return true;
}

void
PredicateCorrelation::Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector &)
{
  correlatePredicatesInRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

}
