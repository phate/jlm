/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/opt/IfConversion.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>

namespace jlm::llvm
{

/** \brief If-Conversion Transformation statistics
 *
 */
class IfConversionStatistics final : public util::Statistics
{
public:
  ~IfConversionStatistics() override = default;

  explicit IfConversionStatistics(const util::FilePath & sourceFile)
      : Statistics(Id::IfConversion, sourceFile)
  {}

  void
  Start() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  Stop() noexcept
  {
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<IfConversionStatistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<IfConversionStatistics>(sourceFile);
  }
};

IfConversion::~IfConversion() noexcept = default;

IfConversion::IfConversion()
    : Transformation("IfConversion")
{}

void
IfConversion::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  auto statistics = IfConversionStatistics::Create(module.SourceFilePath().value());

  statistics->Start();
  HandleRegion(module.Rvsdg().GetRootRegion());
  statistics->Stop();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

void
IfConversion::HandleRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    if (const auto gammaNode = dynamic_cast<const rvsdg::GammaNode *>(&node))
    {
      HandleGammaNode(*gammaNode);
    }
    else if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        const auto subregion = structuralNode->subregion(n);
        HandleRegion(*subregion);
      }
    }
    else if (is<rvsdg::SimpleOperation>(&node))
    {
      // Nothing needs to be done
    }
    else
    {
      JLM_UNREACHABLE("Unsupported node type.");
    }
  }
}

void
IfConversion::HandleGammaNode(const rvsdg::GammaNode & gammaNode)
{
  if (gammaNode.nsubregions() != 2)
    return;

  const auto gammaPredicate = gammaNode.predicate()->origin();
  for (auto [branchResult, gammaOutput] : gammaNode.GetExitVars())
  {
    const auto region0Argument =
        dynamic_cast<const rvsdg::RegionArgument *>(branchResult[0]->origin());
    const auto region1Argument =
        dynamic_cast<const rvsdg::RegionArgument *>(branchResult[1]->origin());

    if (region0Argument == nullptr || region1Argument == nullptr)
    {
      // This output's operands are not just values that are routed through the gamma.
      // Nothing can be done
      continue;
    }

    const auto origin0 = region0Argument->input()->origin();
    const auto origin1 = region1Argument->input()->origin();

    if (origin0 == origin1)
    {
      // Both input operands to the gamma are the same and therefore invariant. No select is needed.
      gammaOutput->divert_users(origin0);
      continue;
    }

    const auto matchNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*gammaPredicate);
    if (is<rvsdg::MatchOperation>(matchNode))
    {
      const auto matchOperation =
          util::assertedCast<const rvsdg::MatchOperation>(&matchNode->GetOperation());
      JLM_ASSERT(matchOperation->nalternatives() == 2);
      JLM_ASSERT(std::distance(matchOperation->begin(), matchOperation->end()) == 1);

      const auto matchOrigin = matchNode->input(0)->origin();
      const auto caseValue = matchOperation->begin()->first;
      const auto caseSubregion = matchOperation->begin()->second;
      const auto defaultSubregion = matchOperation->default_alternative();
      const auto numMatchBits = matchOperation->nbits();
      JLM_ASSERT(caseSubregion != defaultSubregion);

      if (numMatchBits == 1 && caseValue == caseSubregion)
      {
        // We have an identity mapping:
        // 1. 0 -> 0, default 1, or
        // 2. 1 -> 1, default 0
        // There is no need to insert operations for the select predicate
        auto & selectNode = rvsdg::CreateOpNode<SelectOperation>(
            { matchOrigin, origin1, origin0 },
            gammaOutput->Type());
        gammaOutput->divert_users(selectNode.output(0));
      }
      else
      {
        // FIXME: This will recreate the select predicate operations for each gamma output for
        // which we create a select.
        auto & constantNode = rvsdg::CreateOpNode<IntegerConstantOperation>(
            *gammaNode.region(),
            IntegerValueRepresentation(numMatchBits, caseValue));
        auto & eqNode = rvsdg::CreateOpNode<IntegerEqOperation>(
            { constantNode.output(0), matchOrigin },
            numMatchBits);

        auto trueAlternative = caseSubregion == 0 ? origin0 : origin1;
        auto falseAlternative = caseSubregion == 0 ? origin1 : origin0;
        auto & selectNode = rvsdg::CreateOpNode<SelectOperation>(
            { eqNode.output(0), trueAlternative, falseAlternative },
            gammaOutput->Type());
        gammaOutput->divert_users(selectNode.output(0));
      }
    }
    else
    {
      const auto falseAlternative = origin0;
      const auto trueAlternative = origin1;
      auto & controlToIntNode = rvsdg::CreateOpNode<ControlToIntOperation>(
          { gammaPredicate },
          rvsdg::ControlType::Create(2),
          rvsdg::BitType::Create(1));
      auto & selectNode = rvsdg::CreateOpNode<SelectOperation>(
          { controlToIntNode.output(0), trueAlternative, falseAlternative },
          gammaOutput->Type());
      gammaOutput->divert_users(selectNode.output(0));
    }
  }
}

}
