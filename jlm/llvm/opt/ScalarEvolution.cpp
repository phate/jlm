/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

class ScalarEvolution::Statistics final : public util::Statistics
{

public:
  ~Statistics() noexcept override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::ScalarEvolution, sourceFile)
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

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

ScalarEvolution::~ScalarEvolution() noexcept = default;

void
ScalarEvolution::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->Start();

  InductionVariableMap_.clear();
  const rvsdg::Region & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();
  TraverseRegion(rootRegion);

  statistics->Stop();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
};

void
ScalarEvolution::TraverseRegion(const rvsdg::Region & region)
{
  for (const auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        TraverseRegion(subregion);
      }
      if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(structuralNode))
      {
        const auto candidates = FindInductionVariables(*thetaNode);
        InductionVariableMap_.emplace(thetaNode, candidates);
      }
    }
  }
}

bool
ScalarEvolution::IsBasedOnInductionVariable(
    const rvsdg::Output & output,
    InductionVariableSet & candidates)
{
  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    // We know the output is a loop variable. Check if the loop variable is in the set, if so return
    // true, otherwise false
    return candidates.Contains(&output);
  }

  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
    {
      return true;
    }
    for (size_t n = 0; n < simpleNode->ninputs(); ++n)
    {
      const auto origin = simpleNode->input(n)->origin();
      if (!IsBasedOnInductionVariable(*origin, candidates))
      {
        return false;
      }
    }
    if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation())
        || rvsdg::is<IntegerSubOperation>(simpleNode->GetOperation()))
    {
      return true;
    }
    return false;
  }
  // TODO: Handle structural nodes
  return false;
}

ScalarEvolution::InductionVariableSet
ScalarEvolution::FindInductionVariables(const rvsdg::ThetaNode & thetaNode)
{
  const std::vector<rvsdg::ThetaNode::LoopVar> loopVars = thetaNode.GetLoopVars();
  // Starting out, all loop variables are induction variable candidates
  InductionVariableSet inductionVariableCandidates{};
  for (const auto & loopVar : loopVars)
  {
    inductionVariableCandidates.insert(loopVar.pre);
  }
  bool changed = false;
  do
  {
    for (const auto & loopVar : loopVars)
    {
      const rvsdg::Output * origin = loopVar.post->origin();
      if (!IsBasedOnInductionVariable(*origin, inductionVariableCandidates))
      {
        changed = inductionVariableCandidates.Remove(loopVar.pre);
      }
    }
  } while (changed == true);

  return inductionVariableCandidates;
}
}
