/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/lambda.hpp>
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
        const auto inductionVariables = FindInductionVariables(*thetaNode);
        InductionVariableMap_.emplace(thetaNode, inductionVariables);
        CreateChainRecurrences(inductionVariables, *thetaNode);
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
    inductionVariableCandidates.Insert(loopVar.pre);
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

std::optional<const SCEV *>
ScalarEvolution::TryGetSCEVForOutput(const rvsdg::Output & output)
{
  if (const auto it = UniqueSCEVs_.find(&output); it != UniqueSCEVs_.end())
    return it->second.get();
  return std::nullopt;
}

std::unique_ptr<SCEV>
ScalarEvolution::GetOrCreateSCEVForOutput(const rvsdg::Output & output)
{
  if (const auto existing = TryGetSCEVForOutput(output))
  {
    return (*existing)->Clone();
  }

  std::unique_ptr<SCEV> result;
  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    // We know this is a loop variable, create a placeholder SCEV for now, and compute the
    // expression later
    result = std::make_unique<SCEVPlaceholder>(output);
  }
  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
    {
      const auto constOp =
          dynamic_cast<const IntegerConstantOperation *>(&simpleNode->GetOperation());
      const auto value = constOp->Representation().to_uint();
      result = std::make_unique<SCEVConstant>(value);
    }
    if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation()))
    {
      assert(simpleNode->ninputs() == 2);
      const auto lhs = simpleNode->input(0)->origin();
      const auto rhs = simpleNode->input(1)->origin();
      result = std::make_unique<SCEVAddExpr>(
          GetOrCreateSCEVForOutput(*lhs),
          GetOrCreateSCEVForOutput(*rhs));
    }
  }
  // TODO: Handle more cases

  if (!result)
    // If none of the cases match, return an unknown SCEV expression
    result = std::make_unique<SCEVUnknown>();

  // Save the result in the cache
  UniqueSCEVs_[&output] = result->Clone();

  return result;
}

void
ScalarEvolution::CreateChainRecurrences(
    const InductionVariableSet & inductionVariables,
    const rvsdg::ThetaNode & thetaNode)
{

  for (const auto indVarPre : inductionVariables.Items())
  {
    const auto loopVar = thetaNode.MapPreLoopVar(*indVarPre);
    const auto indvarPost = loopVar.post;
    auto scev = GetOrCreateSCEVForOutput(*indvarPost->origin());

    std::cout << scev->DebugString() << '\n';
  }
}

}
