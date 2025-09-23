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

#include <iostream>

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
  TraverseGraph(rvsdgModule.Rvsdg());

  statistics->Stop();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
};

void
ScalarEvolution::TraverseRegion(const rvsdg::Region * region)
{
  for (const auto & node : region->Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        TraverseRegion(&subregion);
      }
      if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(structuralNode))
      {
        const auto candidates = FindInductionVariables(thetaNode);
        InductionVariableMap_.emplace(thetaNode, candidates);
      }
    }
  }
}

void
ScalarEvolution::TraverseGraph(const rvsdg::Graph & rvsdg)
{
  const rvsdg::Region & rootRegion = rvsdg.GetRootRegion();
  TraverseRegion(&rootRegion);
}

bool
ScalarEvolution::IsBasedOnInductionVariable(
    const rvsdg::Output * output,
    const rvsdg::ThetaNode * thetaNode)
{
  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*output))
  {
    // We know the output is a loop variable
    const auto input = thetaNode->input(output->index());

    if (const auto simpleNode = rvsdg::TryGetOwnerNode<const rvsdg::SimpleNode>(*input->origin()))
    {
      if (const auto constOp =
              dynamic_cast<const IntegerConstantOperation *>(&simpleNode->GetOperation()))
      {
        std::cout << constOp->Representation().to_uint();
      }
    }
    return true;
  }

  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*output))
  {
    if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
    {
      const auto constOp =
          dynamic_cast<const IntegerConstantOperation *>(&simpleNode->GetOperation());
      const auto value = constOp->Representation().to_uint();
      std::cout << value;
      return true;
    }
    if (rvsdg::is<IntegerMulOperation>(simpleNode->GetOperation())
        || rvsdg::is<IntegerSubOperation>(simpleNode->GetOperation()))
    {
      // TODO: Handle SUB and MUL nodes
      // For now, just return false
      return false;
    }
    if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation()))
    {
      std::cout << "+";
    }
    for (size_t n = 0; n < simpleNode->ninputs(); ++n)
    {
      const auto origin = simpleNode->input(n)->origin();
      if (!IsBasedOnInductionVariable(origin, thetaNode))
      {
        return false;
      }
    }
    return true;
  }
  // TODO: Handle structural nodes
  return false;
}

ScalarEvolution::InductionVariableSet
ScalarEvolution::FindInductionVariables(const rvsdg::ThetaNode * thetaNode)
{
  /*
   * What do we have to do?
   *
   * Traverse the loop and find (basic) induction variables. These are variables with
   * statements of the form i = i + c or i = i - c (increments by loop-invariant expressions). In
   * the future, support for general induction variables (linear combinations of basic induction
   * variables) will be added.
   *
   * We start with all loop variables in a set of induction variable candidates.
   *
   * How can we traverse the loop?
   * We iterate through each loop variable and traverse bottom-up, checking if they are loop
   * invariant. This is done through recursion, by checking for each node whether both of it's
   * inputs are only based on induction variables. If one of the outputs aren't, we return false and
   * remove the loop variable from the set of candidates. If we make it to the top of the theta
   * node, the loop variable is an induction variable and we return true. In this approach constants
   * are seen as trivial induction variables (incremented by 0 each iteration).
   *
   * We run a fixed-point analysis by iterating through all the induction variable candidates until
   * no change occurs in the set. This is done in a do-while loop.
   */

  const std::vector<rvsdg::ThetaNode::LoopVar> loopVars = thetaNode->GetLoopVars();
  // Starting out, all loop variables are induction variable candidates
  InductionVariableSet inductionVariableCandidates{};
  for (const auto & loopVar : loopVars)
  {
    inductionVariableCandidates.Insert(&loopVar);
  }

  bool changed = false;
  do
  {
    for (const auto & loopVar : loopVars)
    {
      const rvsdg::Output * origin = loopVar.post->origin();

      std::cout << loopVar.pre->debug_string() << ": ";
      if (!IsBasedOnInductionVariable(origin, thetaNode))
      {
        changed = inductionVariableCandidates.Remove(&loopVar);
      }
      std::cout << '\n';
    }

  } while (changed == true);

  return inductionVariableCandidates;
}
}
