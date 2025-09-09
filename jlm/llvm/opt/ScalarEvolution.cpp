#include <iostream>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

#include <variant>

namespace jlm::llvm
{

class ScalarEvolution::Statistics final : public util::Statistics
{

public:
  ~Statistics() override = default;

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

void
ScalarEvolution::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->Start();

  TraverseGraph(rvsdgModule.Rvsdg());

  statistics->Stop();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
};

void
ScalarEvolution::TraverseRegion(rvsdg::Region * region)
{
  for (const auto node : rvsdg::TopDownTraverser(region))
  {
    std::cout << node->DebugString() << '\n';
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(node))
    {
      if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(structuralNode))
        FindInductionVariables(thetaNode);
      else
      {
        for (size_t r = 0; r < structuralNode->nsubregions(); r++)
        {
          TraverseRegion(structuralNode->subregion(r));
        }
      }
    }
  }
}

void
ScalarEvolution::TraverseGraph(const rvsdg::Graph & rvsdg)
{
  rvsdg::Region & rootRegion = rvsdg.GetRootRegion();
  TraverseRegion(&rootRegion);
}

void
ScalarEvolution::FindInductionVariables(const rvsdg::ThetaNode * thetaNode)
{
  /*
   * What do we have to do?
   *
   * Traverse the loop and find Basic Induction Variables (BIVs). These are variables with
   * statements of the form i = i + c or i = i - c (increments by loop-invariant expressions), and
   * put them in a set. Traverse the loop again and find General Induction Variables (GIVs)
   * (linear combinations of BIVs) using the BIVs in the set. Put them in the set too.
   *
   * How can we traverse the loop?
   * We can traverse top down, finding multiplication/addition nodes which use loop variables.
   * Start by having all non loop-invariant loop variables in a set of induction variables, while
   * traversing, look for addition nodes. If one of the loop variables is incremented by an
   * expression that is not loop-invariant, remove it from the set.
   * For other binary operations, remove them from the set for now.
   *
   * How do we make it robust?
   * Need to handle different cases:
   * 1. Loop variable and constant (loop variable at both rhs and lhs)
   * 2. Loop variable and another loop variable
   * 3. No loop variables
   *
   * How do we know if the expression is loop invariant?
   * 1. It is a constant -> trivially invariant
   * 2. It is a loop variable -> Check if it is loop invariant using ThetaLoopVarIsInvariant()
   */

  const std::vector<rvsdg::ThetaNode::LoopVar> loopVars = thetaNode->GetLoopVars();
  InductionVariableSet inductionVariableCandidates{};

  // Starting out, all non loop-invariant loop variables are induction variable candidates
  for (auto loopVar : loopVars)
  {
    if (!rvsdg::ThetaLoopVarIsInvariant(loopVar))
    {
      // Only add loop-variant variables to the set, since loop-invariant variables are trivially
      // not induction variables
      inductionVariableCandidates.insert(loopVar.pre);
    }
  }
  for (const auto node : rvsdg::TopDownTraverser(thetaNode->subregion()))
  {
    std::cout << node->DebugString() << '\n';
    if (const auto simpleNode = dynamic_cast<rvsdg::SimpleNode *>(node))
    {
      if (!rvsdg::is<llvm::IntegerBinaryOperation>(simpleNode))
      {
        // TODO: Handle nodes that are not binary integer operations
        continue;
      }
      // Check for basic induction variables (BIV)

      assert(simpleNode->ninputs() == 2); // Assert because this should never happen
      const rvsdg::Input * i0 = simpleNode->input(0);
      const rvsdg::Input * i1 = simpleNode->input(1);

      auto i0LoopVar = TryGetLoopVarFromInput(i0, loopVars);
      auto i1LoopVar = TryGetLoopVarFromInput(i1, loopVars);

      if (!(i0LoopVar || i1LoopVar))
      {
        // No loop variables in computation, go to the next node.
        continue;
      }

      if (rvsdg::is<llvm::IntegerAddOperation>(simpleNode))
      {
        if (i0LoopVar && i1LoopVar)
        {
          // Both are loop variables, need to check if the rhs is loop-invariant.
          // If it is not, the lhs is not an induction variable
          if (!rvsdg::ThetaLoopVarIsInvariant(i1LoopVar.value()))
          {
            inductionVariableCandidates.erase(i0LoopVar.value().pre);
          }
        }
        else if (i0LoopVar && !i1LoopVar)
        {
          const auto i1OriginNode = std::get<rvsdg::Node *>(i1->origin()->GetOwner());

          // If it is not incremented by a constant value, remove it from candidates
          if (!dynamic_cast<const llvm::IntegerConstantOperation *>(
                  &(i1OriginNode->GetOperation())))
            inductionVariableCandidates.erase(i0LoopVar.value().pre);
        }
        else if (!i0LoopVar && i1LoopVar)
        {
          // Since addition is a commutative operation (a + b = b + a), we need to check cases where
          // the rhs is a loop variable as well
          const auto i0OriginNode = std::get<rvsdg::Node *>(i0->origin()->GetOwner());

          if (!dynamic_cast<const llvm::IntegerConstantOperation *>(
                  &(i0OriginNode->GetOperation())))
            inductionVariableCandidates.erase(i1LoopVar.value().pre);
        }
      }
      // TODO: Handle other operations (SUB, MULT)
      else
      {
        // For other binary operations that are not addition

        if (i0LoopVar)
        {
          // For now, just remove these from the candidates
          inductionVariableCandidates.erase(i0LoopVar.value().pre);
        }
      }
    }
  }

  InductionVariableMap_.emplace(thetaNode, inductionVariableCandidates);

  std::cout << "These are the induction variables for the theta loop: ";
  for (const auto & indVar : InductionVariableMap_[thetaNode])
  {
    std::cout << indVar->debug_string() << " ";
  }
  std::cout << std::endl;
}
}
