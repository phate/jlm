/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/LoopStrengthReduction.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{

LoopStrengthReduction::LoopStrengthReduction()
    : Transformation("LoopStrengthReduction")
{}

LoopStrengthReduction::~LoopStrengthReduction() noexcept = default;

void
LoopStrengthReduction::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  ScalarEvolution scalarEvolution;
  scalarEvolution.Run(rvsdgModule, statisticsCollector);

  auto chrecMap = scalarEvolution.GetChrecMap();
  auto scevMap = scalarEvolution.GetSCEVMap();

  ChrecMap_ = std::move(chrecMap);
  SCEVMap_ = std::move(scevMap);

  ProcessRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

void
LoopStrengthReduction::ProcessRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        ProcessRegion(subregion);
      }
      if (const auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(structuralNode))
      {
        ReduceStrength(*thetaNode);
      }
    }
  }
  region.prune(false);
}

void
LoopStrengthReduction::ReduceStrength(rvsdg::ThetaNode & thetaNode)
{
  // General algorithm from prof. Pingali's notes:
  // https://www.cs.utexas.edu/~pingali/CS380C/2019/lectures/strengthReduction.pdf
  // Adapted to work with RVSDG and to make use of the computed recurrences from the scalar
  // evolution analysis.
  //
  // We look for statements that are linear combinations of loop variables and constants
  // that involve multiplication in some way, and replace them with a simple addition.
  //
  // Modifying the RVSDG is done as follows:
  // For a variable j defined as a linear combination of loop variables and constants with
  // chain recurrence {a,+,b}
  //   1. Add a new loop variable s with input value a
  //   2. Create a new constant operation with value b
  //   3. Insert an add-node which takes as inputs the pre value of s and the constant b
  //   4. Divert all the users of j to use the pre value of s
  //   5. Set the post value of s to the output of the new add-node
  //
  // Dead node elimination handles the removal of the dangling node

  // We traverse the nodes in the theta node in a bottom-up manner starting at the origin output of
  // the post values for the loop variables. We look for candidate operations and add them to the
  // stack of operations to be reduced.
  util::HashSet<rvsdg::Output *> candidateOperations;
  util::HashSet<rvsdg::Output *> visited;
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    ProcessOutput(*loopVar.post->origin(), thetaNode, candidateOperations, visited);
  }

  for (auto & output : candidateOperations.Items())
  {
    ReplaceCandidateOperation(*output, thetaNode);
  }
}

void
LoopStrengthReduction::ProcessOutput(
    rvsdg::Output & output,
    rvsdg::ThetaNode & thetaNode,
    util::HashSet<rvsdg::Output *> & candidateOperations,
    util::HashSet<rvsdg::Output *> & visited)
{
  if (!visited.insert(&output))
    return;

  const auto & [simpleNode, operation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerBinaryOperation>(output);

  if (!simpleNode)
    return;

  // Multiplication, addition and subtraction are candidates for strength reduction
  if (rvsdg::is<IntegerMulOperation>(*operation) || rvsdg::is<IntegerAddOperation>(*operation)
      || rvsdg::is<IntegerSubOperation>(*operation))
  {
    if (SCEVMap_.find(&output) == SCEVMap_.end())
      return;

    if (IsValidCandidateOperation(*SCEVMap_.at(&output)))
    {
      candidateOperations.insert(&output);
      return; // Return early to not create unnecessary induction variables for nested operations
    }
  }

  // For non-candidate operations, we traverse through to the node's inputs
  for (auto & input : node->Inputs())
    ProcessOutput(*input.origin(), thetaNode, candidateOperations, visited);
}

void
LoopStrengthReduction::ReplaceCandidateOperation(
    rvsdg::Output & output,
    rvsdg::ThetaNode & thetaNode)
{
  const auto it = ChrecMap_.find(&output);
  if (it == ChrecMap_.end() || !it->second)
    return;

  auto & chrec = it->second;

  if (SCEVChainRecurrence::IsUnknown(*chrec))
    return;

  // For now, we only support affine recurrences. Maybe down the line we can look into supporting
  // quadratic ones
  if (!SCEVChainRecurrence::IsAffine(*chrec))
    return;

  const auto & startSCEV = chrec->GetStartValue();

  const auto & stepPtr = chrec->GetStep();
  if (!stepPtr.has_value())
    return;
  const auto & stepSCEV = *stepPtr;

  const auto & startConstant = dynamic_cast<const SCEVConstant *>(startSCEV);
  const auto & stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get());

  if (!startConstant || !stepConstant)
    return;

  const auto initialValue = startConstant->GetValue();
  const auto stepValue = stepConstant->GetValue();

  const auto & initialValueNode =
      IntegerConstantOperation::Create(*thetaNode.region(), 32, initialValue);
  auto newIV = thetaNode.AddLoopVar(initialValueNode.output(0));

  const auto & stepValueNode =
      IntegerConstantOperation::Create(*thetaNode.subregion(), 32, stepValue);
  const auto & newAddNode =
      jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ newIV.pre, stepValueNode.output(0) }, 32);

  newIV.post->divert_to(newAddNode.output(0));
  output.divert_users(newIV.pre);

  // Insert the chrec for the new induction variable
  // NOTE: This only updates the copy of the original chrec map from the scalar evolution analysis.
  // In the future, we would want to insert this into the "global" chrec map so other analyses and
  // passes can use it as well.
  ChrecMap_[newIV.pre] = std::move(chrec);
}

bool
LoopStrengthReduction::IsValidCandidateOperation(const SCEV & scevTree)
{
  // Accept any linear combination that involves multiplication somewhere
  return IsLinearCombination(scevTree) && ContainsMul(scevTree);
}

bool
LoopStrengthReduction::IsLinearCombination(const SCEV & scev)
{
  if (dynamic_cast<const SCEVConstant *>(&scev))
    return true;
  if (dynamic_cast<const SCEVPlaceholder *>(&scev))
    return true;

  // Adding together two linear combinations results in a new linear combination
  if (const auto add = dynamic_cast<const SCEVAddExpr *>(&scev))
    return IsLinearCombination(*add->GetLeftOperand())
        && IsLinearCombination(*add->GetRightOperand());

  // Check for linear multiplication (constant multiplied by a linear combination)
  // Multiplying a linear combination with a constant creates a new linear combination
  if (const auto mul = dynamic_cast<const SCEVMulExpr *>(&scev))
    return (dynamic_cast<const SCEVConstant *>(mul->GetLeftOperand())
            && IsLinearCombination(*mul->GetRightOperand()))
        || (dynamic_cast<const SCEVConstant *>(mul->GetRightOperand())
            && IsLinearCombination(*mul->GetLeftOperand()));

  return false;
}

bool
LoopStrengthReduction::ContainsMul(const SCEV & scev)
{
  if (dynamic_cast<const SCEVMulExpr *>(&scev))
    return true;

  if (const auto add = dynamic_cast<const SCEVAddExpr *>(&scev))
    return ContainsMul(*add->GetLeftOperand()) || ContainsMul(*add->GetRightOperand());

  return false;
}
}
