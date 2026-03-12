/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/LoopStrengthReduction.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
class LoopStrengthReduction::Context final
{
public:
  ~Context() = default;

  Context() = default;

  Context(const Context &) = delete;

  Context(Context &&) = delete;

  Context &
  operator=(const Context &) = delete;

  Context &
  operator=(Context &&) = delete;

  static std::unique_ptr<Context>
  Create()
  {
    return std::make_unique<Context>();
  }

  void
  IncrementOperationsReducedCount(const rvsdg::ThetaNode & theta)
  {
    JLM_ASSERT(OperationsReducedMap_.find(&theta) != OperationsReducedMap_.end());
    OperationsReducedMap_[&theta]++;
  }

  void
  InsertIntoOperationsReducedMap(const rvsdg::ThetaNode & theta)
  {
    OperationsReducedMap_.insert({ &theta, 0 });
  }

  const std::unordered_map<const rvsdg::ThetaNode *, size_t> &
  GetOperationsReducedMap() const
  {
    return OperationsReducedMap_;
  }

private:
  std::unordered_map<const rvsdg::ThetaNode *, size_t> OperationsReducedMap_;
};

class LoopStrengthReduction::Statistics final : public util::Statistics
{

public:
  ~Statistics() noexcept override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::LoopStrengthReduction, sourceFile)
  {}

  void
  Start() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  Stop(const Context & context) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(
        Label::NumOperationsReduced,
        GetStatisticsString(context.GetOperationsReducedMap()));
  }

  static std::string
  GetStatisticsString(
      const std::unordered_map<const rvsdg::ThetaNode *, size_t> & operationsReducedMap)
  {
    std::string s = "";
    bool first = true;
    size_t totalCount = 0;
    for (auto & [thetaNode, operationsReduced] : operationsReducedMap)
    {
      totalCount += operationsReduced;
      if (!first)
        s += ',';
      first = false;

      s += "ID(" + std::to_string(thetaNode->subregion()->getRegionId())
         + ")=" + std::to_string(operationsReduced);
    }
    s += ",Total=" + std::to_string(totalCount);
    return s;
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

LoopStrengthReduction::LoopStrengthReduction()
    : Transformation("LoopStrengthReduction")
{}

LoopStrengthReduction::~LoopStrengthReduction() noexcept = default;

void
LoopStrengthReduction::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->Start();

  ScalarEvolution scalarEvolution;
  scalarEvolution.Run(rvsdgModule, statisticsCollector);

  Context_ = Context::Create();

  ChrecMap_ = scalarEvolution.GetChrecMap();
  SCEVMap_ = scalarEvolution.GetSCEVMap();

  ProcessRegion(rvsdgModule.Rvsdg().GetRootRegion());

  statistics->Stop(*Context_);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
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
        Context_->InsertIntoOperationsReducedMap(*thetaNode);
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
  // More precisely, a candidate operation must satisfy two conditions:
  //   1. Its SCEV tree contains at least one multiplication node (SCEVMulExpr),
  //      ensuring we only reduce operations that actually benefit from strength reduction.
  //   2. Its SCEV evaluates to a chain recurrence of the form {a,+,b}, meaning the value
  //      can be expressed as a start value plus a constant step per iteration. This is what
  //      makes it a valid induction variable candidate.
  //
  // Modifying the RVSDG is done as follows:
  // For a variable j defined as a linear combination of loop variables and constants with
  // chain recurrence {a,+,b} where b can also be 0 (invariant):
  //   1. Add a new loop variable s with input value a
  //   2. Divert all the users of j to use the pre value of s
  // If b is not 0:
  //   3. Create a new constant operation with value b
  //   4. Insert an add-node which takes as inputs the pre value of s and the constant b
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
  for (auto & input : simpleNode->Inputs())
    ProcessOutput(*input.origin(), thetaNode, candidateOperations, visited);
}

void
LoopStrengthReduction::ReplaceCandidateOperation(
    rvsdg::Output & output,
    rvsdg::ThetaNode & thetaNode)
{
  JLM_ASSERT(ChrecMap_.find(&output) != ChrecMap_.end());
  auto & chrec = ChrecMap_.at(&output);

  // We only support invariant and affine recurrences (1-2 operands) that are not unknown
  if (SCEVChainRecurrence::IsUnknown(*chrec) || chrec->NumOperands() > 2)
    return;

  const auto * intType = dynamic_cast<const rvsdg::BitType *>(output.Type().get());
  if (!intType)
    return;

  const auto & startSCEV = chrec->GetStartValue();
  const auto & startConstant = dynamic_cast<const SCEVConstant *>(startSCEV);
  if (!startConstant)
    return;

  const auto numBits = intType->nbits();
  const auto startValue = startConstant->GetValue();
  const auto & startValueNode =
      IntegerConstantOperation::Create(*thetaNode.region(), numBits, startValue);
  auto newIV = thetaNode.AddLoopVar(startValueNode.output(0));

  if (SCEVChainRecurrence::IsAffine(*chrec))
  {
    const auto & stepPtr = chrec->GetStep();
    if (!stepPtr.has_value())
      return;

    const auto & stepSCEV = *stepPtr;
    const auto & stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get());

    if (!stepConstant)
      return;

    const auto stepValue = stepConstant->GetValue();
    const auto & stepValueNode =
        IntegerConstantOperation::Create(*thetaNode.subregion(), numBits, stepValue);
    const auto & newAddNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>(
        { newIV.pre, stepValueNode.output(0) },
        numBits);

    newIV.post->divert_to(newAddNode.output(0));
  }

  output.divert_users(newIV.pre);

  // Insert the chrec for the new induction variable
  // NOTE: This only updates the copy of the original chrec map from the scalar evolution
  // analysis. In the future, we would want to insert this into the "global" chrec map so other
  // analyses and passes can use it as well.
  ChrecMap_[newIV.pre] = std::move(chrec);

  Context_->IncrementOperationsReducedCount(thetaNode);
}

bool
LoopStrengthReduction::IsValidCandidateOperation(const SCEV & scevTree)
{
  // Accept any linear combination that involves multiplication somewhere in the tree
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
