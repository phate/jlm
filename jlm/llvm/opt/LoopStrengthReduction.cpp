/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/LoopStrengthReduction.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
class LoopStrengthReduction::Context final
{
public:
  ~Context() noexcept = default;

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
    OperationsReducedMap_[&theta]++;
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
    size_t totalCount = 0;
    for (auto & [thetaNode, operationsReduced] : operationsReducedMap)
    {
      totalCount += operationsReduced;

      s += "ID(" + std::to_string(thetaNode->subregion()->getRegionId())
         + ")=" + std::to_string(operationsReduced);
      s += ",";
    }
    s += "Total=" + std::to_string(totalCount);
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
  // In short, we look for candidate operations and replace them with a new induction variable.
  //
  // A candidate operation must satisfy three conditions:
  //   1. The statement is a linear combinations of loop variables and constants
  //   2. Its RVSDG subtree contains at least one multiplication node (IntegerMulOperation),
  //      ensuring we only reduce operations that actually benefit from strength reduction.
  //   3. Its SCEV evaluates to a chain recurrence of the form {a,+,b}, meaning the value
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
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::SimpleOperation>(output);

  if (!simpleNode)
    return;

  // Multiplication, addition, subtraction and GEP operations are candidates for strength
  // reduction
  if (rvsdg::is<IntegerMulOperation>(*operation) || rvsdg::is<IntegerAddOperation>(*operation)
      || rvsdg::is<IntegerSubOperation>(*operation)
      || rvsdg::is<GetElementPtrOperation>(*operation))
  {
    if (SCEVMap_.find(&output) == SCEVMap_.end())
      return;

    if (IsValidCandidateOperation(output))
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

  if (const auto & bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(output.Type()))
  {
    ReplaceArithmeticOperation(chrec, output, thetaNode, bitType);
  }
  else if (const auto & pointerType = std::dynamic_pointer_cast<const PointerType>(output.Type()))
  {
    ReplaceGEPOperation(chrec, output, thetaNode, pointerType);
  }
  else
  {
    throw std::logic_error("Invalid output type in ReplaceCandidateOperation!");
  }
}

void
LoopStrengthReduction::ReplaceArithmeticOperation(
    std::unique_ptr<SCEVChainRecurrence> & chrec,
    rvsdg::Output & output,
    rvsdg::ThetaNode & thetaNode,
    const std::shared_ptr<const rvsdg::BitType> & bitType)
{
  JLM_ASSERT(chrec->NumOperands() <= 2);

  const auto numBits = bitType->nbits();

  const auto & startSCEV = chrec->GetStartValue();
  const auto & startConstant = dynamic_cast<const SCEVConstant *>(startSCEV);

  JLM_ASSERT(startConstant);

  const auto startValue = startConstant->GetValue();

  if (SCEVChainRecurrence::IsInvariant(*chrec))
  {
    // Chrec has the form {a}, which indicates a loop-invariant (trivial induction variable)
    const auto & startValueNode =
        IntegerConstantOperation::Create(*thetaNode.region(), numBits, startValue);
    const auto newIV = thetaNode.AddLoopVar(startValueNode.output(0));

    output.divert_users(newIV.pre);

    // Insert the chrec for the new induction variable
    // NOTE: This only updates the *copy* of the original chrec map from the scalar evolution
    // analysis. In the future, we would want to insert this into the "global" chrec map so other
    // analyses and transformations can use it as well.
    ChrecMap_[newIV.pre] = std::move(chrec);

    Context_->IncrementOperationsReducedCount(thetaNode);
  }
  else if (SCEVChainRecurrence::IsAffine(*chrec))
  {
    // Chrec has the form {a,+,b} which is a basic induction variable
    const auto & stepPtr = chrec->GetStep();
    if (!stepPtr.has_value())
      return;

    const auto & stepSCEV = *stepPtr;
    const auto & stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get());

    if (!stepConstant)
      return;

    const auto & startValueNode =
        IntegerConstantOperation::Create(*thetaNode.region(), numBits, startValue);
    auto newIV = thetaNode.AddLoopVar(startValueNode.output(0));

    const auto stepValue = stepConstant->GetValue();
    const auto & stepValueNode =
        IntegerConstantOperation::Create(*thetaNode.subregion(), numBits, stepValue);
    auto & newAddNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>(
        { newIV.pre, stepValueNode.output(0) },
        numBits);

    newIV.post->divert_to(newAddNode.output(0));
    output.divert_users(newIV.pre);
void
LoopStrengthReduction::ReplaceGEPOperation(
    std::unique_ptr<SCEVChainRecurrence> & chrec,
    rvsdg::Output & output,
    rvsdg::ThetaNode & thetaNode,
    const std::shared_ptr<const PointerType> & pointerType)
{
  JLM_ASSERT(chrec->NumOperands() <= 2);

  const auto & baseAddressSCEV = chrec->GetStartValue();
  const auto & baseAddressInit = dynamic_cast<const SCEVInit *>(baseAddressSCEV);

  JLM_ASSERT(baseAddressInit);

  const auto baseAddressPointer = baseAddressInit->GetPrePointer();
  const auto baseAddressLoopVar = thetaNode.MapPreLoopVar(*baseAddressPointer);

  if (SCEVChainRecurrence::IsInvariant(*chrec))
  {
    // The offset for the address computed by the GEP is always zero. We can replace the GEP
    // operation with just the base address
    output.divert_users(baseAddressLoopVar.pre);
  }
  else if (SCEVChainRecurrence::IsAffine(*chrec))
  {
    // Chrec has the form {Init(a),+,b} where Init(a) is the (initial) value of the base address
    // and b is the offset (in bytes).
    const auto & stepPtr = chrec->GetStep();

    JLM_ASSERT(stepPtr.has_value());

    const auto & stepSCEV = *stepPtr;
    const auto & stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get());

    JLM_ASSERT(stepConstant);

    const auto newIV = thetaNode.AddLoopVar(baseAddressLoopVar.input->origin());

    const auto stepValue = stepConstant->GetValue();
    const auto & stepValueNode =
        IntegerConstantOperation::Create(*thetaNode.subregion(), 64, stepValue);

    auto newGep = GetElementPtrOperation::Create(
        newIV.pre,
        { stepValueNode.output(0) },
        rvsdg::BitType::Create(8), // Byte
        pointerType);

    newIV.post->divert_to(newGep);
    output.divert_users(newIV.pre);

    ChrecMap_[newIV.pre] = std::move(chrec);
  }
  else
  {
    throw std::logic_error("Invalid chrec size in ReplaceGepOperation!");
  }

  Context_->IncrementOperationsReducedCount(thetaNode);
}

bool
LoopStrengthReduction::IsValidCandidateOperation(const rvsdg::Output & output) const
{
  const auto & scevTree = *SCEVMap_.at(&output);
  // Accept any linear combination that involves multiplication somewhere in the tree
  return IsLinearCombination(scevTree) && ContainsMul(output);
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
LoopStrengthReduction::ContainsMul(const rvsdg::Output & output)
{
  const auto & [simpleNode, mulOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerMulOperation>(output);

  if (!simpleNode)
    return false;

  if (mulOperation)
    return true;

  for (const auto & input : simpleNode->Inputs())
  {
    if (ContainsMul(*input.origin()))
      return true;
  }

  return false;
}
}
