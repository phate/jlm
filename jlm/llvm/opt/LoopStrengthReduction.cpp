/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/LoopStrengthReduction.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

#include <algorithm>

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

  size_t NumArithmeticCandidates = 0;
  size_t NumGEPCandidates = 0;
  size_t NumArithmeticOperationsReduced = 0;
  size_t NumGEPOperationsReduced = 0;

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
        Label::NumLSRCandidates,
        context.NumArithmeticCandidates + context.NumGEPCandidates);
    AddMeasurement(Label::NumArithmeticLSRCandidates, context.NumArithmeticCandidates);
    AddMeasurement(Label::NumGepLSRCandidates, context.NumGEPCandidates);
    AddMeasurement(Label::NumArithmeticOperationsReduced, context.NumArithmeticOperationsReduced);
    AddMeasurement(Label::NumGepOperationsReduced, context.NumGEPOperationsReduced);
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
        thetaNode->subregion()->prune(false);
      }
    }
  }
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
  // A candidate operation must satisfy four conditions:
  //   1. The operation is either arithmetic (+, * or -) or a GEP operation
  //   2. The SCEV tree of the operation is a linear combination of loop variables and constants
  //   3. For arithmetic operations, the subtree must contain at least one multiplication node
  //      (IntegerMulOperation), ensuring we only reduce operations that actually benefit from
  //      strength reduction.
  //   4. Its SCEV evaluates to a chain recurrence of the form {a,+,b}, meaning the value
  //      can be expressed as a start value plus a constant step per iteration. This is what makes
  //      it a valid induction variable candidate. For GEP operations, a, is the base
  //      address of the array we are indexing into.
  //
  // Modifying the RVSDG is done as follows:
  //
  // For an arithmetic candidate operation j with recurrence {a,+,b}:
  //   1. Introduce a new loop variable initialized to the base value a
  //   2. Replace all uses of j with the new loop variable
  // If b is not 0:
  //   3. Update the new loop variable each iteration by incrementing it by b
  //
  // For a GEP which has the recurrence {Init(a),+,b}, where Init(a) is the SCEV for the base
  // address loop variable:
  //   1. Introduce a new loop variable initialized to the base address
  //   2. Replace all uses of the original GEP with the new loop variable
  //   3. Update the new loop variable each iteration by advancing it by offset b using a new GEP
  // In the case where b is invariant, the GEP can be eliminated entirely.

  // We traverse the nodes in the theta node in a bottom-up manner starting at the origin output of
  // the post values for the loop variables. We look for candidate operations and add them to the
  // stack of operations to be reduced.
  util::HashSet<rvsdg::Output *> candidateOperations;
  util::HashSet<rvsdg::Output *> visited;
  DependsOnIVMemo_.clear();
  ContainsMulMemo_.clear();
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

  const auto & simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output);

  if (!simpleNode)
    return;

  // Multiplication, addition, subtraction, shl and GEP operations are candidates for strength
  // reduction
  if (const auto & operation = simpleNode->GetOperation();
      rvsdg::is<IntegerMulOperation>(operation) || rvsdg::is<IntegerAddOperation>(operation)
      || rvsdg::is<IntegerSubOperation>(operation) || rvsdg::is<IntegerShlOperation>(operation)
      || rvsdg::is<GetElementPtrOperation>(operation))
  {
    if (SCEVMap_.find(&output) == SCEVMap_.end())
      return;

    if (IsValidCandidateOperation(output, operation))
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

  JLM_ASSERT(chrec->NumOperands() <= 2);

  const rvsdg::ThetaNode::LoopVar * newIV = nullptr;
  if (const auto & bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(output.Type()))
  {
    const auto numBits = bitType->nbits();
    const auto newIVOpt = CreateNewArithmeticInductionVariable(*chrec, thetaNode, numBits);
    if (!newIVOpt.has_value())
      return;

    newIV = &newIVOpt.value();

    Context_->NumArithmeticOperationsReduced++;
  }
  else if (const auto & pointerType = std::dynamic_pointer_cast<const PointerType>(output.Type()))
  {
    const auto newIVOpt = CreateNewGEPInductionVariable(*chrec, thetaNode, pointerType);
    if (!newIVOpt.has_value())
      return;

    newIV = &newIVOpt.value();

    Context_->NumGEPOperationsReduced++;
  }
  else
  {
    throw std::logic_error("Invalid output type in ReplaceCandidateOperation!");
  }

  JLM_ASSERT(newIV != nullptr);

  output.divert_users(newIV->pre);
  // Insert the chrec for the new induction variable
  // NOTE: This only updates the *copy* of the original chrec map from the scalar evolution
  // analysis. In the future, we would want to insert this into the "global" chrec map so other
  // analyses and transformations can use it as well.
  ChrecMap_[newIV->pre] = std::move(chrec);
  Context_->IncrementOperationsReducedCount(thetaNode);
}

std::optional<rvsdg::Output *>
LoopStrengthReduction::HoistChrec(
    const SCEVChainRecurrence & chrec,
    const rvsdg::ThetaNode & thetaNode,
    const size_t numBits)
{
  if (!SCEVChainRecurrence::IsInvariantInLoop(chrec, thetaNode))
    return std::nullopt;

  auto & targetLoop = chrec.GetLoop();

  rvsdg::Output * chrecOutput = nullptr;
  if (ScalarEvolution::StructurallyEqual(*ChrecMap_.at(&chrec.GetOutput()), chrec))
  {
    // The chrec stored in the chrec map corresponds with the chrec we are processing (it is an
    // induction variable). This means that the chrec has a corresponding output which we can use.
    chrecOutput = &chrec.GetOutput();
  }
  else
  {
    // The chrec doesn't have a corresponding output. This can happen in cases where we have folded
    // some constant into the chrec of an induction variable, either with addition or
    // multiplication. In this case, we need to create a new induction variable for the chrec.
    const auto newIV = CreateNewArithmeticInductionVariable(chrec, targetLoop, numBits);
    if (!newIV.has_value())
      return std::nullopt;

    chrecOutput = newIV->pre;
  }

  if (targetLoop.subregion() == thetaNode.region())
  {
    return chrecOutput;
  }

  if (rvsdg::Region::isAncestor(*targetLoop.subregion(), *thetaNode.subregion()))
  {
    auto & traced = llvm::traceOutput(*chrecOutput, thetaNode.subregion());
    if (traced.region() != thetaNode.region())
      return std::nullopt;

    return &traced;
  }

  // This is fine to do and wont throw since we know, due to the cases above and the invariance
  // check at the start, that the region of the chrec output is guaranteed to be an ancestor of the
  // theta region
  auto & routed = rvsdg::RouteToRegion(*chrecOutput, *thetaNode.region());
  return &routed;
}

std::optional<rvsdg::Output *>
LoopStrengthReduction::HoistSCEVExpresssion(
    const SCEV & scev,
    rvsdg::ThetaNode & thetaNode,
    const size_t numBits)
{
  if (const auto constant = dynamic_cast<const SCEVConstant *>(&scev))
  {
    const auto value = constant->GetValue();
    const auto & constantNode =
        IntegerConstantOperation::Create(*thetaNode.region(), numBits, value);

    return constantNode.output(0);
  }
  if (const auto init = dynamic_cast<const SCEVInit *>(&scev))
  {
    const auto & prePointer = init->GetPrePointer();
    const auto targetLoop = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(prePointer);
    if (targetLoop == nullptr)
      return std::nullopt;

    const auto initLoopVar = targetLoop->MapPreLoopVar(prePointer);

    if (targetLoop->subregion() == thetaNode.subregion())
    {
      return initLoopVar.input->origin();
    }

    if (rvsdg::Region::isAncestor(*targetLoop->subregion(), *thetaNode.subregion()))
    {
      auto & traced = llvm::traceOutput(*initLoopVar.pre, thetaNode.region());

      if (traced.region() != thetaNode.region())
        return std::nullopt;

      return &traced;
    }

    auto & routed = rvsdg::RouteToRegion(*initLoopVar.pre, *thetaNode.region());
    return &routed;
  }
  if (const auto addExpr = dynamic_cast<const SCEVNAryAddExpr *>(&scev))
  {
    const auto rightMost = addExpr->GetOperand(addExpr->NumOperands() - 1);
    const auto rightSideOpt = HoistSCEVExpresssion(*rightMost, thetaNode, numBits);
    if (!rightSideOpt.has_value())
      return std::nullopt;

    const auto newAddExpr = SCEV::CloneAs<SCEVNAryAddExpr>(*addExpr);
    newAddExpr->RemoveOperand(addExpr->NumOperands() - 1);

    std::optional<rvsdg::Output *> leftSideOpt;
    if (newAddExpr->NumOperands() == 1)
    {
      // Only a constant
      const auto constantElement = newAddExpr->GetOperand(0);
      leftSideOpt = HoistSCEVExpresssion(*constantElement, thetaNode, numBits);
    }
    else
    {
      leftSideOpt = HoistSCEVExpresssion(*newAddExpr, thetaNode, numBits);
    }

    if (!leftSideOpt.has_value())
      return std::nullopt;

    rvsdg::Output * leftSide = *leftSideOpt;
    rvsdg::Output * rightSide = *rightSideOpt;
    // If either side is a pointer, we use a GEP instead of IntegerAddOperation
    if (const auto rightIsPtr = rvsdg::is<PointerType>(rightSide->Type()),
        leftIsPtr = rvsdg::is<PointerType>(leftSide->Type());
        leftIsPtr || rightIsPtr)
    {
      const auto ptrSide = leftIsPtr ? leftSide : rightSide;
      auto offsetSide = leftIsPtr ? rightSide : leftSide;

      const auto ptrType = std::dynamic_pointer_cast<const PointerType>(ptrSide->Type());
      JLM_ASSERT(ptrType);
      auto newGep = GetElementPtrOperation::Create(
          ptrSide,
          { offsetSide },
          rvsdg::BitType::Create(8), // Byte
          ptrType);

      return newGep;
    }

    if (const auto leftBitType = std::dynamic_pointer_cast<const rvsdg::BitType>(leftSide->Type());
        leftBitType->nbits() != numBits)
    {
      leftSide = SExtOperation::create(numBits, leftSide);
    }
    if (const auto rightBitType =
            std::dynamic_pointer_cast<const rvsdg::BitType>(rightSide->Type());
        rightBitType->nbits() != numBits)
    {
      rightSide = SExtOperation::create(numBits, rightSide);
    }

    auto & newAddNode =
        jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ leftSide, rightSide }, numBits);

    return newAddNode.output(0);
  }
  if (const auto mulExpr = dynamic_cast<const SCEVNAryMulExpr *>(&scev))
  {
    const auto rightMost = mulExpr->GetOperand(mulExpr->NumOperands() - 1);
    const auto rightSideOpt = HoistSCEVExpresssion(*rightMost, thetaNode, numBits);
    if (!rightSideOpt.has_value())
      return std::nullopt;

    const auto newMulExpr = SCEV::CloneAs<SCEVNAryMulExpr>(*mulExpr);
    newMulExpr->RemoveOperand(mulExpr->NumOperands() - 1);

    std::optional<rvsdg::Output *> leftSideOpt;
    if (newMulExpr->NumOperands() == 1)
    {
      const auto constantElement = newMulExpr->GetOperand(0);
      leftSideOpt = HoistSCEVExpresssion(*constantElement, thetaNode, numBits);
    }
    else
    {
      leftSideOpt = HoistSCEVExpresssion(*newMulExpr, thetaNode, numBits);
    }

    if (!leftSideOpt.has_value())
      return std::nullopt;

    rvsdg::Output * leftSide = *leftSideOpt;
    rvsdg::Output * rightSide = *rightSideOpt;
    // Sign extend in cases where the input does not match the expected number of bits
    if (const auto leftBitType = std::dynamic_pointer_cast<const rvsdg::BitType>(leftSide->Type());
        leftBitType->nbits() != numBits)
    {
      leftSide = SExtOperation::create(numBits, leftSide);
    }
    if (const auto rightBitType =
            std::dynamic_pointer_cast<const rvsdg::BitType>(rightSide->Type());
        rightBitType->nbits() != numBits)
    {
      rightSide = SExtOperation::create(numBits, rightSide);
    }

    auto & newMulNode =
        jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ leftSide, rightSide }, numBits);

    return newMulNode.output(0);
  }
  if (const auto chrec = dynamic_cast<const SCEVChainRecurrence *>(&scev))
  {
    if (!SCEVChainRecurrence::IsAffine(*chrec))
      return std::nullopt;

    return HoistChrec(*chrec, thetaNode, numBits);
  }
  throw std::logic_error("Unknown SCEV type in HoistSCEVExpression\n");
}

std::optional<rvsdg::ThetaNode::LoopVar>
LoopStrengthReduction::CreateNewArithmeticInductionVariable(
    const SCEVChainRecurrence & chrec,
    rvsdg::ThetaNode & thetaNode,
    const size_t numBits)
{
  const auto & startSCEV = chrec.GetStartValue();

  if (SCEVChainRecurrence::IsConstant(chrec))
  {
    // Chrec has the form {a}, which indicates a loop-invariant (trivial induction variable).
    // We can hoist this out of the loop by creating a new loop variable.
    const auto hoistedConstant = HoistSCEVExpresssion(*startSCEV, thetaNode, numBits);
    if (!hoistedConstant.has_value())
      return std::nullopt;

    return thetaNode.AddLoopVar(*hoistedConstant);
  }
  if (SCEVChainRecurrence::IsAffine(chrec))
  {
    // Chrec has the form {a,+,b} which is a basic induction variable
    const auto & stepPtr = chrec.GetStep();

    JLM_ASSERT(stepPtr.has_value());

    const auto & stepSCEV = *stepPtr;

    rvsdg::Output * stepOutput = nullptr;
    if (const auto stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get()))
    {
      const auto stepValue = stepConstant->GetValue();
      const auto & stepValueNode =
          IntegerConstantOperation::Create(*thetaNode.subregion(), numBits, stepValue);
      stepOutput = stepValueNode.output(0);
    }
    else if (const auto stepInit = dynamic_cast<const SCEVInit *>(stepSCEV.get()))
    {
      stepOutput = &stepInit->GetPrePointer();
    }
    else if (const auto stepNAryExpr = dynamic_cast<const SCEVNAryExpr *>(stepSCEV.get()))
    {
      const auto hoistedStep = HoistSCEVExpresssion(*stepNAryExpr, thetaNode, numBits);
      if (!hoistedStep.has_value())
        return std::nullopt;

      const auto newStepIV = thetaNode.AddLoopVar(*hoistedStep);
      stepOutput = newStepIV.pre;
    }
    else
    {
      return std::nullopt;
    }

    const auto hoistedStart = HoistSCEVExpresssion(*startSCEV, thetaNode, numBits);
    if (!hoistedStart.has_value())
      return std::nullopt;

    auto newIV = thetaNode.AddLoopVar(*hoistedStart);

    auto & newAddNode =
        jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ newIV.pre, stepOutput }, numBits);
    newIV.post->divert_to(newAddNode.output(0));

    return newIV;
  }
  throw std::logic_error("Invalid chrec size in CreateNewArithmeticInductionVariable!");
}

std::optional<rvsdg::ThetaNode::LoopVar>
LoopStrengthReduction::CreateNewGEPInductionVariable(
    const SCEVChainRecurrence & chrec,
    rvsdg::ThetaNode & thetaNode,
    const std::shared_ptr<const PointerType> & pointerType)
{
  const auto & baseAddressSCEV = chrec.GetStartValue();
  if (SCEVChainRecurrence::IsConstant(chrec))
  {
    const auto hoistedStart = HoistSCEVExpresssion(*baseAddressSCEV, thetaNode, 64);

    if (!hoistedStart.has_value())
      return std::nullopt;

    return thetaNode.AddLoopVar(*hoistedStart);
  }
  if (SCEVChainRecurrence::IsAffine(chrec))
  {
    // Chrec has the form {Init(a),+,b} where Init(a) is the (initial) value of the base address
    // and b is the offset (in bytes).
    const auto & stepPtr = chrec.GetStep();

    JLM_ASSERT(stepPtr.has_value());

    const auto & stepSCEV = *stepPtr;

    rvsdg::Output * stepOutput = nullptr;
    if (const auto stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get()))
    {
      const auto stepValue = stepConstant->GetValue();
      const auto & stepValueNode =
          IntegerConstantOperation::Create(*thetaNode.subregion(), 64, stepValue);
      stepOutput = stepValueNode.output(0);
    }
    else if (const auto stepInit = dynamic_cast<const SCEVInit *>(stepSCEV.get()))
    {
      stepOutput = &stepInit->GetPrePointer();
    }
    else if (const auto stepNAryExpr = dynamic_cast<const SCEVNAryExpr *>(stepSCEV.get()))
    {
      const auto hoistedStep = HoistSCEVExpresssion(*stepNAryExpr, thetaNode, 64);
      if (!hoistedStep.has_value())
        return std::nullopt;

      const auto newStepIV = thetaNode.AddLoopVar(*hoistedStep);
      stepOutput = newStepIV.pre;
    }
    else
    {
      return std::nullopt;
    }

    const auto hoistedStart = HoistSCEVExpresssion(*baseAddressSCEV, thetaNode, 64);

    if (!hoistedStart.has_value())
      return std::nullopt;

    const auto newIV = thetaNode.AddLoopVar(*hoistedStart);

    if (const auto stepBitType =
            std::dynamic_pointer_cast<const rvsdg::BitType>(stepOutput->Type());
        stepBitType->nbits() != 64)
    {
      stepOutput = SExtOperation::create(64, stepOutput);
    }

    auto newGep = GetElementPtrOperation::Create(
        newIV.pre,
        { stepOutput },
        rvsdg::BitType::Create(8), // Byte
        pointerType);

    newIV.post->divert_to(newGep);

    return newIV;
  }

  throw std::logic_error("Invalid chrec size in CreateNewGEPInductionVariable!");
}

bool
LoopStrengthReduction::IsValidCandidateOperation(
    const rvsdg::Output & output,
    const rvsdg::SimpleOperation & operation)
{
  if (rvsdg::is<GetElementPtrOperation>(operation))
    Context_->NumGEPCandidates++;
  else
    Context_->NumArithmeticCandidates++;

  if (!DependsOnInductionVariable(output))
    return false;

  // We only reduce arithmetic operations if they contain a multiplication somewhere
  if (rvsdg::is<IntegerBinaryOperation>(operation) && !ContainsMul(output))
    return false;

  // We only support invariant and affine recurrences (1-2 operands) that are not unknown
  if (const auto & chrec = ChrecMap_.at(&output);
      chrec->NumOperands() > 2 || ScalarEvolution::IsUnknown(*chrec))
    return false;

  return true;
}

bool
LoopStrengthReduction::DependsOnInductionVariable(const rvsdg::Output & output)
{
  if (const auto it = DependsOnIVMemo_.find(&output); it != DependsOnIVMemo_.end())
    return it->second;

  // Check if the current output is an induction variable (loop variable with predictable
  // evolution)
  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    const auto it = ChrecMap_.find(&output);
    if (it == ChrecMap_.end())
      return DependsOnIVMemo_[&output] = false;

    if (const auto & chrec = it->second; ScalarEvolution::IsUnknown(*chrec))
      return DependsOnIVMemo_[&output] = false;

    return DependsOnIVMemo_[&output] = true;
  }

  const auto & simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output);

  if (!simpleNode)
    return DependsOnIVMemo_[&output] = false;

  for (const auto & input : simpleNode->Inputs())
  {
    if (DependsOnInductionVariable(*input.origin()))
      return DependsOnIVMemo_[&output] = true;
  }

  return DependsOnIVMemo_[&output] = false;
}

bool
LoopStrengthReduction::ContainsMul(const rvsdg::Output & output)
{
  if (const auto it = ContainsMulMemo_.find(&output); it != ContainsMulMemo_.end())
    return it->second;

  const auto & simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output);

  if (!simpleNode)
    return ContainsMulMemo_[&output] = false;

  if (const auto & simpleOperation = simpleNode->GetOperation();
      rvsdg::is<IntegerMulOperation>(simpleOperation)
      || rvsdg::is<IntegerShlOperation>(simpleOperation))
    return ContainsMulMemo_[&output] = true;

  for (const auto & input : simpleNode->Inputs())
  {
    if (ContainsMul(*input.origin()))
      return ContainsMulMemo_[&output] = true;
  }

  return ContainsMulMemo_[&output] = false;
}
}
