/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/opt/LoopStrengthReduction.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/gamma.hpp>
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

  size_t NumCandidates = 0;
  size_t NumArithmeticCandidates = 0;
  size_t NumGepCandidates = 0;
  size_t NumDoesNotDependOnInductionVariable = 0;
  size_t NumDoesNotContainMultiplication = 0;
  size_t NumIsNotLinearCombination = 0;
  size_t NumIsUnknown = 0;
  size_t NumIsNotAffine = 0;
  size_t NumContainsNAryExpr = 0;

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

std::optional<std::vector<rvsdg::StructuralNode *>>
LoopStrengthReduction::FindLoopPath(const rvsdg::ThetaNode & from, const rvsdg::ThetaNode & to)
{
  std::vector<rvsdg::StructuralNode *> reversed;

  const auto * cursor = to.region();
  const auto * target = from.subregion();

  while (cursor != target)
  {
    auto * owner = cursor->node();
    if (!owner)
      return std::nullopt; // hit top without reaching `from`

    if (auto * theta = dynamic_cast<rvsdg::ThetaNode *>(owner))
    {
      reversed.push_back(theta);
      cursor = theta->region();
    }
    else if (auto * gamma = dynamic_cast<rvsdg::GammaNode *>(owner))
    {
      reversed.push_back(gamma);
      cursor = gamma->region();
    }
    else
    {
      return std::nullopt;
    }
  }

  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

std::optional<rvsdg::Output *>
LoopStrengthReduction::RouteValueThroughLoops(
    rvsdg::Output & origin,
    const rvsdg::ThetaNode & from,
    const rvsdg::ThetaNode & to)
{
  if (origin.region() != from.subregion())
    return std::nullopt;

  const auto loopPath = FindLoopPath(from, to);
  if (!loopPath.has_value())
    return std::nullopt;

  auto current = &origin;
  for (const auto intermediateNode : *loopPath)
  {
    if (const auto intermediateTheta = dynamic_cast<rvsdg::ThetaNode *>(intermediateNode))
    {
      const auto loopVar = intermediateTheta->AddLoopVar(current);
      current = loopVar.pre;
    }
    else if (const auto intermediateGamma = dynamic_cast<rvsdg::GammaNode *>(intermediateNode))
    {
      const auto [input, branchArgument] = intermediateGamma->AddEntryVar(current);
      current = branchArgument[1];
    }
  }
  return current;
}

std::optional<rvsdg::ThetaNode::LoopVar>
LoopStrengthReduction::CreateLoopVarForStart(
    const SCEV & startSCEV,
    rvsdg::ThetaNode & thetaNode,
    const size_t numBits)
{
  if (const auto * startConstant = dynamic_cast<const SCEVConstant *>(&startSCEV))
  {
    const auto & node =
        IntegerConstantOperation::Create(*thetaNode.region(), numBits, startConstant->GetValue());
    return thetaNode.AddLoopVar(node.output(0));
  }
  if (const auto * startInit = dynamic_cast<const SCEVInit *>(&startSCEV))
  {
    auto loopVar = thetaNode.MapPreLoopVar(*startInit->GetPrePointer());
    return thetaNode.AddLoopVar(loopVar.input->origin());
  }
  if (const auto * startChrec = dynamic_cast<const SCEVChainRecurrence *>(&startSCEV))
  {
    const auto newOutput = HoistChrec(*startChrec, thetaNode, numBits);

    if (!newOutput.has_value())
      return std::nullopt;

    return thetaNode.AddLoopVar(*newOutput);
  }
  if (const auto * startNAry = dynamic_cast<const SCEVNAryExpr *>(&startSCEV))
  {
    const auto newOutput = HoistNAryExpresssion(*startNAry, thetaNode, numBits);

    if (!newOutput.has_value())
      return std::nullopt;

    return thetaNode.AddLoopVar(*newOutput);
  }
  return std::nullopt;
}

std::optional<rvsdg::Output *>
LoopStrengthReduction::HoistChrec(
    const SCEVChainRecurrence & chrec,
    const rvsdg::ThetaNode & thetaNode,
    const size_t numBits) const
{
  if (!SCEVChainRecurrence::IsInvariantInLoop(chrec, thetaNode))
    return std::nullopt;

  if (ScalarEvolution::StructurallyEqual(*ChrecMap_.at(chrec.GetOutput()), chrec))
  {
    // The chrec stored in the chrec map corresponds with the chrec we are processing (it is an
    // induction variable). This means that the chrec has a corresponding output which we can use.
    if (chrec.GetLoop()->subregion() == thetaNode.region())
    {
      return chrec.GetOutput();
    }

    const auto routed = RouteValueThroughLoops(*chrec.GetOutput(), *chrec.GetLoop(), thetaNode);

    if (routed.has_value())
      return *routed;
  }
  else
  {
    // The chrec doesn't have a corresponding output. This can happen in cases where we have folded
    // some constant into the chrec of an induction variable, either with addition or
    // multiplication. In this case, we need to create a new induction variable for the chrec.
    if (SCEVChainRecurrence::IsAffine(chrec))
    {
      const auto targetLoop = chrec.GetLoop();

      const auto start = chrec.GetStartValue();
      auto step = chrec.GetStep();

      JLM_ASSERT(step.has_value());

      auto startConstant = dynamic_cast<SCEVConstant *>(start);
      auto stepConstant = dynamic_cast<SCEVConstant *>(step->get());

      if (!startConstant || !stepConstant)
        return std::nullopt;

      const auto & constantNode1 = IntegerConstantOperation::Create(
          *targetLoop->region(),
          numBits,
          startConstant->GetValue());

      auto chrecLoopVar = targetLoop->AddLoopVar(constantNode1.output(0));

      const auto & constantNode2 = IntegerConstantOperation::Create(
          *targetLoop->subregion(),
          numBits,
          stepConstant->GetValue());

      auto & newAddNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>(
          { chrecLoopVar.pre, constantNode2.output(0) },
          numBits);

      chrecLoopVar.post->divert_to(newAddNode.output(0));

      if (chrec.GetLoop()->subregion() == thetaNode.region())
      {
        return chrecLoopVar.pre;
      }

      const auto routed = RouteValueThroughLoops(*chrecLoopVar.pre, *targetLoop, thetaNode);

      if (routed.has_value())
        return *routed;
    }
  }
  return std::nullopt;
}

std::optional<rvsdg::Output *>
LoopStrengthReduction::HoistNAryExpresssion(
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
    auto prePointer = init->GetPrePointer();

    auto initLoopVar = thetaNode.MapPreLoopVar(*prePointer);

    return initLoopVar.input->origin();
  }
  if (const auto addExpr = dynamic_cast<const SCEVNAryAddExpr *>(&scev))
  {
    auto rightMost = addExpr->GetOperand(addExpr->NumOperands() - 1);
    auto rightSide = HoistNAryExpresssion(*rightMost, thetaNode, numBits);
    if (!rightSide.has_value())
      return std::nullopt;

    auto newAddExpr = SCEV::CloneAs<SCEVNAryAddExpr>(*addExpr);
    newAddExpr->RemoveOperand(addExpr->NumOperands() - 1);

    std::optional<rvsdg::Output *> leftSide;
    if (newAddExpr->NumOperands() == 1)
    {
      // Only a constant
      const auto constantElement = newAddExpr->GetOperand(0);
      leftSide = HoistNAryExpresssion(*constantElement, thetaNode, numBits);
    }
    else
    {
      leftSide = HoistNAryExpresssion(*newAddExpr, thetaNode, numBits);
    }

    if (!leftSide.has_value())
      return std::nullopt;

    auto & newAddNode =
        jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ *leftSide, *rightSide }, numBits);

    return newAddNode.output(0);
  }
  if (const auto mulExpr = dynamic_cast<const SCEVNAryMulExpr *>(&scev))
  {
    auto rightMost = mulExpr->GetOperand(mulExpr->NumOperands() - 1);
    auto rightSide = HoistNAryExpresssion(*rightMost, thetaNode, numBits);
    if (!rightSide.has_value())
      return std::nullopt;

    auto newMulExpr = SCEV::CloneAs<SCEVNAryMulExpr>(*mulExpr);
    newMulExpr->RemoveOperand(mulExpr->NumOperands() - 1);

    std::optional<rvsdg::Output *> leftSide;
    if (newMulExpr->NumOperands() == 1)
    {
      const auto constantElement = newMulExpr->GetOperand(0);
      leftSide = HoistNAryExpresssion(*constantElement, thetaNode, numBits);
    }
    else
    {
      leftSide = HoistNAryExpresssion(*newMulExpr, thetaNode, numBits);
    }

    if (!leftSide.has_value())
      return std::nullopt;

    auto & newMulNode =
        jlm::rvsdg::CreateOpNode<IntegerMulOperation>({ *leftSide, *rightSide }, numBits);

    return newMulNode.output(0);
  }
  if (const auto chrec = dynamic_cast<const SCEVChainRecurrence *>(&scev))
  {
    return HoistChrec(*chrec, thetaNode, numBits);
  }
  throw std::logic_error("Unknown SCEV type in HoistNAryExpression\n");
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

  if (SCEVChainRecurrence::IsConstant(*chrec))
  {
    // Chrec has the form {a}, which indicates a loop-invariant (trivial induction variable).
    // We can hoist this out of the loop by creating a new loop variable.
    auto newIV = CreateLoopVarForStart(*startSCEV, thetaNode, numBits);

    if (!newIV.has_value())
      return;

    output.divert_users(newIV->pre);
    ChrecMap_[newIV->pre] = std::move(chrec);
  }
  else if (SCEVChainRecurrence::IsAffine(*chrec))
  {
    // Chrec has the form {a,+,b} which is a basic induction variable
    const auto & stepPtr = chrec->GetStep();

    JLM_ASSERT(stepPtr.has_value());

    const auto & stepSCEV = *stepPtr;
    const auto & stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get());
    const auto & stepInit = dynamic_cast<const SCEVInit *>(stepSCEV.get());
    const auto & stepChrec = dynamic_cast<const SCEVChainRecurrence *>(stepSCEV.get());
    const auto & stepNAryExpr = dynamic_cast<const SCEVNAryExpr *>(stepSCEV.get());

    if (stepConstant)
    {
      auto newIV = CreateLoopVarForStart(*startSCEV, thetaNode, numBits);

      if (!newIV.has_value())
        return;

      const auto stepValue = stepConstant->GetValue();
      const auto & stepValueNode =
          IntegerConstantOperation::Create(*thetaNode.subregion(), numBits, stepValue);
      auto & newAddNode = jlm::rvsdg::CreateOpNode<IntegerAddOperation>(
          { newIV->pre, stepValueNode.output(0) },
          numBits);

      newIV->post->divert_to(newAddNode.output(0));
      output.divert_users(newIV->pre);

      // Insert the chrec for the new induction variable
      // NOTE: This only updates the *copy* of the original chrec map from the scalar evolution
      // analysis. In the future, we would want to insert this into the "global" chrec map so other
      // analyses and transformations can use it as well.
      ChrecMap_[newIV->pre] = std::move(chrec);
    }
    else if (stepInit)
    {
      auto newIV = CreateLoopVarForStart(*startSCEV, thetaNode, numBits);

      if (!newIV.has_value())
        return;

      auto stepPrePointer = stepInit->GetPrePointer();
      auto & newAddNode =
          jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ newIV->pre, stepPrePointer }, numBits);

      newIV->post->divert_to(newAddNode.output(0));
      output.divert_users(newIV->pre);
    }
    else if (stepChrec)
    {
      const auto hoisted = HoistChrec(*stepChrec, thetaNode, numBits);

      if (!hoisted.has_value())
        return;

      auto newStepIV = thetaNode.AddLoopVar(*hoisted);

      auto newIV = CreateLoopVarForStart(*startSCEV, thetaNode, numBits);
      auto & newAddNode =
          jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ newIV->pre, newStepIV.pre }, numBits);

      newIV->post->divert_to(newAddNode.output(0));
      output.divert_users(newIV->pre);
    }
    else if (stepNAryExpr)
    {
      const auto hoisted = HoistNAryExpresssion(*stepNAryExpr, thetaNode, numBits);

      if (!hoisted.has_value())
        return;

      auto newStepIV = thetaNode.AddLoopVar(*hoisted);

      auto newIV = CreateLoopVarForStart(*startSCEV, thetaNode, numBits);

      auto & newAddNode =
          jlm::rvsdg::CreateOpNode<IntegerAddOperation>({ newIV->pre, newStepIV.pre }, numBits);

      newIV->post->divert_to(newAddNode.output(0));
      output.divert_users(newIV->pre);
    }
  }
  else
  {
    throw std::logic_error("Invalid chrec size in ReplaceArithmeticOperation!");
  }

  Context_->IncrementOperationsReducedCount(thetaNode);
}

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

  if (SCEVChainRecurrence::IsConstant(*chrec))
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
LoopStrengthReduction::IsValidCandidateOperation(const rvsdg::Output & output)
{
  if (!DependsOnInductionVariable(output))
    return false;

  const auto & simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output);

  JLM_ASSERT(simpleNode);

  const auto & simpleOperation = simpleNode->GetOperation();

  // We only reduce arithmetic operations if they contain a multiplication somewhere
  if (rvsdg::is<IntegerBinaryOperation>(simpleOperation) && !ContainsMul(output))
    return false;

  const auto & chrec = ChrecMap_.at(&output);

  // We only support invariant and affine recurrences (1-2 operands) that are not unknown
  if (SCEVChainRecurrence::IsUnknown(*chrec))
    return false;

  if (chrec->NumOperands() > 2)
    return false;

  const auto & startSCEV = chrec->GetStartValue();

  if (rvsdg::is<GetElementPtrOperation>(simpleOperation))
  {
    const auto & startInit = dynamic_cast<const SCEVInit *>(startSCEV);
    if (!startInit)
      return false;

    if (SCEVChainRecurrence::IsAffine(*chrec))
    {
      const auto & stepPtr = chrec->GetStep();

      JLM_ASSERT(stepPtr.has_value());

      const auto & stepSCEV = *stepPtr;
      const auto & stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get());
      if (!stepConstant)
        return false;
    }
  }

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

    const auto & chrec = it->second;

    if (SCEVChainRecurrence::IsUnknown(*chrec))
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

  const auto & simpleOperation = simpleNode->GetOperation();

  if (rvsdg::is<IntegerMulOperation>(simpleOperation)
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
