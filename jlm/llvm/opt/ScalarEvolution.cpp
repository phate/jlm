/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Statistics.hpp>

<<<<<<< loop-trip-count
#include <cmath>
=======
#include <algorithm>
>>>>>>> master
#include <queue>

namespace jlm::llvm
{

class ScalarEvolution::Context final
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

  void
  AddLoopVar(const rvsdg::Output & var)
  {
    LoopVars_.push_back(&var);
  }

  size_t
  GetNumTotalLoopVars() const
  {
    return LoopVars_.size();
  }

  static std::unique_ptr<Context>
  Create()
  {
    return std::make_unique<Context>();
  }

  std::unique_ptr<SCEVChainRecurrence>
  TryGetChrecForOutput(const rvsdg::Output & output) const
  {
    const auto it = ChrecMap_.find(&output);
    if (it == ChrecMap_.end() || !it->second)
      return nullptr;

    return SCEV::CloneAs<SCEVChainRecurrence>(*it->second);
  }

  std::unique_ptr<SCEV>
  TryGetSCEVForOutput(const rvsdg::Output & output) const
  {
    const auto it = SCEVMap_.find(&output);
    if (it == SCEVMap_.end() || !it->second)
      return nullptr;

    return it->second->Clone();
  }

  void
  InsertChrec(const rvsdg::Output & output, const std::unique_ptr<SCEVChainRecurrence> & chrec)
  {
    ChrecMap_.insert_or_assign(&output, SCEV::CloneAs<SCEVChainRecurrence>(*chrec));
  }

  const std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> &
  GetChrecMap() const noexcept
  {
    return ChrecMap_;
  }

  const std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> &
  GetSCEVMap() const noexcept
  {
    return SCEVMap_;
  }

  int
  GetNumOfChrecsWithOrder(const size_t n) const
  {
    int count = 0;
    for (auto & [out, chrec] : ChrecMap_)
    {
      // Count chrecs with specific order
      if (chrec->GetOperands().size() == n + 1 && !IsUnknown(*chrec))
        count++;
    }
    return count;
  }

  size_t
  GetNumTotalChrecs() const
  {
    int count = 0;
    for (auto & [out, chrec] : ChrecMap_)
    {
      // Only count chrecs that are not unknown
      if (!IsUnknown(*chrec))
        count++;
    }
    return count;
  }

  void
  InsertSCEV(const rvsdg::Output & output, const std::unique_ptr<SCEV> & scev)
  {
    SCEVMap_.insert_or_assign(&output, scev->Clone());
  }

  void
  SetTripCount(const rvsdg::ThetaNode & thetaNode, const TripCount & tripCount)
  {
    TripCountMap_.insert_or_assign(&thetaNode, tripCount);
  }

  TripCount
  GetTripCount(const rvsdg::ThetaNode & thetaNode) const
  {
    return TripCountMap_.at(&thetaNode);
  }

  const std::unordered_map<const rvsdg::ThetaNode *, TripCount> &
  GetTripCountMap() const noexcept
  {
    return TripCountMap_;
  }

private:
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> ChrecMap_;
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> SCEVMap_;
  std::unordered_map<const rvsdg::ThetaNode *, TripCount> TripCountMap_;
  std::vector<const rvsdg::Output *> LoopVars_;
};

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
  Stop(const Context & context) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(Label::NumTotalRecurrences, context.GetNumTotalChrecs());
    AddMeasurement(Label::NumConstantRecurrences, context.GetNumOfChrecsWithOrder(0));
    AddMeasurement(Label::NumFirstOrderRecurrences, context.GetNumOfChrecsWithOrder(1));
    AddMeasurement(Label::NumSecondOrderRecurrences, context.GetNumOfChrecsWithOrder(2));
    AddMeasurement(Label::NumThirdOrderRecurrences, context.GetNumOfChrecsWithOrder(3));
    AddMeasurement(Label::NumLoopVariablesTotal, context.GetNumTotalLoopVars());

    std::string tripCounts = "";
    bool first = true;
    for (auto & [thetaNode, tripCount] : context.GetTripCountMap())
    {
      if (!first)
        tripCounts += ',';
      first = false;

      const std::string count = tripCount.IsFinite()   ? std::to_string(tripCount.GetCount())
                              : tripCount.IsInfinite() ? "Infinite"
                                                       : "CouldNotCompute";
      tripCounts += "ID(" + std::to_string(thetaNode->subregion()->getRegionId()) + ")=" + count;
    }

    AddMeasurement(Label::TripCounts, tripCounts);
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

ScalarEvolution::ScalarEvolution()
    : Transformation("ScalarEvolution")
{}

ScalarEvolution::~ScalarEvolution() noexcept = default;

std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>>
ScalarEvolution::GetChrecMap() const
{
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> mapCopy{};
  for (auto & [output, chrec] : Context_->GetChrecMap())
  {
    mapCopy.emplace(output, SCEV::CloneAs<SCEVChainRecurrence>(*chrec));
  }
  return mapCopy;
}

std::unordered_map<const rvsdg::ThetaNode *, ScalarEvolution::TripCount>
ScalarEvolution::GetTripCountMap() const noexcept
{
  return Context_->GetTripCountMap();
}

void
ScalarEvolution::Run(
    rvsdg::RvsdgModule & rvsdgModule,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());
  statistics->Start();

  Context_ = Context::Create();
  const rvsdg::Region & rootRegion = rvsdgModule.Rvsdg().GetRootRegion();
  AnalyzeRegion(rootRegion);
  CombineChrecsAcrossLoops();

  statistics->Stop(*Context_);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
};

void
ScalarEvolution::AnalyzeRegion(const rvsdg::Region & region)
{
  for (const auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        AnalyzeRegion(subregion);
      }
      if (const auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(structuralNode))
      {
        // Add number of loop vars in theta (for statistics)
        for (const auto loopVar : thetaNode->GetLoopVars())
        {
          if (loopVar.pre->Type()->Kind() != rvsdg::TypeKind::State)
          {
            // Only add loop variables that are not states
            Context_->AddLoopVar(*loopVar.pre);
          }
        }

        PerformSCEVAnalysis(*thetaNode);

        auto tripCount = GetPredictedTripCount(*thetaNode);

        Context_->SetTripCount(*thetaNode, tripCount);
      }
    }
  }
}

bool
ScalarEvolution::StepAlwaysNegative(const SCEV & stepSCEV)
{
  if (auto constantStep = dynamic_cast<const SCEVConstant *>(&stepSCEV))
  {
    return constantStep->GetValue() < 0;
  }
  if (auto recurrenceStep = dynamic_cast<const SCEVChainRecurrence *>(&stepSCEV))
  {
    JLM_ASSERT(SCEVChainRecurrence::IsAffine(*recurrenceStep));

    const auto start = dynamic_cast<const SCEVConstant *>(recurrenceStep->GetStartValue());
    auto stepPtr = recurrenceStep->GetStep();
    const auto step = dynamic_cast<const SCEVConstant *>(stepPtr.get());

    if (!start || !step)
      throw std::logic_error("Step can only contain constant SCEVs!");

    const auto a = start->GetValue();
    const auto b = step->GetValue();

    return a <= 0 && b <= 0 && !(a == 0 && b == 0);
  }
  throw std::logic_error("Wrong type for step!");
}

bool
ScalarEvolution::StepAlwaysPositive(const SCEV & stepSCEV)
{
  if (auto constantStep = dynamic_cast<const SCEVConstant *>(&stepSCEV))
  {
    return constantStep->GetValue() > 0;
  }
  if (auto recurrenceStep = dynamic_cast<const SCEVChainRecurrence *>(&stepSCEV))
  {
    JLM_ASSERT(SCEVChainRecurrence::IsAffine(*recurrenceStep));

    const auto start = dynamic_cast<const SCEVConstant *>(recurrenceStep->GetStartValue());
    auto stepPtr = recurrenceStep->GetStep();
    const auto step = dynamic_cast<const SCEVConstant *>(stepPtr.get());

    if (!start || !step)
      throw std::logic_error("Step can only contain constant SCEVs!");

    const auto a = start->GetValue();
    const auto b = step->GetValue();

    return a >= 0 && b >= 0 && !(a == 0 && b == 0);
  }
  throw std::logic_error("Wrong type for step!");
}

bool
ScalarEvolution::StepAlwaysZero(const SCEV & stepSCEV)
{
  if (auto constantStep = dynamic_cast<const SCEVConstant *>(&stepSCEV))
  {
    return constantStep->GetValue() == 0;
  }
  if (auto recurrenceStep = dynamic_cast<const SCEVChainRecurrence *>(&stepSCEV))
  {
    JLM_ASSERT(SCEVChainRecurrence::IsAffine(*recurrenceStep));

    const auto start = dynamic_cast<const SCEVConstant *>(recurrenceStep->GetStartValue());
    auto stepPtr = recurrenceStep->GetStep();
    const auto step = dynamic_cast<const SCEVConstant *>(stepPtr.get());

    if (!start || !step)
      throw std::logic_error("Step can only contain constant SCEVs!");

    const auto a = start->GetValue();
    const auto b = step->GetValue();

    return a == 0 && b == 0;
  }
  throw std::logic_error("Wrong type for step!");
}

ScalarEvolution::TripCount
ScalarEvolution::GetPredictedTripCount(const rvsdg::ThetaNode & thetaNode)
{
  const auto pred = thetaNode.predicate();
  const auto & [node, matchOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::MatchOperation>(*pred->origin());
  if (!matchOperation)
    throw std::logic_error("Predicate is not connected to a match node!");

  JLM_ASSERT(node->ninputs() == 1); // Match node only has 1 input

  const auto origin = node->input(0)->origin();
  const auto comparisonNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*origin);
  if (!comparisonNode)
    throw std::logic_error("Match node is not connected to a simple node!");

  const auto * comparisonOperation = &comparisonNode->GetOperation();
  if (!(dynamic_cast<const IntegerSltOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerSleOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerUltOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerUleOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerSgtOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerSgeOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerUgtOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerUgeOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerNeOperation *>(comparisonOperation)
        || dynamic_cast<const IntegerEqOperation *>(comparisonOperation)))
    throw std::logic_error("Parent of match node is not a comparison operation!");

  const auto & lhs = *comparisonNode->input(0)->origin();
  const auto & rhs = *comparisonNode->input(1)->origin();
  auto lhsChrec = Context_->TryGetChrecForOutput(lhs);
  auto rhsChrec = Context_->TryGetChrecForOutput(rhs);

  if (!lhsChrec)
    lhsChrec = GetOrCreateChainRecurrence(lhs, *GetOrCreateSCEVForOutput(lhs), thetaNode);

  if (!rhsChrec)
    rhsChrec = GetOrCreateChainRecurrence(rhs, *GetOrCreateSCEVForOutput(rhs), thetaNode);

  int64_t bound = 0;
  std::unique_ptr<SCEVChainRecurrence> chrec{};

  if (SCEVChainRecurrence::IsInvariant(*lhsChrec))
  {
    const auto constantSCEV = dynamic_cast<SCEVConstant *>(lhsChrec->GetOperand(0));
    if (!constantSCEV)
      return TripCount::CouldNotCompute();

    bound = constantSCEV->GetValue();
    chrec = SCEV::CloneAs<SCEVChainRecurrence>(*rhsChrec);
  }
  else if (SCEVChainRecurrence::IsInvariant(*rhsChrec))
  {
    const auto constantSCEV = dynamic_cast<SCEVConstant *>(rhsChrec->GetOperand(0));
    if (!constantSCEV)
      return TripCount::CouldNotCompute();

    bound = constantSCEV->GetValue();
    chrec = SCEV::CloneAs<SCEVChainRecurrence>(*lhsChrec);
  }
  else
  {
    // None of them are invariant, we can't reliably compute the backedge taken count
    return TripCount::CouldNotCompute();
  }

  if (!(SCEVChainRecurrence::IsAffine(*chrec) || SCEVChainRecurrence::IsQuadratic(*chrec)))
  {
    // We can only compute the trip count reliably for affine and quadratic recurrences. In other
    // cases, return "could not compute"
    return TripCount::CouldNotCompute();
  }

  for (const auto op : chrec->GetOperands())
  {
    if (!dynamic_cast<const SCEVConstant *>(op))
    {
      // If any of the operands is not a constant, we cannot compute the trip count, and should
      // return early
      return TripCount::CouldNotCompute();
    }
  }

  const auto start = dynamic_cast<const SCEVConstant *>(chrec->GetStartValue())->GetValue();
  const auto stepSCEV = chrec->GetStep();
  if (!stepSCEV)
  {
    return TripCount::CouldNotCompute();
  }

  if (dynamic_cast<const IntegerSltOperation *>(comparisonOperation)
      || dynamic_cast<const IntegerUltOperation *>(comparisonOperation))
  {
    // Trivial case (backedge is not taken and the only iteration is the first one)
    if (start >= bound)
      return TripCount::Finite(1);
    if (start < bound && StepAlwaysPositive(*stepSCEV))
    {
      const auto backedgeTakenCount =
          ComputeBackedgeTakenCountForChrec(*chrec, bound, comparisonOperation);
      if (backedgeTakenCount.has_value())
      {
        // The trip count for a loop is the backedge taken count plus one
        return TripCount::Finite(*backedgeTakenCount + 1);
      }
    }
  }
  if (dynamic_cast<const IntegerSleOperation *>(comparisonOperation)
      || dynamic_cast<const IntegerUleOperation *>(comparisonOperation))
  {
    if (start > bound)
      return TripCount::Finite(1);
    if (start <= bound && StepAlwaysPositive(*stepSCEV))
    {
      const auto backedgeTakenCount =
          ComputeBackedgeTakenCountForChrec(*chrec, bound, comparisonOperation);
      if (backedgeTakenCount.has_value())
      {
        return TripCount::Finite(*backedgeTakenCount + 1);
      }
    }
  }

  if (dynamic_cast<const IntegerSgtOperation *>(comparisonOperation)
      || dynamic_cast<const IntegerUgtOperation *>(comparisonOperation))
  {
    if (start <= bound)
      return TripCount::Finite(1);
    if (start > bound && StepAlwaysNegative(*stepSCEV))
    {
      const auto backedgeTakenCount =
          ComputeBackedgeTakenCountForChrec(*chrec, bound, comparisonOperation);
      if (backedgeTakenCount.has_value())
      {
        return TripCount::Finite(*backedgeTakenCount + 1);
      }
    }
  }
  if (dynamic_cast<const IntegerSgeOperation *>(comparisonOperation)
      || dynamic_cast<const IntegerUgeOperation *>(comparisonOperation))
  {
    if (start < bound)
      return TripCount::Finite(1);
    if (start >= bound && StepAlwaysNegative(*stepSCEV))
    {
      const auto backedgeTakenCount =
          ComputeBackedgeTakenCountForChrec(*chrec, bound, comparisonOperation);
      if (backedgeTakenCount.has_value())
      {
        return TripCount::Finite(*backedgeTakenCount + 1);
      }
    }
  }

  if (dynamic_cast<const IntegerNeOperation *>(comparisonOperation))
  {
    if (SCEVChainRecurrence::IsAffine(*chrec))
    {
      // With Ne and Eq comparisons, we only compute non-trivial backedge counts for affine
      // recurrences as there is no general way to compute it for quadratic recurrences.
      const auto step = dynamic_cast<const SCEVConstant *>(stepSCEV.get())->GetValue();
      if (StepAlwaysPositive(*stepSCEV))
      {
        const auto backedgeTakenCount =
            ComputeBackedgeTakenCountForChrec(*chrec, bound, comparisonOperation);
        // We need to make sure that it does not pass the bound value (results infinite loop)
        if (start <= bound && (bound - start) % step == 0)
          return TripCount::Finite(*backedgeTakenCount + 1);
      }
      if (StepAlwaysNegative(*stepSCEV))
      {
        const auto backedgeTakenCount =
            ComputeBackedgeTakenCountForChrec(*chrec, bound, comparisonOperation);
        if (start >= bound && (bound - start) % step == 0)
          return TripCount::Finite(*backedgeTakenCount + 1);
      }
    }
    if (start == bound)
      return TripCount::Finite(1);
  }

  if (dynamic_cast<const IntegerEqOperation *>(comparisonOperation))
  {
    if (start == bound)
    {
      if (!StepAlwaysZero(*stepSCEV))
        return TripCount::Finite(2); // Backedge taken once
    }
    else
      return TripCount::Finite(1);
  }

  if (SCEVChainRecurrence::IsQuadratic(*chrec))
  {
    // For quadratic recurrences, the value could evolve in an unpredictable way. In these cases,
    // we should return "could not compute" in order to be safe.
    if (!(StepAlwaysPositive(*stepSCEV) || StepAlwaysNegative(*stepSCEV)
          || StepAlwaysZero(*stepSCEV)))
    {
      return TripCount::CouldNotCompute();
    }
  }

  // If we have not returned a value at this point, we have an infinite loop.
  return TripCount::Infinite();
}

std::optional<size_t>
ScalarEvolution::ComputeBackedgeTakenCountForChrec(
    const SCEVChainRecurrence & chrec,
    const int64_t bound,
    const rvsdg::SimpleOperation * comparisonOperation)
{
  // Figuring out the trip count can be equivalently expressed as:
  // "At which iteration does the loop condition become false?"
  // That’s naturally expressed as:
  // f(i) - k changes sign, where f(i) is the value of the recurrence at iteration i and k is the
  // bound value

  const auto start = dynamic_cast<const SCEVConstant *>(chrec.GetStartValue())->GetValue();
  const auto stepSCEV = chrec.GetStep();

  bool isEqualsComparison = dynamic_cast<const IntegerSleOperation *>(comparisonOperation)
                         || dynamic_cast<const IntegerUleOperation *>(comparisonOperation)
                         || dynamic_cast<const IntegerSgeOperation *>(comparisonOperation)
                         || dynamic_cast<const IntegerUgeOperation *>(comparisonOperation);

  // Check the size of the step recurrence: 1 -> Affine, 2 -> Quadratic
  // We can only compute the backedge taken count for these two cases
  if (SCEVChainRecurrence::IsAffine(chrec))
  {
    const auto stepConstant = dynamic_cast<const SCEVConstant *>(stepSCEV.get());
    const auto step = stepConstant->GetValue();

    // f(i) = a + b * i
    // f(i) = k => a + b * i = k => i = (k - a)/b
    size_t result = std::ceil(static_cast<double>(bound - start) / step);

    if (isEqualsComparison)
    {
      // If we have an equals comparison and the value of the difference between the bound and the
      // start is a whole multiple of the step size, we get another backedge taken
      if ((bound - start) % step == 0)
        result += 1;
    }
    return result;
  }
  if (SCEVChainRecurrence::IsQuadratic(chrec))
  {
    // Create a quadratic equation for the recurrence {a,+,b,+,c}
    // The start value is a, and the increments are b, b+c, b+2c, ..., so the accumulated values are
    //   a+b, (a+b)+(b+c), (a+b)+(b+c)+(b+2c), ..., that is,
    //   a+b, a+2b+c, a+3b+3c, ...
    // After i iterations the  value is a + ib + i(i-1)/2 c = f(i).
    const auto stepRecurrence = dynamic_cast<const SCEVChainRecurrence *>(stepSCEV.get());
    const int64_t stepFirst =
        dynamic_cast<const SCEVConstant *>(stepRecurrence->GetStartValue())->GetValue();
    const int64_t stepSecond =
        dynamic_cast<const SCEVConstant *>(stepRecurrence->GetStep().get())->GetValue();

    // Let f(i) = a + ib + i(i-1)/2 c
    //
    // We want to find out when this polynomial is equal to the compare value, i.e. f(i) = k.
    // This is equivalent with the expression f(i) - k "switching sign" from positive to negative.
    // Conversely, this is also when the predicate condition will no longer hold.
    //
    // The equation f(i) - k = 0 is written as:
    //   a + ib + i(i-1)/2 c - k = 0,  or  2(a-k) + 2b i + i(i-1) c = 0.
    // In a quadratic form it becomes:
    //   c i^2 + (2b - c) i + 2(a-k) = 0.
    //
    // We use the quadratic formula to solve this.

    const int64_t a = stepSecond;
    const int64_t b = 2 * stepFirst - stepSecond;
    const int64_t c = 2 * (start - bound);

    const auto quadraticResult = SolveQuadraticEquation(a, b, c);
    if (!quadraticResult.has_value())
      return std::nullopt;

    size_t result = *quadraticResult;

    if (isEqualsComparison)
    {
      // Same as for affine, but instead of checking using modulo, we evaluate the value at the
      // result and check
      const int64_t valueAtResult =
          start + result * stepFirst + result * (result - 1) / 2 * stepSecond;
      if (valueAtResult == bound)
        result += 1;
    }
    return result;
  }
  return std::nullopt;
}

std::optional<size_t>
ScalarEvolution::SolveQuadraticEquation(int64_t a, int64_t b, int64_t c)
{
  // If a is negative, negate all the coefficients to simplify the math
  if (a < 0)
  {
    a = -a;
    b = -b;
    c = -c;
  }

  const auto d = b * b - 4 * a * c; // Discriminant

  if (d < 0)
    return std::nullopt;

  // Integer square root of the discriminant
  int64_t sq = std::floor(std::sqrt(d));

  // Check if square root is exact
  const bool inexactSq = (sq * sq != d);

  // Adjust if sq^2 > discriminant (shouldn't happen with floor, but just to be safe)
  if (sq * sq > d)
    sq -= 1;

  int64_t x = 0;
  int64_t rem = 0;

  // The vertex (min/max value) of the parabola f(x) = Ax^2 + Bx + C is at -B/2A. Since A > 0, the
  // vertex is at a non-positive x location iff B >= 0. In that case the first zero crossing is the
  // greater root. If B < 0, the vertex is at a positive x location, meaning both roots are positive
  // and the smaller root is the first crossing.
  if (b < 0)
  {
    // The square root is rounded down, so the roots may be inexact. When using the quadratic
    // formula, the low root could be greater than the exact one. To make sure this does not happen,
    // we add 1 if the root is inexact when calculating the low root.
    x = (-b - (sq + (inexactSq ? 1 : 0))) / (2 * a);
    rem = (-b - sq) % (2 * a);
  }
  else
  {
    x = (-b + sq) / (2 * a);
    rem = (-b + sq) % (2 * a);
  }

  // Result should be non-negative
  if (x < 0)
    x = 0;

  // Check for exact solution
  if (!inexactSq && rem == 0)
  {
    return x;
  }

  // The exact value of the square root should be between sq and sq + 1
  // Check for sign change between f(x) and f(x+1)
  const int64_t valueAtX = (a * x + b) * x + c;
  const int64_t valueAtXPlusOne = (a * (x + 1) + b) * (x + 1) + c;

  const bool signChange =
      ((valueAtX < 0) != (valueAtXPlusOne < 0)) || ((valueAtX == 0) != (valueAtXPlusOne == 0));
  // Sign did not change, not a valid solution
  if (!signChange)
    return std::nullopt;

  x += 1;
  return x;
}

void
ScalarEvolution::CombineChrecsAcrossLoops()
{
  bool changed{};
  do
  {
    changed = false;

    std::vector<std::pair<const rvsdg::Output *, std::unique_ptr<SCEV>>> pending;
    for (const auto & [output, chrec] : Context_->GetChrecMap())
    {
      if (auto newSCEV = TryReplaceInitForSCEV(*chrec))
      {
        pending.emplace_back(output, std::move(*newSCEV));
        changed = true;
      }
    }

    for (auto & [output, scev] : pending)
    {
      // Check if the result is actually a chrec
      if (auto * chrec = dynamic_cast<SCEVChainRecurrence *>(scev.get()))
      {
        Context_->InsertChrec(*output, SCEV::CloneAs<SCEVChainRecurrence>(*chrec));
      }
      else
      {
        // The transformation produced a non-chrec SCEV (n-ary expression), store it in the SCEV
        // map instead
        Context_->InsertSCEV(*output, std::move(scev));
      }
    }
  } while (changed);
}

std::optional<std::unique_ptr<SCEV>>
ScalarEvolution::TryReplaceInitForSCEV(const SCEV & scev)
{
  if (const auto initSCEV = dynamic_cast<const SCEVInit *>(&scev))
  {
    // Found an Init node, find the origin of its input value and get or create its chain
    // recurrence
    const auto initPrePointer = initSCEV->GetPrePointer();
    if (const auto innerTheta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*initPrePointer))
    {
      const auto correspondingInput = innerTheta->MapPreLoopVar(*initPrePointer).input;
      const auto & inputOrigin = llvm::traceOutput(*correspondingInput->origin());
      if (const auto originSCEV = Context_->TryGetSCEVForOutput(inputOrigin))
      {
        // We have found a SCEV for the origin of the input, find the corresponding theta node so
        // we can create a recurrence for it
        const auto thetaParent = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(inputOrigin);
        const auto outerTheta =
            thetaParent ? thetaParent
                        : util::assertedCast<rvsdg::ThetaNode>(inputOrigin.region()->node());

        const auto chrec = GetOrCreateChainRecurrence(inputOrigin, *originSCEV, *outerTheta);

        // Create a chain recurrence for the SCEV, with the outer theta as the loop
        return chrec->Clone();
      }
    }
  }
  if (const auto nArySCEV = dynamic_cast<const SCEVNAryExpr *>(&scev))
  {
    // An n-ary scev is any scev with an arbitrary number of operands: chain recurrence, n-ary add
    // and n-ary mult. We want to recursively check all it's operands for Init nodes
    auto clone = SCEV::CloneAs<SCEVNAryExpr>(*nArySCEV);
    const auto operands = nArySCEV->GetOperands();
    bool changed = false;
    for (size_t i = 0; i < operands.size(); ++i)
    {
      if (auto result = TryReplaceInitForSCEV(*operands[i]))
      {
        if (*result)
        {
          // Replace the Init operand with the chrec
          changed = true;
          clone->ReplaceOperand(i, std::move(*result));
        }
      }
    }
    if (!changed)
      return std::nullopt;

    if (dynamic_cast<const SCEVChainRecurrence *>(&scev))
    {
      // Result is a new chain recurrence, return it
      return clone;
    }
    // If it is an n-ary expression (Add or Mul), we try to fold the operands into themselves,
    // e.g. if, after replacing Init nodes with recurrences, we have ({0,+,1} + {1,+,2}) in an
    // n-ary add expression, we can fold this into {1,+,3}.
    return FoldNAryExpression(*clone);
  }
  // Default is to just return nothing
  return std::nullopt;
}

void
ScalarEvolution::PerformSCEVAnalysis(const rvsdg::ThetaNode & thetaNode)
{
  std::vector<rvsdg::ThetaNode::LoopVar> nonStateLoopVars;
  for (const auto loopVar : thetaNode.GetLoopVars())
  {
    if (loopVar.pre->Type()->Kind() != rvsdg::TypeKind::State)
    {
      nonStateLoopVars.push_back(loopVar);
    }
  }

  for (const auto loopVar : nonStateLoopVars)
  {
    const auto post = loopVar.post;
    // We compute the SCEV for each non-state loop variable in a recursive bottom up fashion,
    // starting at the post's origin
    auto scev = GetOrCreateSCEVForOutput(*post->origin());
    Context_->InsertSCEV(*loopVar.output, scev); // Save the SCEV at the theta outputs as well
  }

  auto dependencyGraph = CreateDependencyGraph(thetaNode);

  util::HashSet<const rvsdg::Output *> validOutputs{};
  for (const auto & [output, deps] : dependencyGraph)
  {
    if (CanCreateChainRecurrence(*output, dependencyGraph))
      validOutputs.insert(output);
  }

  // Filter the dependency graph to only contain the outputs of the SCEVs that are valid chain
  // recurrences and update dependencies accordingly
  auto filteredDependencyGraph = dependencyGraph;
  for (auto it = filteredDependencyGraph.begin(); it != filteredDependencyGraph.end();)
  {
    if (!validOutputs.Contains(it->first))
    {
      for (auto & [node, deps] : filteredDependencyGraph)
        deps.erase(it->first);
      it = filteredDependencyGraph.erase(it);
    }
    else
      ++it;
  }

  const auto order = TopologicalSort(filteredDependencyGraph);

  for (const auto output : order)
  {
    std::unique_ptr<SCEV> scev{};
    if (const auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*output);
        &thetaNode == theta)
    {
      // For loop variables, we need to retrieve and use the SCEV saved at the post's origin,
      // equivalent to a "backedge" which describes how the value at the pre pointer is updated
      const auto & newOutput = *thetaNode.MapPreLoopVar(*output).post->origin();
      scev = Context_->TryGetSCEVForOutput(newOutput);
    }
    else
      scev = Context_->TryGetSCEVForOutput(*output);

    JLM_ASSERT(scev);

    auto chrec = GetOrCreateChainRecurrence(*output, *scev, thetaNode);
    Context_->InsertChrec(*output, chrec);
  }

  for (const auto & [output, scev] : Context_->GetSCEVMap())
  {
    if (std::find(order.begin(), order.end(), output) == order.end())
    {
      auto unknownChainRecurrence = SCEVChainRecurrence::Create(thetaNode);
      unknownChainRecurrence->AddOperand(SCEVUnknown::Create());
      Context_->InsertChrec(*output, unknownChainRecurrence);
    }
  }
}

std::unique_ptr<SCEV>
ScalarEvolution::GetOrCreateSCEVForOutput(const rvsdg::Output & output)
{
  if (const auto existing = Context_->TryGetSCEVForOutput(output))
    return existing->Clone();

  std::unique_ptr<SCEV> result{};
  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    // We know this is a loop variable, create a placeholder SCEV for now, and compute the
    // expression later
    result = SCEVPlaceholder::Create(output);
  }
  if (const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (rvsdg::is<LoadOperation>(simpleNode->GetOperation()))
    {
      const auto addressInputOrigin = LoadOperation::AddressInput(*simpleNode).origin();
      result = SCEVLoad::Create(GetOrCreateSCEVForOutput(*addressInputOrigin));
    }
    if (rvsdg::is<IOBarrierOperation>(simpleNode->GetOperation()))
    {
      const auto barredInputOrigin = IOBarrierOperation::BarredInput(*simpleNode).origin();
      result = GetOrCreateSCEVForOutput(*barredInputOrigin);
    }
    if (rvsdg::is<SExtOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() == 1);
      result = GetOrCreateSCEVForOutput(*simpleNode->input(0)->origin());
    }
    if (rvsdg::is<GetElementPtrOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() >= 2);
      const auto baseIndex = simpleNode->input(0)->origin();
      JLM_ASSERT(is<PointerType>(baseIndex->Type()));

      const auto gepOp = dynamic_cast<const GetElementPtrOperation *>(&simpleNode->GetOperation());
      const auto & pointeeType = gepOp->GetPointeeType();

      auto baseScev = GetOrCreateSCEVForOutput(*baseIndex);

      auto baseOffsetIndex = GetOrCreateSCEVForOutput(*simpleNode->input(1)->origin());

      const auto wholeTypeSize = GetTypeAllocSize(pointeeType);
      std::unique_ptr<SCEV> offset =
          SCEVMulExpr::Create(std::move(baseOffsetIndex), SCEVConstant::Create(wholeTypeSize));
      if (auto innerOffset = ComputeSCEVForGepInnerOffset(*simpleNode, 2, pointeeType))
        offset = SCEVAddExpr::Create(std::move(offset), std::move(innerOffset));

      result = SCEVAddExpr::Create(std::move(baseScev), std::move(offset));
    }
    if (rvsdg::is<IntegerConstantOperation>(simpleNode->GetOperation()))
    {
      const auto constOp =
          dynamic_cast<const IntegerConstantOperation *>(&simpleNode->GetOperation());
      const auto value = constOp->Representation().to_int();
      result = SCEVConstant::Create(value);
    }
    if (rvsdg::is<IntegerBinaryOperation>(simpleNode->GetOperation()))
    {
      JLM_ASSERT(simpleNode->ninputs() == 2);
      const auto lhs = simpleNode->input(0)->origin();
      const auto rhs = simpleNode->input(1)->origin();

      auto lhsScev = GetOrCreateSCEVForOutput(*lhs);
      auto rhsScev = GetOrCreateSCEVForOutput(*rhs);
      if (rvsdg::is<IntegerAddOperation>(simpleNode->GetOperation()))
      {
        result = SCEVAddExpr::Create(std::move(lhsScev), std::move(rhsScev));
      }
      if (rvsdg::is<IntegerSubOperation>(simpleNode->GetOperation()))
      {
        auto rhsNegativeScev = GetNegativeSCEV(*rhsScev);

        result = SCEVAddExpr::Create(std::move(lhsScev), std::move(rhsNegativeScev));
      }
      if (rvsdg::is<IntegerMulOperation>(simpleNode->GetOperation()))
      {
        result = SCEVMulExpr::Create(std::move(lhsScev), std::move(rhsScev));
      }
    }
  }

  if (!result)
    // If none of the cases match, return an unknown SCEV expression
    result = SCEVUnknown::Create();

  // Save the result in the cache
  Context_->InsertSCEV(output, result);

  return result;
}

std::unique_ptr<SCEV>
ScalarEvolution::ComputeSCEVForGepInnerOffset(
    const rvsdg::SimpleNode & gepNode,
    const size_t inputIndex,
    const rvsdg::Type & type)
{
  JLM_ASSERT(inputIndex >= 2);

  if (inputIndex >= gepNode.ninputs())
  {
    return nullptr;
  }

  const auto gepInput = gepNode.input(inputIndex);
  if (const auto arrayType = dynamic_cast<const ArrayType *>(&type))
  {
    const auto & elementType = *arrayType->GetElementType();
    auto offset = SCEVMulExpr::Create(
        GetOrCreateSCEVForOutput(*gepInput->origin()),
        SCEVConstant::Create(GetTypeAllocSize(elementType)));

    auto subOffset = ComputeSCEVForGepInnerOffset(gepNode, inputIndex + 1, elementType);

    if (!subOffset)
      return offset;

    return SCEVAddExpr::Create(std::move(offset), std::move(subOffset));
  }
  if (const auto structType = dynamic_cast<const StructType *>(&type))
  {
    const auto indexingValue = tryGetConstantSignedInteger(*gepInput->origin());

    if (!indexingValue.has_value())
      return nullptr;

    const auto & fieldType = structType->getElementType(*indexingValue);

    auto offset = SCEVConstant::Create(structType->GetFieldOffset(*indexingValue));

    auto subOffset = ComputeSCEVForGepInnerOffset(gepNode, inputIndex + 1, *fieldType);

    if (!subOffset)
      return offset;

    return SCEVAddExpr::Create(std::move(offset), std::move(subOffset));
  }
  throw std::logic_error("Unknown GEP type!");
}

void
ScalarEvolution::FindDependenciesForSCEV(
    const SCEV & scev,
    DependencyMap & dependencies,
    const DependencyOp op = DependencyOp::None)
{
  if (const auto placeholderSCEV = dynamic_cast<const SCEVPlaceholder *>(&scev))
  {
    if (const auto dependency = placeholderSCEV->GetPrePointer())
    {
      // Retrieves dependency info struct from the map
      // In the case where the dependency does not already exist, a new struct is created with the
      // default count being 0 and the default operation being None
      auto & depInfo = dependencies[dependency];
      depInfo.operation = op;
      depInfo.count++;
    }
  }

  if (const auto addSCEV = dynamic_cast<const SCEVAddExpr *>(&scev))
  {
    FindDependenciesForSCEV(*addSCEV->GetLeftOperand(), dependencies, DependencyOp::Add);
    FindDependenciesForSCEV(*addSCEV->GetRightOperand(), dependencies, DependencyOp::Add);
  }

  if (const auto mulSCEV = dynamic_cast<const SCEVMulExpr *>(&scev))
  {
    // Only pass Mul down if we haven't already seen Add in the path from root
    // If op is already Add, preserve it; otherwise use Mul
    const DependencyOp opToPass = (op == DependencyOp::Add) ? DependencyOp::Add : DependencyOp::Mul;
    FindDependenciesForSCEV(*mulSCEV->GetLeftOperand(), dependencies, opToPass);
    FindDependenciesForSCEV(*mulSCEV->GetRightOperand(), dependencies, opToPass);
  }
}

ScalarEvolution::DependencyGraph
ScalarEvolution::CreateDependencyGraph(const rvsdg::ThetaNode & thetaNode) const
{
  DependencyGraph graph{};

  for (const auto & [output, scev] : Context_->GetSCEVMap())
  {
    DependencyMap dependencies{};
    if (const auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*output);
        theta == &thetaNode)
    {
      // We know this is a pre pointer, so we map it to loop var and use the SCEV for the
      // post's origin (backedge) instead
      const auto loopVar = theta->MapPreLoopVar(*output);
      auto newScev = Context_->TryGetSCEVForOutput(*loopVar.post->origin());

      FindDependenciesForSCEV(*newScev.get(), dependencies);
    }
    else
      FindDependenciesForSCEV(*scev.get(), dependencies);

    graph[output] = dependencies;
  }
  return graph;
}

// Implementation of Kahn's algorithm for topological sort
std::vector<const rvsdg::Output *>
ScalarEvolution::TopologicalSort(const DependencyGraph & dependencyGraph)
{
  const size_t numVertices = dependencyGraph.size();
  std::unordered_map<const rvsdg::Output *, int> indegree(numVertices);
  std::queue<const rvsdg::Output *> q{};
  for (const auto & [node, deps] : dependencyGraph)
  {
    for (const auto & dep : deps)
    {
      if (const auto ptr = dep.first; ptr == node)
        continue; // Ignore self-edges
      // To begin with, the indegree is just the number of incoming edges
      indegree[node] += 1;
    }
    if (indegree[node] == 0)
    {
      // Add nodes with no incoming edges to the queue, we know that these have no dependencies
      q.push(node);
    }
  }

  std::vector<const rvsdg::Output *> result{};
  while (!q.empty())
  {
    const rvsdg::Output * currentNode = q.front();
    q.pop();
    result.push_back(currentNode);

    for (const auto & [node, deps] : dependencyGraph)
    {
      if (node == currentNode)
        continue;

      for (const auto & dep : deps)
      {
        const auto ptr = dep.first;
        if (ptr == node)
          continue; // Skip self-edges
        if (ptr == currentNode)
        {
          // Update the indegree of nodes depending on this one
          indegree[node] -= 1;
          if (indegree[node] == 0)
            q.push(node);
        }
      }
    }
  }
  JLM_ASSERT(result.size() == numVertices);
  return result;
}

std::unique_ptr<SCEVChainRecurrence>
ScalarEvolution::GetOrCreateChainRecurrence(
    const rvsdg::Output & output,
    const SCEV & scev,
    const rvsdg::ThetaNode & thetaNode)
{
  if (const auto existing = Context_->TryGetChrecForOutput(output))
  {
    return SCEV::CloneAs<SCEVChainRecurrence>(*existing);
  }

  auto stepRecurrence = GetOrCreateStepForSCEV(output, scev, thetaNode);

  if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    // Find the start value for the recurrence
    const auto inputOrigin = thetaNode.MapPreLoopVar(output).input->origin();
    if (const auto constantInteger = tryGetConstantSignedInteger(*inputOrigin))
    {
      // If the input value is a constant, create a SCEV representation and set it as start
      // value (first operand in rec)
      stepRecurrence->AddOperandToFront(SCEVConstant::Create(*constantInteger));
    }
    else
    {
      // If not, create a SCEVInit node representing the start value
      stepRecurrence->AddOperandToFront(SCEVInit::Create(output));
    }
  }
  return stepRecurrence;
}

std::unique_ptr<SCEVChainRecurrence>
ScalarEvolution::GetOrCreateStepForSCEV(
    const rvsdg::Output & output,
    const SCEV & scevTree,
    const rvsdg::ThetaNode & thetaNode)
{
  if (const auto existing = Context_->TryGetChrecForOutput(output))
  {
    return SCEV::CloneAs<SCEVChainRecurrence>(*existing);
  }

  auto chrec = SCEVChainRecurrence::Create(thetaNode);

  if (const auto scevUnknown = dynamic_cast<const SCEVUnknown *>(&scevTree))
  {
    chrec->AddOperand(scevUnknown->Clone());
    return chrec;
  }
  if (const auto scevConstant = dynamic_cast<const SCEVConstant *>(&scevTree))
  {
    // This is a constant, we add it as the only operand
    chrec->AddOperand(scevConstant->Clone());
    return chrec;
  }
  if (dynamic_cast<const SCEVLoad *>(&scevTree))
  {
    // The load operation relies on memory, which we treat as opaque
    chrec->AddOperand(SCEVUnknown::Create());
    return chrec;
  }
  if (const auto scevPlaceholder = dynamic_cast<const SCEVPlaceholder *>(&scevTree))
  {
    if (scevPlaceholder->GetPrePointer() == &output)
    {
      // Since we are only interested in the step value, and not the initial value, we can ignore
      // ourselves by returning an empty chain recurrence (treated as the identity element - 0 for
      // addition and 1 for multiplication)
      return chrec;
    }
    if (auto storedRec = Context_->TryGetChrecForOutput(*scevPlaceholder->GetPrePointer()))
    {
      // We have a dependency of another IV
      // Get it's saved value. This is safe to do due to the topological ordering
      return storedRec;
    }
    chrec->AddOperand(SCEVUnknown::Create());
    return chrec;
  }
  if (const auto scevAddExpr = dynamic_cast<const SCEVAddExpr *>(&scevTree))
  {
    const auto lhsStep = GetOrCreateStepForSCEV(output, *scevAddExpr->GetLeftOperand(), thetaNode);
    const auto rhsStep = GetOrCreateStepForSCEV(output, *scevAddExpr->GetRightOperand(), thetaNode);

    return SCEV::CloneAs<SCEVChainRecurrence>(*ApplyAddFolding(lhsStep.get(), rhsStep.get()));
  }
  if (const auto scevMulExpr = dynamic_cast<const SCEVMulExpr *>(&scevTree))
  {
    const auto lhsStep = GetOrCreateStepForSCEV(output, *scevMulExpr->GetLeftOperand(), thetaNode);
    const auto rhsStep = GetOrCreateStepForSCEV(output, *scevMulExpr->GetRightOperand(), thetaNode);

    return SCEV::CloneAs<SCEVChainRecurrence>(*ApplyMulFolding(lhsStep.get(), rhsStep.get()));
  }
  throw std::logic_error("Invalid SCEV type when creating chrec!");
}

std::unique_ptr<SCEV>
ScalarEvolution::FoldNAryExpression(SCEVNAryExpr & expression)
{
  // In some cases, we end up with an n-ary expression like (1 + Init(a1) + 2).
  // This method folds the constant operands, turning it into (3 + Init(a1)).
  bool folded{};
  do
  {
    folded = false;
    for (size_t i = 0; i < expression.GetOperands().size(); ++i)
    {
      std::vector<const SCEV *> ops = expression.GetOperands();
      if (dynamic_cast<const SCEVInit *>(ops[i]))
        continue; // Cannot fold init
      for (size_t j = i + 1; j < expression.GetOperands().size(); ++j)
      {
        if (dynamic_cast<const SCEVInit *>(ops[j]))
          continue;

        // Both are foldable (constants or recurrences) fold them according to the rules
        std::unique_ptr<SCEV> foldedOperand{};
        if (dynamic_cast<SCEVNAryAddExpr *>(&expression))
        {
          foldedOperand = ApplyAddFolding(ops[i], ops[j]);
        }
        else if (dynamic_cast<SCEVNAryMulExpr *>(&expression))
        {
          foldedOperand = ApplyMulFolding(ops[i], ops[j]);
        }
        else
        {
          throw std::logic_error("Invalid n-ary SCEV expression type in FoldNAryExpression!");
        }
        expression.RemoveOperand(j);
        expression.ReplaceOperand(i, foldedOperand);
        folded = true;
        break;
      }
      if (folded)
        break;
    }
  } while (folded);

  if (expression.GetOperands().size() == 1)
  {
    // If there is only one operand in the n-ary expression, we just return the operand
    return expression.GetOperand(0)->Clone();
  }

  return expression.Clone();
}

std::unique_ptr<SCEV>
ScalarEvolution::ApplyAddFolding(const SCEV * lhsOperand, const SCEV * rhsOperand)
{
  // We have the following folding rules from the CR algebra:
  // G + {e,+,f}         =>       {G + e,+,f}         (1)
  // {e,+,f} + {g,+,h}   =>       {e + g,+,f + h}     (2)
  //
  // And by generalizing rule 2, we have that:
  // {G,+,0} + {e,+,f} = {G + e,+,0 + f} = {G + e,+,f}
  //
  // Since we represent constants in the SCEVTree as recurrences consisting of only a SCEVConstant
  // node, we can therefore pad the constant recurrence with however many zeroes we need for the
  // length of the other recurrence. This effectively lets us apply both rules in one go.
  //
  // For constants and unknowns this is trivial, however it becomes a bit complicated when we
  // factor in SCEVInit nodes. These nodes represent the initial value of an IV in the case where
  // the exact value is unknown at compile time. E.g. function argument or result from a
  // call-instruction. In the cases where we have to fold one or more of these init-nodes, we
  // create an n-ary add expression (add expression with an arbitrary number of operands), and add
  // this to the chrec. Folding two of these n-ary add expressions will result in another n-ary
  // add expression, which consists of all the operands in both the left and the right expression.

  // The if-chain below goes through each of the possible combinations of lhs and rhs values
  if (const auto *lhsUnknown = dynamic_cast<const SCEVUnknown *>(lhsOperand),
      *rhsUnknown = dynamic_cast<const SCEVUnknown *>(rhsOperand);
      lhsUnknown || rhsUnknown)
  {
    // If one of the sides is unknown. Return unknown
    return SCEVUnknown::Create();
  }

  const auto lhsChrec = dynamic_cast<const SCEVChainRecurrence *>(lhsOperand);
  const auto rhsChrec = dynamic_cast<const SCEVChainRecurrence *>(rhsOperand);
  if (lhsChrec && rhsChrec)
  {
    auto newChrec = SCEVChainRecurrence::Create(*lhsChrec->GetLoop());
    if (lhsChrec->GetLoop() != rhsChrec->GetLoop())
    {
      newChrec->AddOperand(SCEVNAryAddExpr::Create(lhsChrec->Clone(), rhsChrec->Clone()));
      return newChrec;
    }

    const auto lhsSize = lhsChrec->GetOperands().size();
    const auto rhsSize = rhsChrec->GetOperands().size();
    for (size_t i = 0; i < std::max(lhsSize, rhsSize); ++i)
    {
      const SCEV * lhs{};
      const SCEV * rhs{};
      if (i < lhsSize)
        lhs = lhsChrec->GetOperand(i);

      if (i < rhsSize)
        rhs = rhsChrec->GetOperand(i);
      newChrec->AddOperand(ApplyAddFolding(lhs, rhs));
    }
    return newChrec;
  }

  // Chrec + any other operand
  // This handles Init, Constant, and any other SCEV type uniformly
  if (lhsChrec || rhsChrec)
  {
    auto * chrec = lhsChrec ? lhsChrec : rhsChrec;
    const auto * otherOperand = lhsChrec ? rhsOperand : lhsOperand;

    // Skip if otherOperand is zero constant (identity for addition)
    if (const auto constant = dynamic_cast<const SCEVConstant *>(otherOperand))
    {
      if (!SCEVConstant::IsNonZero(constant))
      {
        return chrec->Clone();
      }
    }
    auto newChrec = SCEVChainRecurrence::Create(*chrec->GetLoop());
    const auto chrecOperands = chrec->GetOperands();

    bool isFirst = true;
    for (const auto operand : chrecOperands)
    {
      if (isFirst)
      {
        // Recursively fold the start value with the other operand
        newChrec->AddOperand(ApplyAddFolding(operand, otherOperand));
        isFirst = false;
      }
      else
      {
        newChrec->AddOperand(operand->Clone());
      }
    }
    return newChrec;
  }

  const auto lhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(lhsOperand);
  const auto rhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(rhsOperand);
  // Handle n-ary multiply expressions - they become terms in an n-ary add expression
  if (lhsNAryMulExpr && rhsNAryMulExpr)
  {
    // Two multiply expressions - create add expression with both
    return SCEVNAryAddExpr::Create(lhsNAryMulExpr->Clone(), rhsNAryMulExpr->Clone());
  }

  const auto lhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(lhsOperand);
  const auto rhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(rhsOperand);
  if ((lhsNAryMulExpr && rhsNAryAddExpr) || (rhsNAryMulExpr && lhsNAryAddExpr))
  {
    // Multiply expression with add expression - Clone the add expression and add the multiply as
    // a term
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    auto * addExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    auto newAddExpr = SCEV::CloneAs<SCEVNAryExpr>(*addExpr);
    newAddExpr->AddOperand(mulExpr->Clone());
    return newAddExpr->Clone();
  }

  const auto lhsInit = dynamic_cast<const SCEVInit *>(lhsOperand);
  const auto rhsInit = dynamic_cast<const SCEVInit *>(rhsOperand);
  if ((lhsNAryMulExpr && rhsInit) || (rhsNAryMulExpr && lhsInit))
  {
    // Multiply expression with init - create add expression
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    const auto * init = lhsInit ? lhsInit : rhsInit;
    return SCEVNAryAddExpr::Create(mulExpr->Clone(), init->Clone());
  }

  const auto lhsConstant = dynamic_cast<const SCEVConstant *>(lhsOperand);
  const auto rhsConstant = dynamic_cast<const SCEVConstant *>(rhsOperand);
  if ((lhsNAryMulExpr && SCEVConstant::IsNonZero(rhsConstant))
      || (rhsNAryMulExpr && SCEVConstant::IsNonZero(lhsConstant)))
  {
    // Multiply expression with nonzero constant - create add expression
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return SCEVNAryAddExpr::Create(mulExpr->Clone(), constant->Clone());
  }

  if (lhsNAryMulExpr || rhsNAryMulExpr)
  {
    // Single multiply expression, no folding necessary
    const auto * mulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    return mulExpr->Clone();
  }

  if (lhsInit && rhsInit)
  {
    // We have two init nodes. Create a nAryAdd with lhsInit and rhsInit
    return SCEVNAryAddExpr::Create(lhsInit->Clone(), rhsInit->Clone());
  }

  if ((lhsInit && rhsNAryAddExpr) || (rhsInit && lhsNAryAddExpr))
  {
    // We have an init and an add expr. Clone the add expression and add the init as an operand
    const auto * init = lhsInit ? lhsInit : rhsInit;
    auto * nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    auto newAddExpr = SCEV::CloneAs<SCEVNAryAddExpr>(*nAryAddExpr);
    newAddExpr->AddOperand(init->Clone());
    return newAddExpr->Clone();
  }

  if ((lhsInit && SCEVConstant::IsNonZero(rhsConstant))
      || (rhsInit && SCEVConstant::IsNonZero(lhsConstant)))
  {
    // We have an init and a nonzero constant. Create a nAryAdd with init and constant
    const auto * init = lhsInit ? lhsInit : rhsInit;
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return SCEVNAryAddExpr::Create(init->Clone(), constant->Clone());
  }

  if (lhsInit || rhsInit)
  {
    // Only one operand. Add it
    const auto * init = lhsInit ? lhsInit : rhsInit;
    return init->Clone();
  }

  if (lhsNAryAddExpr && rhsNAryAddExpr)
  {
    // We have two add expressions. Clone the lhs and add the rhs operands
    auto lhsNewNAryAddExpr = SCEV::CloneAs<SCEVNAryAddExpr>(*lhsNAryAddExpr);
    for (auto op : rhsNAryAddExpr->GetOperands())
    {
      lhsNewNAryAddExpr->AddOperand(op->Clone());
    }
    return lhsNewNAryAddExpr;
  }

  if ((lhsNAryAddExpr && SCEVConstant::IsNonZero(rhsConstant))
      || (rhsNAryAddExpr && SCEVConstant::IsNonZero(lhsConstant)))
  {
    // We have an add expr and a nonzero constant. Clone the add expr and add the constant
    auto * nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    auto newNAryAddExpr = SCEV::CloneAs<SCEVNAryAddExpr>(*nAryAddExpr);

    // Check if there is already a constant operand in the n-ary expression
    // If so, fold the new constant with the old one instead of adding it as an operand
    bool folded = false;
    for (size_t i = 0; i < newNAryAddExpr->GetOperands().size(); ++i)
    {
      if (const auto existingConstant =
              dynamic_cast<const SCEVConstant *>(newNAryAddExpr->GetOperands()[i]))
      {
        // Fold the two constants together directly
        auto foldedConstant = ApplyAddFolding(existingConstant, constant);
        newNAryAddExpr->ReplaceOperand(i, foldedConstant);
        folded = true;
        break;
      }
    }

    if (!folded)
    {
      // No existing constant to fold with, just append
      newNAryAddExpr->AddOperand(constant->Clone());
    }

    return newNAryAddExpr;
  }

  if (lhsNAryAddExpr || rhsNAryAddExpr)
  {
    const auto * nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    return nAryAddExpr->Clone();
  }
  if (lhsConstant && rhsConstant)
  {
    // Two constants, get their value, and combine them (fold)
    const auto lhsValue = lhsConstant->GetValue();
    const auto rhsValue = rhsConstant->GetValue();

    return SCEVConstant::Create(lhsValue + rhsValue);
  }

  if (lhsConstant || rhsConstant)
  {
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return constant->Clone();
  }

  return SCEVUnknown::Create();
}

std::unique_ptr<SCEV>
ScalarEvolution::ApplyMulFolding(const SCEV * lhsOperand, const SCEV * rhsOperand)
{
  // We have the following folding rules from the CR algebra:
  // G * {e,+,f}         =>       {G * e,+,G * f}
  // {e,+,f} * {g,+,h}   =>       {e * g,+,e * h + f * g + f * h,+,2*f*h}
  //
  // Similar to addition, we need to handle SCEVInit nodes and n-ary expressions.
  // For multiplication with init nodes, we create n-ary multiply expressions.

  if (const auto *lhsUnknown = dynamic_cast<const SCEVUnknown *>(lhsOperand),
      *rhsUnknown = dynamic_cast<const SCEVUnknown *>(rhsOperand);
      lhsUnknown || rhsUnknown)
  {
    return SCEVUnknown::Create();
  }

  const auto lhsChrec = dynamic_cast<const SCEVChainRecurrence *>(lhsOperand);
  const auto rhsChrec = dynamic_cast<const SCEVChainRecurrence *>(rhsOperand);
  if (lhsChrec && rhsChrec)
  {
    if (lhsChrec->GetLoop() != rhsChrec->GetLoop())
    {
      return SCEVNAryMulExpr::Create(lhsChrec->Clone(), rhsChrec->Clone());
    }

    auto newChrec = SCEVChainRecurrence::Create(*lhsChrec->GetLoop());
    const auto lhsSize = lhsChrec->GetOperands().size();
    const auto rhsSize = rhsChrec->GetOperands().size();

    if (lhsSize == 0)
    {
      for (auto operand : rhsChrec->GetOperands())
      {
        newChrec->AddOperand(operand->Clone());
      }
    }
    else if (rhsSize == 0)
    {
      for (auto operand : lhsChrec->GetOperands())
      {
        newChrec->AddOperand(operand->Clone());
      }
    }
    // Handle G * {e,+,f,...} where G is loop invariant
    if (lhsSize == 1)
    {
      // G * {e,+,f,...} = {G * e,+,G * f,...}
      auto lhs = lhsChrec->GetOperand(0);

      for (auto rhs : rhsChrec->GetOperands())
      {
        newChrec->AddOperand(ApplyMulFolding(lhs, rhs));
      }
    }
    else if (rhsSize == 1)
    {
      // {e,+,f,...} * G = {e * G,+,f * G,...}
      auto rhs = rhsChrec->GetOperand(0);

      for (auto lhs : lhsChrec->GetOperands())
      {
        newChrec->AddOperand(ApplyMulFolding(lhs, rhs));
      }
    }
    else if (lhsSize == 2 && rhsSize == 2)
    {
      // {e,+,f} * {g,+,h} = {e*g,+,e*h + f*g + f*h,+,2*f*h}
      const auto e = lhsChrec->GetOperand(0);
      const auto f = lhsChrec->GetOperand(1);
      const auto g = rhsChrec->GetOperand(0);
      const auto h = rhsChrec->GetOperand(1);

      // First step: e * g
      newChrec->AddOperand(ApplyMulFolding(e, g));

      // Second step: e * h + f * g + f * h
      const auto eh = ApplyMulFolding(e, h);
      const auto fg = ApplyMulFolding(f, g);
      const auto fh = ApplyMulFolding(f, h);
      const auto sum1 = ApplyAddFolding(eh.get(), fg.get());
      auto sum2 = ApplyAddFolding(sum1.get(), fh.get());
      newChrec->AddOperand(std::move(sum2));

      // Third step: 2 * f * h
      const auto two = SCEVConstant::Create(2);
      newChrec->AddOperand(ApplyMulFolding(two.get(), fh.get()));
    }
    else
    {
      // For other cases, return unknown
      newChrec->AddOperand(SCEVUnknown::Create());
    }
    return newChrec;
  }

  // Chrec * any other operand
  // This handles Init, Constant, and any other SCEV type uniformly
  if (lhsChrec || rhsChrec)
  {
    auto * chrec = lhsChrec ? lhsChrec : rhsChrec;
    const auto * otherOperand = lhsChrec ? rhsOperand : lhsOperand;

    // Skip if other operand is constant one (identity for multiplication)
    if (auto constant = dynamic_cast<const SCEVConstant *>(otherOperand))
    {
      if (constant->GetValue() == 1)
      {
        return chrec->Clone();
      }
    }
    auto newChrec = SCEVChainRecurrence::Create(*chrec->GetLoop());
    auto chrecOperands = chrec->GetOperands();

    for (size_t i = 0; i < chrecOperands.size(); ++i)
    {
      auto operand = chrecOperands[i];
      // Recursively fold the start value with the other operand
      newChrec->AddOperand(ApplyMulFolding(operand, otherOperand));
    }
    return newChrec;
  }

  const auto lhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(lhsOperand);
  const auto rhsNAryAddExpr = dynamic_cast<const SCEVNAryAddExpr *>(rhsOperand);
  if (lhsNAryAddExpr || rhsNAryAddExpr)
  {
    // Handle n-ary add expressions - distribute multiplication
    // (a + b + c) × G = a×G + b×G + c×G
    const auto nAryAddExpr = lhsNAryAddExpr ? lhsNAryAddExpr : rhsNAryAddExpr;
    const auto other = lhsNAryAddExpr ? rhsOperand : lhsOperand;

    auto resultAddExpr = SCEVNAryAddExpr::Create();
    for (auto operand : nAryAddExpr->GetOperands())
    {
      auto product = ApplyMulFolding(operand, other);
      resultAddExpr->AddOperand(std::move(product));
    }
    return resultAddExpr;
  }

  const auto lhsInit = dynamic_cast<const SCEVInit *>(lhsOperand);
  const auto rhsInit = dynamic_cast<const SCEVInit *>(rhsOperand);
  if (lhsInit && rhsInit)
  {
    // Two init nodes - create n-ary multiply expression
    return SCEVNAryMulExpr::Create(lhsInit->Clone(), rhsInit->Clone());
  }

  const auto lhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(lhsOperand);
  const auto rhsNAryMulExpr = dynamic_cast<const SCEVNAryMulExpr *>(rhsOperand);
  if ((lhsInit && rhsNAryMulExpr) || (rhsInit && lhsNAryMulExpr))
  {
    // Init node with n-ary multiply expression - Clone mult expr and add init as an operand
    const auto * init = lhsInit ? lhsInit : rhsInit;
    auto * nAryMulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    auto newNAryMulExpr = SCEV::CloneAs<SCEVNAryMulExpr>(*nAryMulExpr);
    newNAryMulExpr->AddOperand(init->Clone());
    return newNAryMulExpr->Clone();
  }

  const auto lhsConstant = dynamic_cast<const SCEVConstant *>(lhsOperand);
  const auto rhsConstant = dynamic_cast<const SCEVConstant *>(rhsOperand);
  if ((lhsInit && rhsConstant && rhsConstant->GetValue() != 1)
      || (rhsInit && lhsConstant && lhsConstant->GetValue() != 1))
  {
    // Init node with non-one constant - create n-ary multiply expression
    const auto * init = lhsInit ? lhsInit : rhsInit;
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return SCEVNAryMulExpr::Create(init->Clone(), constant->Clone());
  }

  if (lhsInit || rhsInit)
  {
    // Single init node, no folding necessary
    const auto * init = lhsInit ? lhsInit : rhsInit;
    return init->Clone();
  }

  if (lhsNAryMulExpr && rhsNAryMulExpr)
  {
    // Two n-ary mult expressions - combine operands
    auto lhsNewNAryMulExpr = SCEV::CloneAs<SCEVNAryMulExpr>(*lhsNAryMulExpr);
    for (auto op : rhsNAryMulExpr->GetOperands())
    {
      lhsNewNAryMulExpr->AddOperand(op->Clone());
    }
    return lhsNewNAryMulExpr;
  }

  if ((lhsNAryMulExpr && rhsConstant && rhsConstant->GetValue() != 1)
      || (rhsNAryMulExpr && lhsConstant && lhsConstant->GetValue() != 1))
  {
    // N-ary mult expression with non-one constant - Clone mult expression and add constant
    auto * nAryMulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    auto * constant = lhsConstant ? lhsConstant : rhsConstant;

    auto newNAryMulExpr = SCEV::CloneAs<SCEVNAryMulExpr>(*nAryMulExpr);

    bool folded = false;
    for (size_t i = 0; i < newNAryMulExpr->GetOperands().size(); ++i)
    {
      if (const auto existingConstant =
              dynamic_cast<const SCEVConstant *>(newNAryMulExpr->GetOperands()[i]))
      {
        // Fold the two constants together directly
        auto foldedConstant = ApplyMulFolding(existingConstant, constant);
        newNAryMulExpr->ReplaceOperand(i, foldedConstant);
        folded = true;
        break;
      }
    }

    if (!folded)
    {
      // No existing constant to fold with, just append
      newNAryMulExpr->AddOperand(constant->Clone());
    }

    return newNAryMulExpr;
  }

  if (lhsNAryMulExpr || rhsNAryMulExpr)
  {
    const auto * nAryMulExpr = lhsNAryMulExpr ? lhsNAryMulExpr : rhsNAryMulExpr;
    return nAryMulExpr->Clone();
  }

  if (lhsConstant && rhsConstant)
  {
    // Two constants - fold by multiplying values together
    const auto lhsValue = lhsConstant->GetValue();
    const auto rhsValue = rhsConstant->GetValue();
    return SCEVConstant::Create(lhsValue * rhsValue);
  }

  if (lhsConstant || rhsConstant)
  {
    const auto * constant = lhsConstant ? lhsConstant : rhsConstant;
    return constant->Clone();
  }

  return SCEVUnknown::Create();
}

std::unique_ptr<SCEV>
ScalarEvolution::GetNegativeSCEV(const SCEV & scev)
{
  // -(c)
  if (const auto c = dynamic_cast<const SCEVConstant *>(&scev))
  {
    const auto value = c->GetValue();
    return SCEVConstant::Create(-value);
  }
  // -(-x) -> x
  if (const auto mul = dynamic_cast<const SCEVMulExpr *>(&scev))
  {
    if (const auto c = dynamic_cast<const SCEVConstant *>(mul->GetLeftOperand());
        c && c->GetValue() == -1)
    {
      return mul->GetRightOperand()->Clone();
    }
    if (const auto c = dynamic_cast<const SCEVConstant *>(mul->GetRightOperand());
        c && c->GetValue() == -1)
    {
      return mul->GetLeftOperand()->Clone();
    }
  } // -(x + y) -> (-x) + (-y)
  if (const auto add = dynamic_cast<const SCEVAddExpr *>(&scev))
  {
    return SCEVAddExpr::Create(
        GetNegativeSCEV(*add->GetLeftOperand()),
        GetNegativeSCEV(*add->GetRightOperand()));
  }
  // General case: -(x) -> (-1) * x
  return SCEVMulExpr::Create(SCEVConstant::Create(-1), scev.Clone());
}

bool
ScalarEvolution::IsUnknown(const SCEVChainRecurrence & chrec)
{
  for (const auto operand : chrec.GetOperands())
  {
    if (dynamic_cast<const SCEVUnknown *>(operand))
    {
      return true;
    }
  }
  return false;
}

bool
ScalarEvolution::CanCreateChainRecurrence(
    const rvsdg::Output & output,
    DependencyGraph & dependencyGraph)
{
  auto deps = dependencyGraph[&output];
  if (deps.find(&output) != deps.end())
    if (deps[&output].count != 1)
    {
      // First check that variable has only one self-reference
      return false;
    }

  // Check that it has no reference via a mult-operation
  for (auto [out, dependencyInfo] : deps)
  {
    if (dependencyInfo.operation == DependencyOp::Mul)
    {
      return false;
    }
  }

  // Then check for cycles through other variables
  std::unordered_set<const rvsdg::Output *> visited{};
  std::unordered_set<const rvsdg::Output *> recursionStack{};
  return !HasCycleThroughOthers(output, output, dependencyGraph, visited, recursionStack);
}

bool
ScalarEvolution::HasCycleThroughOthers(
    const rvsdg::Output & currentOutput,
    const rvsdg::Output & originalOutput,
    DependencyGraph & dependencyGraph,
    std::unordered_set<const rvsdg::Output *> & visited,
    std::unordered_set<const rvsdg::Output *> & recursionStack)
{
  visited.insert(&currentOutput);
  recursionStack.insert(&currentOutput);

  for (const auto & [depPtr, depCount] : dependencyGraph[&currentOutput])
  {
    // Ignore self-references
    if (depPtr == &currentOutput)
      continue;

    // Found a cycle back to the ORIGINAL node we started from
    // This means the original output is explicitly part of the cycle
    if (depPtr == &originalOutput)
      return true;

    // Already explored this branch, no cycle containing the original output
    if (visited.find(depPtr) != visited.end())
      continue;

    // Recursively check dependencies, keeping track of the original node
    if (HasCycleThroughOthers(*depPtr, originalOutput, dependencyGraph, visited, recursionStack))
      return true;
  }

  recursionStack.erase(&currentOutput);
  return false;
}

bool
ScalarEvolution::StructurallyEqual(const SCEV & a, const SCEV & b)
{
  if (typeid(a) != typeid(b))
    return false;

  if (dynamic_cast<const SCEVUnknown *>(&a))
    return true;

  if (auto * constantA = dynamic_cast<const SCEVConstant *>(&a))
  {
    auto * constantB = dynamic_cast<const SCEVConstant *>(&b);
    return constantA->GetValue() == constantB->GetValue();
  }

  if (auto * initA = dynamic_cast<const SCEVInit *>(&a))
  {
    auto * initB = dynamic_cast<const SCEVInit *>(&b);
    return initA->GetPrePointer() == initB->GetPrePointer();
  }

  if (auto * binaryExprA = dynamic_cast<const SCEVBinaryExpr *>(&a))
  {
    auto * binaryExprB = dynamic_cast<const SCEVBinaryExpr *>(&b);
    return StructurallyEqual(*binaryExprA->GetLeftOperand(), *binaryExprB->GetLeftOperand())
        && StructurallyEqual(*binaryExprA->GetRightOperand(), *binaryExprB->GetRightOperand());
  }

  if (auto * chrecA = dynamic_cast<const SCEVChainRecurrence *>(&a))
  {
    auto * chrecB = dynamic_cast<const SCEVChainRecurrence *>(&b);
    if (chrecA->GetLoop() != chrecB->GetLoop())
      return false;
    if (chrecA->GetOperands().size() != chrecB->GetOperands().size())
      return false;
    for (size_t i = 0; i < chrecA->GetOperands().size(); ++i)
    {
      if (!StructurallyEqual(*chrecA->GetOperands()[i], *chrecB->GetOperands()[i]))
        return false;
    }
    return true;
  }

  if (auto * nAryExprA = dynamic_cast<const SCEVNAryExpr *>(&a))
  {
    auto * nAryExprB = dynamic_cast<const SCEVNAryExpr *>(&b);
    if (nAryExprA->GetOperands().size() != nAryExprB->GetOperands().size())
      return false;
    for (size_t i = 0; i < nAryExprA->GetOperands().size(); ++i)
    {
      if (!StructurallyEqual(*nAryExprA->GetOperands()[i], *nAryExprB->GetOperands()[i]))
        return false;
    }
    return true;
  }

  return false;
}
}
