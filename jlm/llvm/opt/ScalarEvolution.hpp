/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_SCALAR_EVOLUTION_HPP
#define JLM_LLVM_OPT_SCALAR_EVOLUTION_HPP

#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::llvm
{
class SCEV
{
  friend class ScalarEvolution;

public:
  virtual ~SCEV() noexcept = default;

  virtual std::string
  DebugString() const = 0;

  virtual std::unique_ptr<SCEV>
  Clone() const = 0;

  template<typename T>
  static std::unique_ptr<T>
  CloneAs(const SCEV & scev)
  {
    auto cloned = scev.Clone();
    auto * ptr = dynamic_cast<T *>(cloned.release());
    JLM_ASSERT(ptr);
    return std::unique_ptr<T>(ptr);
  }
};

class SCEVUnknown final : public SCEV
{
public:
  explicit SCEVUnknown()
  {}

  std::string
  DebugString() const override
  {
    return "Unknown";
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    return std::make_unique<SCEVUnknown>();
  }

  static std::unique_ptr<SCEVUnknown>
  Create()
  {
    return std::make_unique<SCEVUnknown>();
  }
};

class SCEVInit final : public SCEV
{
public:
  explicit SCEVInit(rvsdg::Output & pre, rvsdg::ThetaNode & loop)
      : PrePointer_{ &pre },
        Loop_{ &loop }
  {}

  rvsdg::Output *
  GetPrePointer() const
  {
    return PrePointer_;
  }

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    oss << "Init(" << PrePointer_->debug_string() << ")";
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    return std::make_unique<SCEVInit>(*PrePointer_, *Loop_);
  }

  static std::unique_ptr<SCEVInit>
  Create(rvsdg::Output & prePointer, rvsdg::ThetaNode & loop)
  {
    return std::make_unique<SCEVInit>(prePointer, loop);
  }

  rvsdg::ThetaNode *
  GetLoop() const
  {
    return Loop_;
  }

private:
  rvsdg::Output * PrePointer_;

protected:
  rvsdg::ThetaNode * Loop_;
};

class SCEVPlaceholder final : public SCEV
{
public:
  explicit SCEVPlaceholder(rvsdg::Output & pre)
      : PrePointer_{ &pre }
  {}

  rvsdg::Output *
  GetPrePointer() const
  {
    return PrePointer_;
  }

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    oss << "PH(" << PrePointer_->debug_string() << ")";
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    return std::make_unique<SCEVPlaceholder>(*PrePointer_);
  }

  static std::unique_ptr<SCEVPlaceholder>
  Create(rvsdg::Output & PrePointer_)
  {
    return std::make_unique<SCEVPlaceholder>(PrePointer_);
  }

private:
  rvsdg::Output * PrePointer_;
};

class SCEVConstant final : public SCEV
{
public:
  explicit SCEVConstant(const int64_t value)
      : Value_{ value }
  {}

  int64_t
  GetValue() const
  {
    return Value_;
  }

  std::string
  DebugString() const override
  {
    return std::to_string(Value_);
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    return std::make_unique<SCEVConstant>(Value_);
  }

  static std::unique_ptr<SCEVConstant>
  Create(const int64_t value)
  {
    return std::make_unique<SCEVConstant>(value);
  }

  static bool
  IsNonZero(const SCEVConstant * c)
  {
    return c && c->GetValue() != 0;
  }

private:
  int64_t Value_;
};

class SCEVBinaryExpr : public SCEV
{
public:
  SCEVBinaryExpr()
      : LeftOperand_{},
        RightOperand_{}
  {}

  SCEVBinaryExpr(std::unique_ptr<SCEV> left, std::unique_ptr<SCEV> right)
      : LeftOperand_{ std::move(left) },
        RightOperand_{ std::move(right) }
  {}

  SCEV *
  GetLeftOperand() const
  {
    return LeftOperand_.get();
  }

  SCEV *
  GetRightOperand() const
  {
    return RightOperand_.get();
  }

  void
  SetLeftOperand(std::unique_ptr<SCEV> op)
  {
    LeftOperand_ = std::move(op);
  }

  void
  SetRightOperand(std::unique_ptr<SCEV> op)
  {
    RightOperand_ = std::move(op);
  }

protected:
  std::unique_ptr<SCEV> LeftOperand_;
  std::unique_ptr<SCEV> RightOperand_;
};

class SCEVAddExpr final : public SCEVBinaryExpr
{
public:
  SCEVAddExpr(std::unique_ptr<SCEV> left, std::unique_ptr<SCEV> right)
      : SCEVBinaryExpr(std::move(left), std::move(right))
  {}

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    const std::string leftStr = LeftOperand_ ? LeftOperand_->DebugString() : "null";
    const std::string rightStr = RightOperand_ ? RightOperand_->DebugString() : "null";
    oss << "(" << leftStr << " + " << rightStr << ")";
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    std::unique_ptr<SCEV> leftClone = LeftOperand_ ? LeftOperand_->Clone() : nullptr;
    std::unique_ptr<SCEV> rightClone = RightOperand_ ? RightOperand_->Clone() : nullptr;
    return std::make_unique<SCEVAddExpr>(std::move(leftClone), std::move(rightClone));
  }

  static std::unique_ptr<SCEVAddExpr>
  Create(std::unique_ptr<SCEV> left, std::unique_ptr<SCEV> right)
  {
    return std::make_unique<SCEVAddExpr>(std::move(left), std::move(right));
  }
};

class SCEVMulExpr final : public SCEVBinaryExpr
{
public:
  SCEVMulExpr(std::unique_ptr<SCEV> left, std::unique_ptr<SCEV> right)
      : SCEVBinaryExpr(std::move(left), std::move(right))
  {}

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    const std::string leftStr = LeftOperand_ ? LeftOperand_->DebugString() : "null";
    const std::string rightStr = RightOperand_ ? RightOperand_->DebugString() : "null";
    oss << "(" << leftStr << " * " << rightStr << ")";
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    std::unique_ptr<SCEV> leftClone = LeftOperand_ ? LeftOperand_->Clone() : nullptr;
    std::unique_ptr<SCEV> rightClone = RightOperand_ ? RightOperand_->Clone() : nullptr;
    return std::make_unique<SCEVMulExpr>(std::move(leftClone), std::move(rightClone));
  }

  static std::unique_ptr<SCEVMulExpr>
  Create(std::unique_ptr<SCEV> left, std::unique_ptr<SCEV> right)
  {
    return std::make_unique<SCEVMulExpr>(std::move(left), std::move(right));
  }
};

class SCEVNAryExpr : public SCEV
{
public:
  explicit SCEVNAryExpr()
      : Operands_{}
  {}

  template<typename... Args>
  explicit SCEVNAryExpr(Args &&... operands)
      : Operands_{}
  {
    (AddOperand(std::forward<Args>(operands)), ...);
  }

  template<typename... Args>
  void
  AddOperands(Args &&... operands)
  {
    (AddOperand(std::forward<Args>(operands)), ...);
  }

  void
  AddOperand(std::unique_ptr<SCEV> scev)
  {
    Operands_.push_back(std::move(scev));
  }

  void
  ReplaceOperand(const size_t index, const std::unique_ptr<SCEV> & operand)
  {
    Operands_[index] = operand->Clone();
  }

  void
  RemoveOperand(const size_t index)
  {
    if (index < Operands_.size())
    {
      Operands_.erase(Operands_.begin() + index);
    }
  }

  std::vector<SCEV *>
  GetOperands() const
  {
    std::vector<SCEV *> operands{};
    for (auto & op : Operands_)
    {
      operands.push_back(op.get());
    }
    return operands;
  }

  SCEV *
  GetOperand(const size_t index) const
  {
    return Operands_.at(index).get();
  }

  size_t
  NumOperands() const
  {
    return Operands_.size();
  }

  /**
   * Checks the operands of the given \p chrec to see if any of them are unknown.
   *
   * @param chrec the chain recurrence to be checked
   * @return true if the recurrence contains an unknown, false otherwise
   */
  bool static IsUnknown(const SCEVNAryExpr & chrec)
  {
    for (const auto operand : chrec.GetOperands())
    {
      if (auto operandNary = dynamic_cast<const SCEVNAryExpr *>(operand))
      {
        if (IsUnknown(*operandNary))
        {
          return true;
        }
      }
      else if (dynamic_cast<const SCEVUnknown *>(operand))
      {
        return true;
      }
    }
    return false;
  }

protected:
  std::vector<std::unique_ptr<SCEV>> Operands_;
};

class SCEVChainRecurrence final : public SCEVNAryExpr
{
public:
  explicit SCEVChainRecurrence(rvsdg::ThetaNode & theta, rvsdg::Output & output)
      : SCEVNAryExpr(),
        Loop_{ &theta },
        Output_{ &output }
  {}

  template<typename... Args>
  explicit SCEVChainRecurrence(
      rvsdg::ThetaNode & theta,
      rvsdg::Output & output,
      Args &&... operands)
      : SCEVNAryExpr(std::forward<Args>(operands)...),
        Loop_{ &theta },
        Output_{ &output }
  {}

  rvsdg::ThetaNode *
  GetLoop() const
  {
    return Loop_;
  }

  rvsdg::Output *
  GetOutput() const
  {
    return Output_;
  }

  SCEV *
  GetStartValue() const
  {
    return Operands_[0].get();
  }

  bool static IsConstant(const SCEVChainRecurrence & chrec)
  {
    return chrec.GetOperands().size() == 1;
  }

  bool static IsAffine(const SCEVChainRecurrence & chrec)
  {
    return chrec.GetOperands().size() == 2;
  }

  bool static IsQuadratic(const SCEVChainRecurrence & chrec)
  {
    return chrec.GetOperands().size() == 3;
  }

  bool static IsInvariantInLoop(const SCEVChainRecurrence & chrec, const rvsdg::ThetaNode & theta)
  {
    return chrec.GetLoop() != &theta;
  }

  std::optional<std::unique_ptr<SCEV>>
  GetStep() const
  {
    if (Operands_.size() < 2)
    {
      return std::nullopt;
    }
    if (Operands_.size() == 2)
    {
      return Operands_[1]->Clone();
    }
    auto newRec = SCEVChainRecurrence::Create(*Loop_, *Output_);
    for (auto & operand : util::IteratorRange(std::next(Operands_.begin()), Operands_.end()))
    {
      newRec->AddOperand(operand->Clone());
    }
    return newRec;
  }

  void
  AddOperandToFront(const std::unique_ptr<SCEV> & initScev)
  {
    Operands_.insert(Operands_.begin(), initScev->Clone());
  }

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < Operands_.size(); ++i)
    {
      oss << Operands_.at(i)->DebugString();
      if (i < Operands_.size() - 1)
        oss << ",+,";
    }
    oss << "}" << "<" << Loop_->subregion()->getRegionId() << ">";
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    auto copy = std::make_unique<SCEVChainRecurrence>(*Loop_, *Output_);
    for (const auto & op : Operands_)
    {
      copy->AddOperand(op->Clone());
    }
    return copy;
  }

  static std::unique_ptr<SCEVChainRecurrence>
  Create(rvsdg::ThetaNode & loop, rvsdg::Output & output)
  {
    return std::make_unique<SCEVChainRecurrence>(loop, output);
  }

  template<typename... Args>
  static std::unique_ptr<SCEVChainRecurrence>
  Create(rvsdg::ThetaNode & loop, rvsdg::Output & output, Args &&... operands)
  {
    return std::make_unique<SCEVChainRecurrence>(loop, output, std::forward<Args>(operands)...);
  }

protected:
  rvsdg::ThetaNode * Loop_;
  rvsdg::Output * Output_;
};

class SCEVNAryAddExpr final : public SCEVNAryExpr
{
  friend class ScalarEvolution;

public:
  explicit SCEVNAryAddExpr()
      : SCEVNAryExpr()
  {}

  template<typename... Args>
  explicit SCEVNAryAddExpr(Args &&... operands)
      : SCEVNAryExpr(std::forward<Args>(operands)...)
  {}

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < Operands_.size(); ++i)
    {
      oss << Operands_.at(i)->DebugString();
      if (i < Operands_.size() - 1)
        oss << " + ";
    }
    oss << ")";
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    auto copy = std::make_unique<SCEVNAryAddExpr>();
    for (const auto & op : Operands_)
    {
      copy->AddOperand(op->Clone());
    }
    return copy;
  }

  template<typename... Args>
  static std::unique_ptr<SCEVNAryAddExpr>
  Create(Args &&... operands)
  {
    return std::make_unique<SCEVNAryAddExpr>(std::forward<Args>(operands)...);
  }
};

class SCEVNAryMulExpr final : public SCEVNAryExpr
{
  friend class ScalarEvolution;

public:
  explicit SCEVNAryMulExpr()
      : SCEVNAryExpr()
  {}

  template<typename... Args>
  explicit SCEVNAryMulExpr(Args &&... operands)
      : SCEVNAryExpr(std::forward<Args>(operands)...)
  {}

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < Operands_.size(); ++i)
    {
      oss << Operands_.at(i)->DebugString();
      if (i < Operands_.size() - 1)
        oss << " * ";
    }
    oss << ")";
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    auto copy = std::make_unique<SCEVNAryMulExpr>();
    for (const auto & op : Operands_)
    {
      copy->AddOperand(op->Clone());
    }
    return copy;
  }

  template<typename... Args>
  static std::unique_ptr<SCEVNAryMulExpr>
  Create(Args &&... operands)
  {
    return std::make_unique<SCEVNAryMulExpr>(std::forward<Args>(operands)...);
  }
};

class ScalarEvolution final : public jlm::rvsdg::Transformation
{
  class Context;
  class Statistics;

public:
  enum class DependencyOp
  {
    Add,
    Mul,
    None,
  };

  struct DependencyInfo
  {
    // Helper struct to keep track of dependencies between loop variables.

    int count; // How many times the dependency occurs. A variable can be dependent on other
               // variables (or itself) multiple times.
    DependencyOp operation; // The operation of the dependency (Add, Mul or None)

    explicit DependencyInfo(const int c = 0, const DependencyOp op = DependencyOp::None)
        : count(c),
          operation(op)
    {}
  };

  typedef std::unordered_map<rvsdg::Output *, DependencyInfo> DependencyMap;

  typedef std::unordered_map<rvsdg::Output *, DependencyMap> DependencyGraph;

  ~ScalarEvolution() noexcept override;

  ScalarEvolution();

  ScalarEvolution(const ScalarEvolution &) = delete;

  ScalarEvolution(ScalarEvolution &&) = delete;

  ScalarEvolution &
  operator=(const ScalarEvolution &) = delete;

  ScalarEvolution &
  operator=(ScalarEvolution &&) = delete;

  /**
   * Returns a copy of the chrec map containing the computed chain recurrences after running
   * the analysis.
   */
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>>
  GetChrecMap() const;

  /**
   * Returns a copy of the scev map containing the computed scev trees after running
   * the analysis.
   */
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>>
  GetSCEVMap() const;

  std::unordered_map<const rvsdg::ThetaNode *, size_t>
  GetTripCountMap() const noexcept;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  std::optional<size_t>
  GetPredictedTripCount(rvsdg::ThetaNode & thetaNode);

  void
  AnalyzeRegion(rvsdg::Region & region);

  /**
   * Goes through all chain recurrences stored in the context (across different loops), and
   * stitches them together wherever possible.
   */
  void
  CombineChrecsAcrossLoops();

  static bool
  StructurallyEqual(const SCEV & a, const SCEV & b);

private:
  /**
   * Tries to compute the number of times the backedge is taken in a loop where the condition for
   * the predicate checks whether the variable described by chrec satisfies the comparison operation
   * when compared to the bound.
   *
   * Figuring out the number of times the backedge is taken can be equivalently expressed as:
   * "At which iteration does the loop condition become false?"
   * That’s naturally expressed as:
   * f(i) - k changes sign, where f(i) is the value of the recurrence at iteration i and k is the
   * bound value
   *
   * @param chrec The chain recurrence to compare with the bound.
   * @param bound The compare value for the operation.
   * @param comparisonOperation The comparison operation that is used in the predicate of the loop.
   * @return The backedge taken count, or std::nullopt if the count cannot be computed or the loop
   * does not terminate.
   */
  static std::optional<size_t>
  ComputeBackedgeTakenCountForChrec(
      const SCEVChainRecurrence & chrec,
      int64_t bound,
      const rvsdg::SimpleOperation * comparisonOperation);

  /**
   * \brief Tries to find a solution to the quadratic equation a^2 x + b x + c = 0 using integer
   * arithmetic.
   *
   * Credit to the SolveQuadraticEquationWrap() method in the LLVM APInt.cpp file
   * (https://llvm.org/doxygen/APInt_8cpp_source.html#l02823) for the general approach.
   *
   * @return A solution to the equation which is greater than zero, or std::nullopt if none can be
   * found
   */
  static std::optional<size_t>
  SolveQuadraticEquation(int64_t a, int64_t b, int64_t c);

  static bool
  IsStepNegative(const SCEV & stepSCEV);

  static bool
  IsStepPositive(const SCEV & stepSCEV);

  static bool
  IsStepZero(const SCEV & stepSCEV);

  static std::unique_ptr<SCEV>
  GetNegativeSCEV(const SCEV & scev);

  std::unique_ptr<SCEV>
  GetOrCreateSCEVForOutput(rvsdg::Output & output);

  DependencyGraph
  CreateDependencyGraph(const rvsdg::ThetaNode & thetaNode) const;

  static void
  FindDependenciesForSCEV(const SCEV & scev, DependencyMap & dependencies, DependencyOp op);

  static std::vector<rvsdg::Output *>
  TopologicalSort(DependencyGraph & dependencyGraph);

  void
  PerformSCEVAnalysis(rvsdg::ThetaNode & thetaNode);

  std::unique_ptr<SCEV>
  ComputeSCEVForGepInnerOffset(
      const rvsdg::SimpleNode & gepNode,
      size_t inputIndex,
      const rvsdg::Type & type);

  /**
   * Recursively traverses chain recurrences to find Init nodes that can be replaced with their (now
   * computed) corresponding chain recurrences and replaces them
   *
   * @param scev The SCEV expression to be traversed
   * @param output
   * @return The resulting recurrence, or std::nullopt if no change was made
   */
  std::optional<std::unique_ptr<SCEV>>
  TryReplaceInitForSCEV(const SCEV & scev, rvsdg::Output & output);

  std::unique_ptr<SCEVChainRecurrence>
  GetOrCreateChainRecurrence(
      rvsdg::Output & output,
      const SCEV & scev,
      rvsdg::ThetaNode & thetaNode);

  std::unique_ptr<SCEVChainRecurrence>
  GetOrCreateStepForSCEV(
      rvsdg::Output & output,
      const SCEV & scevTree,
      rvsdg::ThetaNode & thetaNode);

  /**
   * \brief Apply folding rules for addition to combine two SCEV operands into one.
   * @param lhsOperand The left-hand side operand of the add operation
   * @param rhsOperand The right-hand side operand of the add operation
   * @param output
   * @param output
   * @return A unique ptr to the new operand
   */
  static std::unique_ptr<SCEV>
  ApplyAddFolding(SCEV * lhsOperand, SCEV * rhsOperand, rvsdg::Output & output);

  static std::unique_ptr<SCEVChainRecurrence>
  ComputeProductOfChrecs(
      SCEVChainRecurrence * lhsChrec,
      SCEVChainRecurrence * rhsChrec,
      rvsdg::Output & output);

  /**
   * \brief Apply folding rules for multiplication to combine two SCEV operands into one.
   * @param lhsOperand The left-hand side operand of the mul operation
   * @param rhsOperand The right-hand side operand of the mul operation
   * @param output
   * @return A unique ptr to the new operand
   */
  static std::unique_ptr<SCEV>
  ApplyMulFolding(SCEV * lhsOperand, SCEV * rhsOperand, rvsdg::Output & output);

  /**
   * \brief Try to combine the constants in an n-ary expression (Add or Mul) into themselves.
   * @param expression The expression to be folded
   * @param output
   * @return The unique ptr to the expression
   */
  static std::unique_ptr<SCEV>
  FoldNAryExpression(SCEVNAryExpr & expression, rvsdg::Output & output);

  /**
   * Checks the dependencies of the input variable to determine if we can create a chain recurrence
   * using it's SCEV.
   *
   * The requirements are:
   * - No more than one self-dependency (indicates a self-dependent variable)
   * - No cyclic dependencies (A depends on B and B depends on A)
   * - No dependencies via multiplication (results in a geometric update sequence, which we treat as
   * an invalid induction variable)
   *
   * @param output The output to check.
   * @param dependencyGraph The dependency graph which stores the dependencies of all outputs in the
   * loop.
   * @return True if the requirements are fulfilled, false otherwise.
   */
  static bool
  CanCreateChainRecurrence(rvsdg::Output & output, DependencyGraph & dependencyGraph);

  static bool
  HasCycleThroughOthers(
      rvsdg::Output & currentOutput,
      const rvsdg::Output & originalOutput,
      DependencyGraph & dependencyGraph,
      std::unordered_set<const rvsdg::Output *> & visited,
      std::unordered_set<const rvsdg::Output *> & recursionStack);

  std::unique_ptr<Context> Context_;
};

}

#endif
