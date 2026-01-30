/*
 * Copyright 2025 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_SCALAR_EVOLUTION_HPP
#define JLM_LLVM_OPT_SCALAR_EVOLUTION_HPP

#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <string>

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
  explicit SCEVInit(const rvsdg::Output & pre)
      : PrePointer_{ &pre }
  {}

  const rvsdg::Output *
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
    return std::make_unique<SCEVInit>(*PrePointer_);
  }

  static std::unique_ptr<SCEVInit>
  Create(const rvsdg::Output & prePointer)
  {
    return std::make_unique<SCEVInit>(prePointer);
  }

private:
  const rvsdg::Output * PrePointer_;
};

class SCEVPlaceholder final : public SCEV
{
public:
  explicit SCEVPlaceholder(const rvsdg::Output & pre)
      : PrePointer_{ &pre }
  {}

  const rvsdg::Output *
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
  Create(const rvsdg::Output & PrePointer_)
  {
    return std::make_unique<SCEVPlaceholder>(PrePointer_);
  }

private:
  const rvsdg::Output * PrePointer_;
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
  SwapOperand(const size_t index, const std::unique_ptr<SCEV> & operand)
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

  std::vector<const SCEV *>
  GetOperands() const
  {
    std::vector<const SCEV *> operands{};
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

protected:
  std::vector<std::unique_ptr<SCEV>> Operands_;
};

class SCEVChainRecurrence final : public SCEVNAryExpr
{
public:
  explicit SCEVChainRecurrence(const rvsdg::ThetaNode & theta)
      : SCEVNAryExpr(),
        Loop_{ &theta }
  {}

  const rvsdg::ThetaNode *
  GetLoop() const
  {
    return Loop_;
  }

  SCEV *
  GetStartValue() const
  {
    return Operands_[0].get();
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
    auto copy = std::make_unique<SCEVChainRecurrence>(*Loop_);
    for (const auto & op : Operands_)
    {
      copy->AddOperand(op->Clone());
    }
    return copy;
  }

  static std::unique_ptr<SCEVChainRecurrence>
  Create(const rvsdg::ThetaNode & loop)
  {
    return std::make_unique<SCEVChainRecurrence>(loop);
  }

protected:
  const rvsdg::ThetaNode * Loop_;
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
  class Statistics;

  class Context final
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
    InsertChrec(const rvsdg::Output & output, const std::unique_ptr<SCEVChainRecurrence> & chrec);

    std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>>
    GetChrecs() const;

    int
    GetNumOfChrecsWithOrder(const int n) const;

    size_t
    GetNumTotalChrecs() const;

    void
    AddLoopVar(const rvsdg::Output & var);

    void
    InsertSCEV(const rvsdg::Output & output, const std::unique_ptr<SCEV> & scev);

    std::unique_ptr<SCEVChainRecurrence>
    TryGetChrecForOutput(const rvsdg::Output & output) const;

    std::unique_ptr<SCEV>
    TryGetSCEVForOutput(const rvsdg::Output & output) const;

    size_t
    GetNumTotalLoopVars() const;

    static std::unique_ptr<Context>
    Create()
    {
      return std::make_unique<Context>();
    }

  private:
    std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> ChrecMap_;
    std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> SCEVMap_;
    std::vector<const rvsdg::Output *> LoopVars_;
  };

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

  typedef std::unordered_map<const rvsdg::Output *, DependencyInfo> DependencyMap;

  typedef std::unordered_map<const rvsdg::Output *, DependencyMap> IVDependencyGraph;

  ~ScalarEvolution() noexcept override;

  ScalarEvolution();

  ScalarEvolution(const ScalarEvolution &) = delete;

  ScalarEvolution(ScalarEvolution &&) = delete;

  ScalarEvolution &
  operator=(const ScalarEvolution &) = delete;

  ScalarEvolution &
  operator=(ScalarEvolution &&) = delete;

  static std::unique_ptr<Context>
  CreateContext()
  {
    return Context::Create();
  }

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  AnalyzeRegion(const rvsdg::Region & region, Context & ctx);

  static void
  CombineChrecsAcrossLoops(Context & ctx);

  static bool
  StructurallyEqual(const SCEV & a, const SCEV & b);

private:
  static std::unique_ptr<SCEV>
  GetNegativeSCEV(const SCEV & scev);

  static std::unique_ptr<SCEV>
  GetOrCreateSCEVForOutput(const rvsdg::Output & output, Context & ctx);

  static IVDependencyGraph
  CreateDependencyGraph(
      const std::vector<rvsdg::ThetaNode::LoopVar> & loopVars,
      const Context & ctx);

  static void
  FindDependenciesForSCEV(const SCEV & scev, DependencyMap & dependencies, DependencyOp op);

  static std::vector<const rvsdg::Output *>
  TopologicalSort(const IVDependencyGraph & dependencyGraph);

  static void
  PerformSCEVAnalysis(const rvsdg::ThetaNode & thetaNode, Context & ctx);

  static std::optional<std::unique_ptr<SCEV>>
  TryReplaceInitForSCEV(const SCEV & scev, Context & ctx);

  static std::unique_ptr<SCEVChainRecurrence>
  GetOrCreateChainRecurrence(
      const rvsdg::Output & output,
      const SCEV & scev,
      const rvsdg::ThetaNode & thetaNode,
      Context & ctx);

  static std::unique_ptr<SCEVChainRecurrence>
  GetOrCreateStepForSCEV(
      const rvsdg::Output & output,
      const SCEV & scevTree,
      const rvsdg::ThetaNode & thetaNode,
      Context & ctx);

  static std::unique_ptr<SCEV>
  ApplyAddFolding(const SCEV * lhsOperand, const SCEV * rhsOperand);

  static std::unique_ptr<SCEV>
  ApplyMulFolding(const SCEV * lhsOperand, const SCEV * rhsOperand);

  static std::unique_ptr<SCEV>
  FoldNAryExpression(SCEVNAryExpr & expression);

  static bool
  IsValidInductionVariable(const rvsdg::Output & variable, IVDependencyGraph & dependencyGraph);

  /**
   * Checks the operands of the given \p chrec to see if any of them are unknown.
   *
   * @param chrec the chain recurrence to be checked
   * @return true if the recurrence contains an unknown, false otherwise
   */
  static bool
  IsUnknown(const SCEVChainRecurrence & chrec);

  static bool
  HasCycleThroughOthers(
      const rvsdg::Output & currentIV,
      const rvsdg::Output & originalIV,
      IVDependencyGraph & dependencyGraph,
      std::unordered_set<const rvsdg::Output *> & visited,
      std::unordered_set<const rvsdg::Output *> & recursionStack);
};

}

#endif
