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

private:
  const rvsdg::Output * PrePointer_;
};

class SCEVConstant final : public SCEV
{
public:
  explicit SCEVConstant(const uint64_t value)
      : Value_{ value }
  {}

  uint64_t
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

private:
  uint64_t Value_;
};

class SCEVAddExpr final : public SCEV
{
public:
  SCEVAddExpr()
      : LeftOperand_{},
        RightOperand_{}
  {}

  SCEVAddExpr(std::unique_ptr<SCEV> left, std::unique_ptr<SCEV> right)
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

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    std::string leftStr = LeftOperand_ ? LeftOperand_->DebugString() : "null";
    std::string rightStr = RightOperand_ ? RightOperand_->DebugString() : "null";
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

private:
  std::unique_ptr<SCEV> LeftOperand_;
  std::unique_ptr<SCEV> RightOperand_;
};

class SCEVChainRecurrence final : public SCEV
{
  friend class ScalarEvolution;

public:
  explicit SCEVChainRecurrence(const rvsdg::ThetaNode & theta)
      : Operands_{},
        Loop_{ &theta }
  {}

  void
  AddOperand(std::unique_ptr<SCEV> scev)
  {
    Operands_.push_back(std::move(scev));
  }

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
  SetStartValue(const std::unique_ptr<SCEV> & initScev)
  {
    Operands_.insert(Operands_.begin(), initScev->Clone());
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
    oss << "}" << "<" << Loop_->DebugString() << ">";
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

protected:
  std::vector<std::unique_ptr<SCEV>> Operands_;
  const rvsdg::ThetaNode * Loop_;
};

class SCEVNAryAddExpr final : public SCEV
{
  friend class ScalarEvolution;

public:
  explicit SCEVNAryAddExpr()
      : Operands_{}
  {}

  template<typename... Args>
  explicit SCEVNAryAddExpr(Args &&... operands)
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

protected:
  std::vector<std::unique_ptr<SCEV>> Operands_;
};

class ScalarEvolution final : public jlm::rvsdg::Transformation
{
  class Statistics;

public:
  typedef util::HashSet<const rvsdg::Output *>
      InductionVariableSet; // Stores the pointers to the output result from the subregion for the
                            // induction variables

  typedef std::unordered_map<const rvsdg::Output *, std::unordered_map<const rvsdg::Output *, int>>
      IVDependencyGraph;

  ~ScalarEvolution() noexcept override;

  ScalarEvolution()
      : Transformation("ScalarEvolution")
  {}

  ScalarEvolution(const ScalarEvolution &) = delete;

  ScalarEvolution(ScalarEvolution &&) = delete;

  ScalarEvolution &
  operator=(const ScalarEvolution &) = delete;

  ScalarEvolution &
  operator=(ScalarEvolution &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>>
  PerformSCEVAnalysis(const rvsdg::ThetaNode & thetaNode);

  static bool
  StructurallyEqual(const SCEV & a, const SCEV & b);

private:
  std::unordered_map<const rvsdg::ThetaNode *, InductionVariableSet> InductionVariableMap_;
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> UniqueSCEVs_;
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>>
      ChainRecurrenceMap_;

  void
  TraverseRegion(const rvsdg::Region & region);

  std::unique_ptr<SCEV>
  GetOrCreateSCEVForOutput(const rvsdg::Output & output);

  std::optional<const SCEV *>
  TryGetSCEVForOutput(const rvsdg::Output & output);

  IVDependencyGraph
  CreateDependencyGraph(const rvsdg::ThetaNode & thetaNode) const;

  static std::unordered_map<const rvsdg::Output *, int>
  FindDependenciesForSCEV(const SCEV & currentSCEV, const rvsdg::Output & currentIV);

  static std::vector<const rvsdg::Output *>
  TopologicalSort(const IVDependencyGraph & dependencyGraph);

  std::unique_ptr<SCEVChainRecurrence>
  CreateChainRecurrence(
      const rvsdg::Output & IV,
      const SCEV & scevTree,
      const rvsdg::ThetaNode & thetaNode);

  static std::unique_ptr<SCEV>
  ApplyFolding(SCEV * lhsOperand, SCEV * rhsOperand);

  static bool
  IsValidInductionVariable(const rvsdg::Output & variable, IVDependencyGraph & dependencyGraph);

  static bool
  HasCycleThroughOthers(
      const rvsdg::Output & currentIV,
      IVDependencyGraph & dependencyGraph,
      std::unordered_set<const rvsdg::Output *> & visited,
      std::unordered_set<const rvsdg::Output *> & recursionStack);
};

}

#endif
