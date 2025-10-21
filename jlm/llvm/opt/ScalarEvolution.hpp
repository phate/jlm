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

  const std::unique_ptr<SCEV>
  GetLeftOperand() const
  {
    return LeftOperand_->Clone();
  }

  const std::unique_ptr<SCEV>
  GetRightOperand() const
  {
    return RightOperand_->Clone();
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
    oss << "{" << leftStr << ",+," << rightStr << "}";
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

class SCEVChrecExpr final : public SCEV
{
  friend class ScalarEvolution;

public:
  explicit SCEVChrecExpr(const rvsdg::ThetaNode * theta)
      : Operands_{},
        Loop_{ theta }
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

  std::string
  DebugString() const override
  {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < Operands_.size(); ++i)
    {
      oss << Operands_.at(i)->DebugString() << "+";
    }
    oss << "}" << '\n';
    return oss.str();
  }

  std::unique_ptr<SCEV>
  Clone() const override
  {
    auto copy = std::make_unique<SCEVChrecExpr>(Loop_);
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

  static InductionVariableSet
  FindInductionVariables(const rvsdg::ThetaNode & thetaNode);

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>>
  CreateChainRecurrences(
      const InductionVariableSet & inductionVariableCandidates,
      const rvsdg::ThetaNode & thetaNode);

  static bool
  StructurallyEqual(const SCEV & a, const SCEV & b);

private:
  std::unordered_map<const rvsdg::ThetaNode *, InductionVariableSet> InductionVariableMap_;
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> UniqueSCEVs_;

  void
  TraverseRegion(const rvsdg::Region & region);

  static bool
  IsBasedOnInductionVariable(const rvsdg::Output & output, InductionVariableSet & candidates);

  std::unique_ptr<SCEV>
  GetOrCreateSCEVForOutput(const rvsdg::Output & output);

  std::optional<const SCEV *>
  TryGetSCEVForOutput(const rvsdg::Output & output);

  IVDependencyGraph
  CreateDependencyGraph(
      const InductionVariableSet & inductionVariables,
      const rvsdg::ThetaNode & thetaNode) const;

  static std::unordered_map<const rvsdg::Output *, int>
  FindDependenciesForSCEV(const SCEV & currentSCEV, const rvsdg::Output & currentIV);

  static std::vector<const rvsdg::Output *>
  TopologicalSort(const IVDependencyGraph & dependencyGraph);

  std::unique_ptr<SCEV>
  ReplacePlaceholders(
      const SCEV & scevTree,
      const rvsdg::Output & currentIV,
      const rvsdg::ThetaNode & thetaNode,
      const InductionVariableSet & validIVs);

  static bool
  IsValidInductionVariable(const rvsdg::Output & variable, IVDependencyGraph & dependencyGraph);

  static bool
  HasCycleThroughOthers(
      const rvsdg::Output * current,
      IVDependencyGraph & dependencyGraph,
      std::unordered_set<const rvsdg::Output *> & visited,
      std::unordered_set<const rvsdg::Output *> & recursionStack);
};

}

#endif
