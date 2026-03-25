/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_LOOP_STRENGTH_REDUCTION_HPP
#define JLM_LLVM_OPT_LOOP_STRENGTH_REDUCTION_HPP

#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
class LoopStrengthReduction final : public jlm::rvsdg::Transformation
{
  class Statistics;
  class Context;

public:
  ~LoopStrengthReduction() noexcept override;

  LoopStrengthReduction();

  LoopStrengthReduction(const LoopStrengthReduction &) = delete;

  LoopStrengthReduction(LoopStrengthReduction &&) = delete;

  LoopStrengthReduction &
  operator=(const LoopStrengthReduction &) = delete;

  LoopStrengthReduction &
  operator=(LoopStrengthReduction &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  ProcessRegion(rvsdg::Region & region);

  void
  ReduceStrength(rvsdg::ThetaNode & thetaNode);

  /**
   * Checks if the RVSDG subtree of the output contains an IntegerMulExpr somewhere in the tree.
   *
   * @param output The output to be checked
   * @return true if the subtree contains a multiplication operation, otherwise false.
   */
  static bool
  ContainsMul(const rvsdg::Output & output);

  bool
  IsValidCandidateOperation(const rvsdg::Output & output) const;

  /**
   * Checks if the SCEV tree is a valid linear combination of placeholders and constants,
   * i.e. sums of terms where each term is either: a constant, a placeholder (loop variable), a
   * linear multiplication (constant * placeholder)
   *
   * @param scev The SCEV tree to be checked
   * @return True if the scev tree is a linear combination, otherwise false.
   */
  static bool
  IsLinearCombination(const SCEV & scev);

  void
  ProcessOutput(
      rvsdg::Output & output,
      rvsdg::ThetaNode & thetaNode,
      util::HashSet<rvsdg::Output *> & candidateOperations,
      util::HashSet<rvsdg::Output *> & visited);

  void
  ReplaceCandidateOperation(rvsdg::Output & output, rvsdg::ThetaNode & thetaNode);

  void
  ReplaceGEPOperation(
      std::unique_ptr<SCEVChainRecurrence> & chrec,
      rvsdg::Output & output,
      rvsdg::ThetaNode & thetaNode,
      const std::shared_ptr<const PointerType> & pointerType);

  void
  ReplaceArithmeticOperation(
      std::unique_ptr<SCEVChainRecurrence> & chrec,
      rvsdg::Output & output,
      rvsdg::ThetaNode & thetaNode,
      const std::shared_ptr<const rvsdg::BitType> & bitType);

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> ChrecMap_;
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> SCEVMap_;

  std::unique_ptr<Context> Context_;
};

}

#endif
