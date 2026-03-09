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

  static bool
  ContainsMul(const SCEV & scev);

  static bool
  IsLinearMul(const SCEV & scev);

  static bool
  IsValidCandidateOperation(const SCEV & scevTree);

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
      std::vector<rvsdg::Output *> & candidateOperations,
      std::unordered_set<rvsdg::Output *> & visited);

  void
  ReplaceCandidateOperation(rvsdg::Output & output, rvsdg::ThetaNode & thetaNode);

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> ChrecMap_;
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> SCEVMap_;
};

}

#endif
