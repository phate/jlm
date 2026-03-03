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
};

}

#endif
