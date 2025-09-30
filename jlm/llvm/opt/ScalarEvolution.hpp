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
class ScalarEvolution final : public jlm::rvsdg::Transformation
{
  class Statistics;

public:
  typedef util::HashSet<const rvsdg::Output *>
      InductionVariableSet; // Stores the pointers to the output result from the subregion for the
                            // induction variables

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

private:
  std::unordered_map<const rvsdg::ThetaNode *, InductionVariableSet> InductionVariableMap_;

  void
  TraverseRegion(const rvsdg::Region & region);

  static bool
  IsBasedOnInductionVariable(const rvsdg::Output & output, InductionVariableSet & candidates);
};

}

#endif
