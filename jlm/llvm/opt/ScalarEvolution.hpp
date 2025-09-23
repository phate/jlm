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
  typedef util::HashSet<const rvsdg::ThetaNode::LoopVar *> InductionVariableSet;

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

  void
  TraverseGraph(const rvsdg::Graph & rvsdg);

  void
  TraverseRegion(const rvsdg::Region * region);

  static InductionVariableSet
  FindInductionVariables(const rvsdg::ThetaNode * thetaNode);

  static bool
  IsBasedOnInductionVariable(const rvsdg::Output * output, const rvsdg::ThetaNode * thetaNode);

private:
  std::unordered_map<const rvsdg::ThetaNode *, InductionVariableSet> InductionVariableMap_;
};

}

#endif
