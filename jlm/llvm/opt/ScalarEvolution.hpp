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
  TraverseRegion(rvsdg::Region * region);

  void
  FindInductionVariables(const rvsdg::ThetaNode * thetaNode);

private:
  /**
   * \brief Method used to check if the input to a node is a loop variable.
   *
   * \returns
   *   An std::optional with the loop variable, if found.
   */
  static std::optional<rvsdg::ThetaNode::LoopVar>
  TryGetLoopVarFromInput(
      const rvsdg::Input * input,
      std::vector<rvsdg::ThetaNode::LoopVar> loopVars)
  {
    for (auto & loopVar : loopVars)
    {
      if (input->origin() == loopVar.pre)
      {
        return loopVar;
      }
    }
    return std::nullopt;
  }

  typedef util::HashSet<rvsdg::Output *>
      InductionVariableSet; // This set holds the ".pre"-pointer for the induction variables in a
                            // theta node

  std::unordered_map<const rvsdg::ThetaNode *, InductionVariableSet> InductionVariableMap_;
};

}

#endif
