/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PREDICATECORRELATION_HPP
#define JLM_LLVM_OPT_PREDICATECORRELATION_HPP

#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/Transformation.hpp>

#include <optional>
#include <vector>

namespace jlm
{

namespace rvsdg
{
class Region;
class ThetaNode;
}

namespace util
{
class StatisticsCollector;
}
}

namespace jlm::llvm
{

/**
 * Predicate Correlation correlates the predicates between theta and gamma nodes, and
 * redirects their respective predicates to match operation nodes.
 *
 * ### Theta-Gamma Predicate Correlation
 * If a theta node's predicate originates from a gamma node with two control flow constants, then
 * the theta node's predicate is redirected to the gamma node's predicate match node.
 */
class PredicateCorrelation final : public rvsdg::Transformation
{
public:
  ~PredicateCorrelation() noexcept override;

  PredicateCorrelation()
      : Transformation("PredicateCorrelation")
  {}

  PredicateCorrelation(const PredicateCorrelation &) = delete;

  PredicateCorrelation &
  operator=(const PredicateCorrelation &) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  correlatePredicatesInRegion(rvsdg::Region & region);

  /**
   * Performs theta-gamma predicate correlation
   *
   * @param thetaNode The theta node for which to correlate.
   */
  static void
  correlatePredicatesInTheta(rvsdg::ThetaNode & thetaNode);

  /**
   * Takes the output of a gamma node and if the output's respective branch results in every
   * subregion originate from a control constant, then it returns a vector of the control constant
   * alternatives.
   *
   * @param gammaOutput The output of a gamma node.
   * @return The control constant alternatives for each of the gamma node's subregion, or
   * std::nullopt;
   */
  static std::optional<std::vector<size_t>>
  extractControlConstantAlternatives(const rvsdg::Output & gammaOutput);
};

}

#endif
