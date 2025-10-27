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

  static void
  correlatePredicatesInTheta(rvsdg::ThetaNode & thetaNode);

  static std::optional<std::vector<size_t>>
  extractControlConstantAlternatives(const rvsdg::Output & gammaOutput);
};

}

#endif
