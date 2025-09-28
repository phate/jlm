/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PartialRedundancyElimination_HPP
#define JLM_LLVM_OPT_PartialRedundancyElimination_HPP

#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <stdint.h>
#include <jlm/rvsdg/node.hpp>

namespace jlm::rvsdg
{
class DeltaNode;
class GammaNode;
class Graph;
class LambdaNode;
class Output;
class StructuralNode;
class ThetaNode;
class Region;
}

namespace jlm::llvm
{

class StrictHasher
{

  /** \brief Partial Redundancy Elimination
   *
   * This hasher assumes all function arguments are different.
   *
   *
   */
public:
  void populate_table(rvsdg::Region& reg);
private:
  std::unordered_map<jlm::rvsdg::Node*, uint64_t> hashes;

};

/** \brief Partial Redundancy Elimination
 *
 * Todo: description here
 *
 *
 */
class PartialRedundancyElimination final : public rvsdg::Transformation
{
  class Context;
  class Statistics;

public:
  ~PartialRedundancyElimination() noexcept override;

  PartialRedundancyElimination();

  PartialRedundancyElimination(const PartialRedundancyElimination &) = delete;
  PartialRedundancyElimination(PartialRedundancyElimination &&) = delete;

  PartialRedundancyElimination &
  operator=(const PartialRedundancyElimination &) = delete;
  PartialRedundancyElimination &
  operator=(PartialRedundancyElimination &&) = delete;

  /*void
  run(rvsdg::Region & region);*/

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
 /* void
  MarkRegion(const rvsdg::Region & region);

  void
  MarkOutput(const jlm::rvsdg::Output & output);

  void
  SweepRvsdg(rvsdg::Graph & rvsdg) const;

  void
  SweepRegion(rvsdg::Region & region) const;

  void
  SweepStructuralNode(rvsdg::StructuralNode & node) const;

  void
  SweepGamma(rvsdg::GammaNode & gammaNode) const;

  void
  SweepTheta(rvsdg::ThetaNode & thetaNode) const;

  void
  SweepLambda(rvsdg::LambdaNode & lambdaNode) const;

  void
  SweepPhi(rvsdg::PhiNode & phiNode) const;

  static void
  SweepDelta(rvsdg::DeltaNode & deltaNode);
*/
  std::unique_ptr<Context> Context_;
};

}

#endif
