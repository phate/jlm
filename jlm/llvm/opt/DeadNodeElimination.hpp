/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_DEADNODEELIMINATION_HPP
#define JLM_LLVM_OPT_DEADNODEELIMINATION_HPP

#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/Transformation.hpp>

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

/** \brief Dead Node Elimination Optimization
 *
 * Dead Node Elimination removes all nodes that do not contribute to the result of a computation. A
 * node is considered dead if all its outputs are dead, and an output is considered dead if it has
 * no users or all its users are already dead. An input (and therefore an outputs' user) is
 * considered dead if the corresponding node is dead. We call all nodes, inputs, and outputs that
 * are not dead alive.
 *
 * The Dead Node Elimination optimization consists of two phases: mark and sweep. The mark phase
 * traverses the RVSDG and marks all nodes, inputs, and outputs that it finds as alive, while the
 * sweep phase removes then all nodes, inputs, and outputs that were not discovered by the mark
 * phase, i.e., all dead nodes, inputs, and outputs.
 *
 * Please see TestDeadNodeElimination.cpp for Dead Node Elimination examples.
 */
class DeadNodeElimination final : public rvsdg::Transformation
{
  class Context;
  class Statistics;

public:
  ~DeadNodeElimination() noexcept override;

  DeadNodeElimination();

  DeadNodeElimination(const DeadNodeElimination &) = delete;

  DeadNodeElimination(DeadNodeElimination &&) = delete;

  DeadNodeElimination &
  operator=(const DeadNodeElimination &) = delete;

  DeadNodeElimination &
  operator=(DeadNodeElimination &&) = delete;

  void
  run(rvsdg::Region & region);

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  markRegion(const rvsdg::Region & region);

  void
  markOutput(const rvsdg::Output & output);

  void
  sweepRvsdg(rvsdg::Graph & rvsdg) const;

  void
  sweepRegion(rvsdg::Region & region) const;

  void
  sweepStructuralNode(rvsdg::StructuralNode & node) const;

  void
  sweepGamma(rvsdg::GammaNode & gammaNode) const;

  void
  sweepTheta(rvsdg::ThetaNode & thetaNode) const;

  void
  sweepLambda(rvsdg::LambdaNode & lambdaNode) const;

  void
  sweepPhi(rvsdg::PhiNode & phiNode) const;

  static void
  sweepDelta(rvsdg::DeltaNode & deltaNode);

  std::unique_ptr<Context> Context_{};
};

}

#endif
