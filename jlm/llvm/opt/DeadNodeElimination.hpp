/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_DEADNODEELIMINATION_HPP
#define JLM_LLVM_OPT_DEADNODEELIMINATION_HPP

#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Graph;
class output;
class StructuralNode;
class ThetaNode;
class Region;
}

namespace jlm::llvm
{

namespace delta
{
class node;
}

class LambdaNode;

namespace phi
{
class node;
}

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
  MarkRegion(const rvsdg::Region & region);

  void
  MarkOutput(const jlm::rvsdg::output & output);

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
  SweepPhi(phi::node & phiNode) const;

  static void
  SweepDelta(delta::node & deltaNode);

  std::unique_ptr<Context> Context_;
};

}

#endif
