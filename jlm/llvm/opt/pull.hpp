/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PULL_HPP
#define JLM_LLVM_OPT_PULL_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Region;
}

namespace jlm::llvm
{

/**
 * \brief Node Sinking Optimization
 */
class NodeSinking final : public rvsdg::Transformation
{
public:
  class Statistics;

  ~NodeSinking() noexcept override;

  NodeSinking()
      : Transformation("NodeSinking")
  {}

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  /**
   * Sink all nodes that are dependent on \p gammaNode and are from the same region as \p gammaNode
   * into the gamma node's subregions.
   *
   * @param gammaNode A \ref rvsdg::GammaNode
   * @return The number of dependent nodes that were sunk into the gamma node.
   */
  static size_t
  sinkDependentNodesIntoGamma(rvsdg::GammaNode & gammaNode);

private:
  /**
   * Collects all nodes that (in-)directly dependent on \p node.
   *
   * @param node A node for which to compute the dependent nodes.
   * @return A set of all the dependent nodes.
   */
  static util::HashSet<rvsdg::Node *>
  collectDependentNodes(const rvsdg::Node & node);

  /**
   * Sort \p nodes by their depth.
   *
   * @param nodes A set of nodes.
   * @return A sorted vector of nodes.
   *
   * \pre All nodes in \p must be from the same region.
   */
  static std::vector<rvsdg::Node *>
  sortByDepth(const util::HashSet<rvsdg::Node *> & nodes);
};

void
pullin_top(rvsdg::GammaNode * gamma);

void
pull(rvsdg::GammaNode * gamma);

void
pull(rvsdg::Region * region);

}

#endif
