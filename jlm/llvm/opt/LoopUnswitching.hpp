/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_LOOPUNSWITCHING_HPP
#define JLM_LLVM_OPT_LOOPUNSWITCHING_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Node;
class Region;
class SubstitutionMap;
class ThetaNode;
}

namespace jlm::llvm
{

class ThetaGammaPredicateCorrelation;

/**
 * \brief LoopUnswitching
 */
class LoopUnswitching final : public rvsdg::Transformation
{
public:
  class Statistics;

  ~LoopUnswitching() noexcept override;

  LoopUnswitching()
      : Transformation("LoopUnswitching")
  {}

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector);

private:
  static void
  HandleRegion(rvsdg::Region & region);

  static bool
  UnswitchLoop(rvsdg::ThetaNode & thetaNode);

  static rvsdg::GammaNode *
  IsUnswitchable(const rvsdg::ThetaNode & thetaNode);

  static void
  SinkNodesIntoGamma(rvsdg::GammaNode & gammaNode, const rvsdg::ThetaNode & thetaNode);

  static std::vector<std::vector<rvsdg::Node *>>
  CollectPredicateNodes(const rvsdg::ThetaNode & thetaNode, const rvsdg::GammaNode & gammaNode);

  static void
  CopyPredicateNodes(
      rvsdg::Region & target,
      rvsdg::SubstitutionMap & substitutionMap,
      const std::vector<std::vector<rvsdg::Node *>> & nodes);

  /**
   * Checks that the post-values from all loop variables are originating from \p gammaNode.
   *
   * @param thetaNode The theta node for which to perform the check.
   * @param gammaNode The gamma node through which all loop variables need to be routed.
   * @return True, if all post-values originate from \p gammaNode, otherwise false.
   */
  static bool
  allLoopVarsAreRoutedThroughGamma(
      const rvsdg::ThetaNode & thetaNode,
      const rvsdg::GammaNode & gammaNode);

  static rvsdg::SubstitutionMap
  handleGammaExitRegion(
      const ThetaGammaPredicateCorrelation & correlation,
      rvsdg::GammaNode & newGammaNode,
      const rvsdg::SubstitutionMap & substitutionMap);

  static rvsdg::SubstitutionMap
  handleGammaRepetitionRegion(
      const ThetaGammaPredicateCorrelation & correlation,
      rvsdg::GammaNode & newGammaNode,
      const std::vector<std::vector<rvsdg::Node *>> & predicateNodes,
      const rvsdg::SubstitutionMap & substitutionMap);
};

}

#endif
