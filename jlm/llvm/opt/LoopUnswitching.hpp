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
 * Determines for a given theta and gamma node, encapsulated in the \ref
 * ThetaGammaPredicateCorrelation, whether loop unswitching should be performed.
 */
class LoopUnswitchingHeuristic
{
public:
  virtual ~LoopUnswitchingHeuristic() noexcept;

  virtual bool
  shouldUnswitchLoop(ThetaGammaPredicateCorrelation & correlation) const noexcept = 0;
};

/**
 * Default heuristic for loop unswitching. The heuristic checks that there are no structural
 * nodes as part of the shared predicate nodes between the theta and gamma node. The reasoning
 * is that the predicate nodes are duplicated, which would mean that we would duplicate conditionals
 * and loops if we would permit structural nodes as part of the predicate nodes.
 */
class LoopUnswitchingDefaultHeuristic final : public LoopUnswitchingHeuristic
{
public:
  ~LoopUnswitchingDefaultHeuristic() noexcept override;

  bool
  shouldUnswitchLoop(ThetaGammaPredicateCorrelation & correlation) const noexcept override;

  static std::shared_ptr<const LoopUnswitchingDefaultHeuristic>
  create();
};

/**
 * \brief LoopUnswitching
 *
 * Loop unswitching transforms a theta node with a gamma node in its subregion to a gamma node that
 * contains a theta node in one of its subregions, avoiding the conditional in every loop iteration.
 *
 * The pass transforms the following graph:
 * -----------------------------------------------------------------
 * | theta                                                         |
 * |                                                               |
 * |     *P*                                                       |
 * |  ----|---                                                     |
 * |  |      |                                                     |
 * |  |   ---|--------------------------------------------------   |
 * |  |   | gamma                   |                          |   |
 * |  |   |                         |                          |   |
 * |  |   |         *X*             |        *R*               |   |
 * |  |   |_________________________|__________________________|   |
 * |  |                                                            |
 * |__|____________________________________________________________|
 *
 * to the following graph:
 *    *P*
 *     |
 * ----|------------------------------------------------------------
 * | gamma                |                                        |
 * |                      |  ------------------------------------- |
 * |                      |  | theta                             | |
 * |                      |  |                                   | |
 * |                      |  |               *R*                 | |
 * |                      |  |                                   | |
 * |                      |  |                                   | |
 * |                      |  |    *P*                            | |
 * |                      |  |     |                             | |
 * |                      |  |_____|_____________________________| |
 * |                      |                                        |
 * |______________________|________________________________________|
 *
 *                              *X*
 *
 * where
 * 1. *P* is the predicate subgraph, i.e., all nodes that are responsible for computing the
 * predicate of the theta and gamma node. The theta and gamma node must have the same predicate for
 * the transformation to occur.
 * 2. *X* is the exit subregion. It denotes the region that is executed once the predicate evaluates
 * to false and the loop is exited.
 * 3. *R* is the repetition subregion. It denotes the region that is executed if the predicate
 * evaluates to true the loop is repeated.
 *
 * The predicate subgraph *P* is duplicated as part of the transformation.
 */
class LoopUnswitching final : public rvsdg::Transformation
{
public:
  class Statistics;

  ~LoopUnswitching() noexcept override;

  explicit LoopUnswitching(std::shared_ptr<const LoopUnswitchingHeuristic> heuristic)
      : Transformation("LoopUnswitching"),
        heuristic_(std::move(heuristic))
  {}

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

  static void
  CreateAndRun(
      rvsdg::RvsdgModule & rvsdgModule,
      util::StatisticsCollector & statisticsCollector,
      std::shared_ptr<const LoopUnswitchingHeuristic> heuristic);

private:
  void
  HandleRegion(rvsdg::Region & region);

  bool
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

  std::shared_ptr<const LoopUnswitchingHeuristic> heuristic_;
};

}

#endif
