/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSIS_TOPDOWNMEMORYNODEELIMINATOR_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSIS_TOPDOWNMEMORYNODEELIMINATOR_HPP

#include <jlm/llvm/opt/alias-analyses/MemoryNodeEliminator.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
class CallNode;
class CallTypeClassifier;

namespace lambda
{
class node;
}

namespace phi
{
class node;
}
}

namespace jlm::rvsdg
{
class gamma_node;
class node;
class region;
class simple_node;
class structural_node;
class theta_node;
}

namespace jlm::llvm::aa
{

/** \brief Top-down memory node eliminator
 *
 * The key idea of the TopDownMemoryNodeEliminator is to restrict the lifetime of memory states by
 * eliminating the respective memory nodes from regions where the corresponding RVSDG nodes are not
 * alive. For example, the lifetime of a stack allocation from an alloca node is only within the
 * function the alloca node is alive.
 *
 * The AgnosticMemoryNodeProvider and the RegionAwareMemoryNodeProvider are only region-aware, but
 * not lifetime-aware. In other words, they restrict the number of regions a memory state needs to
 * be routed through, but do not limit the lifetime of the respective memory states. The
 * provisioning produced by these memory node providers serves as seed provisioning for the
 * TopDownMemoryNodeEliminator, which restricts then the lifetime of memory locations.
 *
 * The TopDownMemoryNodeEliminator only restricts the lifetime of memory states from alloca nodes
 * before the nodes are alive.
 */
class TopDownMemoryNodeEliminator final : public MemoryNodeEliminator
{
  class Context;
  class Provisioning;
  class Statistics;

public:
  ~TopDownMemoryNodeEliminator() noexcept override;

  TopDownMemoryNodeEliminator();

  TopDownMemoryNodeEliminator(const TopDownMemoryNodeEliminator &) = delete;

  TopDownMemoryNodeEliminator(TopDownMemoryNodeEliminator &&) = delete;

  TopDownMemoryNodeEliminator &
  operator=(const TopDownMemoryNodeEliminator &) = delete;

  TopDownMemoryNodeEliminator &
  operator=(TopDownMemoryNodeEliminator &&) = delete;

  std::unique_ptr<MemoryNodeProvisioning>
  EliminateMemoryNodes(
      const RvsdgModule & rvsdgModule,
      const MemoryNodeProvisioning & seedProvisioning,
      util::StatisticsCollector & statisticsCollector) override;

  /**
   * Creates a TopDownMemoryNodeEliminator and calls the EliminateMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provisioning should be performed.
   * @param seedProvisioning A provisioning from which memory nodes will be eliminated.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return A new instance of MemoryNodeProvisioning.
   */
  static std::unique_ptr<MemoryNodeProvisioning>
  CreateAndEliminate(
      const RvsdgModule & rvsdgModule,
      const MemoryNodeProvisioning & seedProvisioning,
      util::StatisticsCollector & statisticsCollector);

  /**
   * Creates a TopDownMemoryNodeEliminator and calls the EliminateMemoryNodes() method.
   *
   * @param rvsdgModule The RVSDG module on which the provisioning should be performed.
   * @param seedProvisioning A provisioning from which memory nodes will be eliminated.
   *
   * @return A new instance of MemoryNodeProvisioning.
   */
  static std::unique_ptr<MemoryNodeProvisioning>
  CreateAndEliminate(
      const RvsdgModule & rvsdgModule,
      const MemoryNodeProvisioning & seedProvisioning);

private:
  void
  EliminateTopDown(const RvsdgModule & rvsdgModule);

  void
  EliminateTopDownRootRegion(rvsdg::region & region);

  void
  EliminateTopDownRegion(rvsdg::region & region);

  void
  EliminateTopDownStructuralNode(const rvsdg::structural_node & structuralNode);

  void
  EliminateTopDownLambda(const lambda::node & lambdaNode);

  void
  EliminateTopDownLambdaEntry(const lambda::node & lambdaNode);

  void
  EliminateTopDownLambdaExit(const lambda::node & lambdaNode);

  void
  EliminateTopDownPhi(const phi::node & phiNode);

  void
  EliminateTopDownGamma(const rvsdg::gamma_node & gammaNode);

  void
  EliminateTopDownTheta(const rvsdg::theta_node & thetaNode);

  void
  EliminateTopDownSimpleNode(const rvsdg::simple_node & simpleNode);

  void
  EliminateTopDownAlloca(const rvsdg::simple_node & node);

  void
  EliminateTopDownCall(const CallNode & callNode);

  void
  EliminateTopDownNonRecursiveDirectCall(
      const CallNode & callNode,
      const CallTypeClassifier & callTypeClassifier);

  void
  EliminateTopDownRecursiveDirectCall(
      const CallNode & callNode,
      const CallTypeClassifier & callTypeClassifier);

  void
  EliminateTopDownExternalCall(
      const CallNode & callNode,
      const CallTypeClassifier & callTypeClassifier);

  void
  EliminateTopDownIndirectCall(
      const CallNode & indirectCall,
      const CallTypeClassifier & callTypeClassifier);

  /**
   * Collects for every tail-lambda all the memory nodes that would be alive at the beginning of a
   * tail-lambda's execution. A tail-lambda is a lambda that is only dead or exported, i.e., no
   * other node in \p rvsdgModule depends on it.
   *
   * @param rvsdgModule RVSDG module the analysis is performed on.
   *
   * @see graph::ExtractTailNodes()
   */
  void
  InitializeLiveNodesOfTailLambdas(const RvsdgModule & rvsdgModule);

  /**
   * Initializes the memory nodes that are alive at the beginning of every tail-lambda.
   *
   * @param tailLambdaNode Lambda node for which the memory nodes are initialized.
   *
   * @see InitializeLiveNodesOfTailLambdas()
   */
  void
  InitializeLiveNodesOfTailLambda(const lambda::node & tailLambdaNode);

  /**
   * The function checks the following invariants:
   *
   * 1. The set of memory nodes computed for each region and call node by
   * TopDownMemoryNodeEliminator are a subset of the corresponding set of memory nodes from the seed
   * provisioning.
   *
   * @param rvsdgModule The RVSDG module for which the provisioning is computed.
   * @param seedProvisioning The seed provisioning. \see EliminateMemoryNodes
   * @param provisioning The computed provisioning from TopDownMemoryNodeEliminator.
   *
   * @return Returns true if all invariants are fulfilled, otherwise false.
   */
  static bool
  CheckInvariants(
      const RvsdgModule & rvsdgModule,
      const MemoryNodeProvisioning & seedProvisioning,
      const Provisioning & provisioning);

  std::unique_ptr<Context> Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSIS_TOPDOWNMEMORYNODEELIMINATOR_HPP
