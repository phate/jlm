/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSIS_TOPDOWNMODREFELIMINATOR_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSIS_TOPDOWNMODREFELIMINATOR_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefEliminator.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
class CallTypeClassifier;
}

namespace jlm::rvsdg
{
class GammaNode;
class LambdaNode;
class Node;
class PhiNode;
class Region;
class SimpleNode;
class StructuralNode;
class ThetaNode;
}

namespace jlm::llvm::aa
{

/** \brief Top-down Mod/Ref eliminator
 *
 * The key idea of the \ref TopDownModRefEliminator is to restrict the lifetime of memory states by
 * eliminating the respective memory nodes from regions where the corresponding RVSDG nodes are not
 * alive. For example, the lifetime of a stack allocation from an alloca node is only within the
 * function the alloca node is alive.
 *
 * The \ref AgnosticModRefSummarizer and the \ref RegionAwareModRefSummarizer are only region-aware,
 * but not lifetime-aware. In other words, they restrict the number of regions a memory state needs
 * to be routed through, but do not limit the lifetime of the respective memory states. The Mod/Ref
 * summary produced by these summarizers serves as seed summary for the TopDownModRefEliminator,
 * which restricts then the lifetime of memory locations.
 *
 * The \ref TopDownModRefEliminator only restricts the lifetime of memory states from alloca nodes
 * before the nodes are alive.
 */
class TopDownModRefEliminator final : public ModRefEliminator
{
  class Context;
  class ModRefSummary;
  class Statistics;

public:
  ~TopDownModRefEliminator() noexcept override;

  TopDownModRefEliminator();

  TopDownModRefEliminator(const TopDownModRefEliminator &) = delete;

  TopDownModRefEliminator(TopDownModRefEliminator &&) = delete;

  TopDownModRefEliminator &
  operator=(const TopDownModRefEliminator &) = delete;

  TopDownModRefEliminator &
  operator=(TopDownModRefEliminator &&) = delete;

  std::unique_ptr<aa::ModRefSummary>
  EliminateModRefs(
      const rvsdg::RvsdgModule & rvsdgModule,
      const aa::ModRefSummary & seedModRefSummary,
      util::StatisticsCollector & statisticsCollector) override;

  /**
   * Creates a \ref TopDownModRefEliminator and calls the \ref EliminateModRefs() method.
   *
   * @param rvsdgModule The RVSDG module for which the Mod/Ref summary should be produced.
   * @param modRefSummary A Mod/Ref summary from which memory nodes will be eliminated.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return A new instance of \ref ModRefSummary.
   */
  static std::unique_ptr<aa::ModRefSummary>
  CreateAndEliminate(
      const rvsdg::RvsdgModule & rvsdgModule,
      const aa::ModRefSummary & modRefSummary,
      util::StatisticsCollector & statisticsCollector);

  /**
   * Creates a \ref TopDownModRefEliminator and calls the \ref EliminateModRefs() method.
   *
   * @param rvsdgModule The RVSDG module for which the Mod/Ref summary should be produced.
   * @param seedModRefSummary A Mod/Ref summary from which memory nodes will be eliminated.
   *
   * @return A new instance of \ref ModRefSummary.
   */
  static std::unique_ptr<aa::ModRefSummary>
  CreateAndEliminate(
      const rvsdg::RvsdgModule & rvsdgModule,
      const aa::ModRefSummary & seedModRefSummary);

private:
  void
  EliminateTopDown(const rvsdg::RvsdgModule & rvsdgModule);

  /**
   * Processes the inter-procedural RVSDG nodes (lambda, phi, and delta nodes) in the root region
   * or a phi subregion bottom-up. The bottom-up visitation ensures that all call nodes are
   * visited before the respective lambda nodes are visited.
   *
   * @param region The RVSDG root region or a phi subregion.
   */
  void
  EliminateTopDownRootRegion(rvsdg::Region & region);

  /**
   * Processes the intra-procedural nodes in a lambda, theta, or gamma subregion top-down. The
   * top-down visitation ensures that the live memory nodes are added to the live sets when the
   * respective RVSDG nodes appear in the execution order.
   *
   * @param region A lambda, theta, or gamma subregion.
   */
  void
  EliminateTopDownRegion(rvsdg::Region & region);

  void
  EliminateTopDownStructuralNode(const rvsdg::StructuralNode & structuralNode);

  void
  EliminateTopDownLambda(const rvsdg::LambdaNode & lambdaNode);

  void
  EliminateTopDownLambdaEntry(const rvsdg::LambdaNode & lambdaNode);

  void
  EliminateTopDownLambdaExit(const rvsdg::LambdaNode & lambdaNode);

  void
  EliminateTopDownPhi(const rvsdg::PhiNode & phiNode);

  void
  EliminateTopDownGamma(const rvsdg::GammaNode & gammaNode);

  void
  EliminateTopDownTheta(const rvsdg::ThetaNode & thetaNode);

  void
  EliminateTopDownSimpleNode(const rvsdg::SimpleNode & simpleNode);

  void
  EliminateTopDownAlloca(const rvsdg::SimpleNode & node);

  void
  EliminateTopDownCall(const rvsdg::SimpleNode & callNode);

  void
  EliminateTopDownNonRecursiveDirectCall(
      const rvsdg::SimpleNode & callNode,
      const CallTypeClassifier & callTypeClassifier);

  void
  EliminateTopDownRecursiveDirectCall(
      const rvsdg::SimpleNode & callNode,
      const CallTypeClassifier & callTypeClassifier);

  void
  EliminateTopDownExternalCall(
      const rvsdg::SimpleNode & callNode,
      const CallTypeClassifier & callTypeClassifier);

  void
  EliminateTopDownIndirectCall(
      const rvsdg::SimpleNode & indirectCall,
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
  InitializeLiveNodesOfTailLambdas(const rvsdg::RvsdgModule & rvsdgModule);

  /**
   * Initializes the memory nodes that are alive at the beginning of every tail-lambda.
   *
   * @param tailLambdaNode Lambda node for which the memory nodes are initialized.
   *
   * @see InitializeLiveNodesOfTailLambdas()
   */
  void
  InitializeLiveNodesOfTailLambda(const rvsdg::LambdaNode & tailLambdaNode);

  /**
   * The function checks the following invariants:
   *
   * 1. The set of memory nodes computed for each region and call node by
   * \ref TopDownModRefEliminator are a subset of the corresponding set of memory nodes from the
   * seed mod/ref summary.
   *
   * @param rvsdgModule The RVSDG module for which the mod/ref summary is computed.
   * @param seedModRefSummary The seed Mod/Ref summary. \see EliminateModRefs
   * @param modRefSummary The computed Mod/Ref summary from \ref TopDownModRefEliminator.
   *
   * @return Returns true if all invariants are fulfilled, otherwise false.
   */
  static bool
  CheckInvariants(
      const rvsdg::RvsdgModule & rvsdgModule,
      const aa::ModRefSummary & seedModRefSummary,
      const aa::ModRefSummary & modRefSummary);

  std::unique_ptr<Context> Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSIS_TOPDOWNMODREFELIMINATOR_HPP
