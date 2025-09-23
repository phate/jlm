/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

namespace jlm::llvm::aa
{

class RegionAwareModRefSummary;
using ModRefSetIndex = uint32_t;

/** \brief Region-aware mod/ref summarizer
 *
 * The key idea of the region-aware memory mod/ref summarizer is to only provide memory locations
 * for a structural node that are actually utilized within its regions. This ensures that no
 * superfluous states will be routed through structural nodes and renders them independent if they
 * do not reference the same memory location. The region-aware analysis proceeds as follows:
 *
 * 1. Simple Alloca Set Creation: An Alloca is "simple" if its address is never stored to
 * any memory location, except for other simple Allocas.
 * The PointsToGraph is used to determine which allocas are simple.
 *
 * 2. Create sets of Non-Reentrant allocas for each function.
 *
 * 3. Call Graph Creation: creates a call graph by looking at all call operations.
 * This graph includes calls to external functions, and calls from external functions.
 * Each function is assigned to a strongly connected component.
 *
 * 4. Find allocas that are dead in each SCC:
 * For each SCC in the call graph, only allocas defined within the SCC,
 * or within one of its predecessors, can be live.
 * All other allocas are placed in the DeadAllocasInScc lists.
 *
 * 5. Mod/Ref Graph Building: Creates a graph containing nodes for loads, stores, calls,
 * regions and functions. Each node has a Mod/Ref set, and edges propagate info.
 * Special edges are used between function body region -> function,
 * which filter away all simple allocas defined in the function that are not recursive.
 *
 * 6. Mod/Ref Graph Solving: Mod/Ref sets are propagated along edges in the graph
 *
 * @see ModRefSummarizer
 * @see MemoryStateEncoder
 */
class RegionAwareModRefSummarizer final : public ModRefSummarizer
{
public:
  class Statistics;
  struct Context;

  ~RegionAwareModRefSummarizer() noexcept override;

  RegionAwareModRefSummarizer();

  RegionAwareModRefSummarizer(const RegionAwareModRefSummarizer &) = delete;

  RegionAwareModRefSummarizer(RegionAwareModRefSummarizer &&) = delete;

  RegionAwareModRefSummarizer &
  operator=(const RegionAwareModRefSummarizer &) = delete;

  RegionAwareModRefSummarizer &
  operator=(RegionAwareModRefSummarizer &&) = delete;

  std::unique_ptr<ModRefSummary>
  SummarizeModRefs(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector) override;

  /**
   * Creates a RegionAwareModRefSummarizer and calls the SummarizeModRefs() method.
   *
   * @param rvsdgModule The RVSDG module for which the \ref ModRefSummary should be computed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   * @param statisticsCollector The statistics collector for collecting pass statistics.
   *
   * @return A new instance of ModRefSummary.
   */
  static std::unique_ptr<ModRefSummary>
  Create(
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph,
      util::StatisticsCollector & statisticsCollector);

  /**
   * Creates a RegionAwareModRefSummarizer and calls the SummarizeModRefs() method.
   *
   * @param rvsdgModule The RVSDG module for which the \ref ModRefSummary should be computed.
   * @param pointsToGraph The PointsToGraph corresponding to the RVSDG module.
   *
   * @return A new instance of ModRefSummary.
   */
  static std::unique_ptr<ModRefSummary>
  Create(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph);

private:
  /**
   * Creates a set containing all simple Allocas is the PointsToGraph.
   * An Alloca is simple if it is only reachable from other simple Allocas,
   * or from RegisterNodes, in the PointsToGraph.
   */
  static util::HashSet<const PointsToGraph::MemoryNode *>
  CreateSimpleAllocaSet(const PointsToGraph & pointsToGraph);

  /**
   * Creates a set for each function f containing allocas defined in f that are both
   * simple, and not accessible through f's parameters. These allocas are known as Non-Reentrant.
   */
  void
  CreateNonReentrantAllocaSets();

  /**
   * Creates a call graph including all functions in the module, and groups all functions into SCCs.
   * The resulting SCCs and topological order will be stored in the `FunctionSCCs_` field.
   *
   * @param rvsdgModule the module for which a mod/ref summary is computed.
   */
  void
  CreateCallGraph(const rvsdg::RvsdgModule & rvsdgModule);

  /**
   * Collects all lambda nodes defined in the given module, in an unspecified order.
   * @param rvsdgModule the module
   * @return a list of all lambda nodes in the module
   */
  static std::vector<const rvsdg::LambdaNode *>
  CollectLambdaNodes(const rvsdg::RvsdgModule & rvsdgModule);

  /**
   * For each SCC in the call graph, determines which allocas may be live while a
   * function from the SCC is at the top of the call stack.
   */
  void
  FindAllocasLiveInSccs();

  /**
   * Creates one ModRefSet which is responsible for representing all reads and writes
   * that may happen in external functions.
   */
  void
  CreateExternalModRefSet();

  /**
   * Adds the fact that everything in the ModRefSet \p from should also be included
   * in the ModRefSet \p to.
   */
  void
  AddModRefSimpleConstraint(ModRefSetIndex from, ModRefSetIndex to);

  /**
   * Adds the fact that everything in the ModRefSet \p from should also be included
   * in the ModRefSet of the function \p to, except MemoryNodes that can be shown to not need
   * their state routed into the function by callers.
   */
  void
  AddModRefComplexConstraint(ModRefSetIndex from, const rvsdg::LambdaNode & to);

  /**
   * Creates ModRefSets for regions and nodes within the function.
   * The flow of MemoryNodes between sets is modeled by adding edges to the constraint graph.
   */
  void
  AnnotateFunction(const rvsdg::LambdaNode & lambda);

  /**
   * Recursive call used to create ModRefSets for the given region, its nodes and its sub-regions.
   * @param region the region to create ModRefSets for.
   * @param lambda the function this region belongs to
   */
  ModRefSetIndex
  AnnotateRegion(const rvsdg::Region & region, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateStructuralNode(
      const rvsdg::StructuralNode & structuralNode,
      const rvsdg::LambdaNode & lambda);

  std::optional<ModRefSetIndex>
  AnnotateSimpleNode(const rvsdg::SimpleNode & simpleNode, const rvsdg::LambdaNode & lambda);

  /**
   * Helper function for filling ModRefSets based on the pointer being operated on
   * @param modRefSetIndex the index of the ModRefSet representing some memory operation
   * @param origin the output producing the pointer value being operated on
   * @param lambda the function the operation is happening in
   */
  void
  AnnotateWithPointerOrigin(
      ModRefSetIndex modRefSetIndex,
      const rvsdg::Output & origin,
      const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateLoad(const rvsdg::SimpleNode & loadNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateStore(const rvsdg::SimpleNode & storeNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateAlloca(const rvsdg::SimpleNode & allocaNode);

  ModRefSetIndex
  AnnotateMalloc(const rvsdg::SimpleNode & mallocNode);

  ModRefSetIndex
  AnnotateFree(const rvsdg::SimpleNode & freeNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateMemcpy(const rvsdg::SimpleNode & memcpyNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateCall(const rvsdg::SimpleNode & callNode, const rvsdg::LambdaNode & lambda);

  /**
   * Uses the simple and complex constraints to propagate MemoryNodes between ModRefSets
   * until all constraints are satisfied.
   */
  void
  SolveModRefSetConstraintGraph();

  /**
   * Helper function for debugging, listing out all functions, grouped by call graph SCC.
   */
  static std::string
  CallGraphSCCsToString(const RegionAwareModRefSummarizer & summarizer);

  /**
   * Converts \p rvsdg to an annotated region tree. This method is very useful for debugging the
   * RegionAwareMemoryNodeProvider.
   *
   * @param rvsdg The RVSDG that is converted to a region tree.
   * @param modRefSummary The Mod/Ref summary used for annotating the region tree.
   *
   * @return A string that contains the region tree.
   */
  static std::string
  ToRegionTree(const rvsdg::Graph & rvsdg, const RegionAwareModRefSummary & modRefSummary);

  /**
   * The Mod/Ref summary produced by this summarizer
   */
  std::unique_ptr<RegionAwareModRefSummary> ModRefSummary_;

  std::unique_ptr<Context> Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP
