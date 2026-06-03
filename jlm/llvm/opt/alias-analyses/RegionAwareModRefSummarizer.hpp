/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

#include <queue>

namespace jlm::llvm::aa
{

class RegionAwareModRefSummary;
using ModRefSetIndex = uint32_t;

/** \brief Region-aware mod/ref summarizer
 *
 * The key idea of the region-aware memory mod/ref summarizer is to only route memory locations
 * into structural nodes if the memory is actually used within its regions. This ensures that no
 * superfluous states will be routed through structural nodes and renders them independent if they
 * do not reference the same memory location. The region-aware analysis proceeds as follows:
 *
 * 1. Call Graph Creation: creates a call graph by looking at all call operations.
 * This graph includes calls to external functions, and calls from external functions.
 * Each function is assigned to a strongly connected component.
 *
 * 2. Simple Alloca Set Creation: An alloca is "simple" if its address is never stored to
 * any memory location, except for other simple allocas.
 * The PointsToGraph is used to determine which allocas are simple.
 *
 * 3. Create sets of non-reentrant allocas for each region.
 * The requirements are:
 *  - the alloca must be simple
 *  - the alloca must not be reachable from any of the region's arguments,
 *    when following points-to edges in the \ref PointsToGraph.
 *
 * 4. Mod/Ref Graph Building: Creates a graph containing nodes for loads, stores, calls,
 * regions and functions. Each node has a Mod/Ref set, and edges propagate info.
 * Special edges are used between function body region -> function,
 * which filter away all simple allocas defined in the function that are not recursive.
 *
 * 5. Mod/Ref Graph Solving: Mod/Ref sets and flags are propagated along edges in the graph
 *
 * 6. Mod/Ref set materialization, converting implicit memory nodes into explicit memory nodes.
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

  RegionAwareModRefSummarizer &
  operator=(const RegionAwareModRefSummarizer &) = delete;

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
   * Creates a call graph including all functions in the module, and groups all functions into SCCs.
   * The resulting SCCs and topological order will be stored in the Context_ field.
   *
   * @param rvsdgModule the module for which a mod/ref summary is computed.
   */
  void
  createCallGraph(const rvsdg::RvsdgModule & rvsdgModule);

  /**
   * Creates a set containing all simple Allocas is the PointsToGraph.
   * An Alloca is simple if it is only reachable from other simple Allocas,
   * or from RegisterNodes, in the PointsToGraph.
   */
  static util::HashSet<PointsToGraph::NodeIndex>
  CreateSimpleAllocaSet(const PointsToGraph & pointsToGraph);

  /**
   *
   */
  util::HashSet<PointsToGraph::NodeIndex>
  getReachableSimpleAllocas(std::queue<PointsToGraph::NodeIndex> & nodes);

  /**
   * Gets the set of simple alloca nodes that it is possible to reach from \p region's arguments.
   * Reachability is defined in terms of the \ref PointsToGraph. A simple alloca is by definition
   * only reachable from register nodes and other simple alloca nodes,
   * so other types of memory nodes in the points-to graph can be ignored.
   * @param region the region whose arguments are checked
   * @return the set of simple allocas reachable from region arguments
   */
  util::HashSet<PointsToGraph::NodeIndex>
  GetSimpleAllocasReachableFromRegionArguments(const rvsdg::Region & region);

  /**
   * Gets the set of simple alloca nodes that it is possible to reach from the \p call's arguments.
   * Reachability is defined in terms of the \ref PointsToGraph. A simple alloca is by definition
   * only reachable from register nodes and other simple alloca nodes,
   * so other types of memory nodes in the points-to graph can be ignored.
   * @param call the region whose arguments are checked
   * @return the set of simple allocas reachable from the call's arguments
   */
  util::HashSet<PointsToGraph::NodeIndex>
  GetSimpleAllocasReachableFromCallArguments(const rvsdg::SimpleNode & call);

  /**
   * Uses the call graph to determine if the given function can ever be involved
   * in a recursive chain of function calls.
   *
   * @param lambda the function in question.
   * @return true if it is possible for lambda to be involved in recursion, false otherwise
   */
  bool
  IsRecursionPossible(const rvsdg::LambdaNode & lambda) const;

  /**
   * Creates subsets of the allocas defined in each region in the program,
   * containing only the allocas that are determined to be non-reentrant.
   * The requirements are:
   *  - The alloca is simple, i.e., not reachable from memory nodes in the \ref PointsToGraph.
   *  - It is not possible to reach the alloca from any of the region's arguments,
   *    by following edges in the \ref PointsToGraph.
   * @return the total number of non-reentrant allocas in the program
   */
  size_t
  CreateNonReentrantAllocaSets();

  /**
   * Adds the fact that everything in the ModRefSet \p from should also be included
   * in the ModRefSet \p to.
   */
  void
  AddModRefSimpleConstraint(ModRefSetIndex from, ModRefSetIndex to);

  /**
   * Defines a set of memory nodes to be blocked from the ModRefSet with the given \p index.
   * A ModRefSet can have at most one such blocklist.
   * The reference to the blocklist must stay valid until solving is finished.
   *
   * Note: The blocklist only prevents propagation during solving,
   * so the user must avoid adding blocked memory nodes manually.
   *
   * @see VerifyBlocklists to check that no blocked memory nodes have been added
   */
  void
  AddModRefSetBlocklist(
      ModRefSetIndex index,
      const util::HashSet<PointsToGraph::NodeIndex> & blocklist);

  /**
   * Creates ModRefSets for regions and nodes within the function.
   * The flow of MemoryNodes between sets is modeled by adding edges to the constraint graph.
   */
  void
  AnnotateFunction(const rvsdg::LambdaNode & lambda);

  /**
   * Recursive call used to make the given region, its nodes and its sub-regions all
   * be represented by the given ModRefSet.
   * @param region the region whose operations should be represented by the ModRefSet
   * @param modRefSet the index of the ModRefSet used to represent the region
   * @param lambda the function this region belongs to
   */
  void
  AnnotateRegion(
      const rvsdg::Region & region,
      ModRefSetIndex modRefSet,
      const rvsdg::LambdaNode & lambda);

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
   * @param minTargetSize an optional size requirement for targeted memory locations
   * @param mayMod if true, the operation may modifiy the memory, otherwise just read
   */
  void
  addPointerOriginTargets(
      ModRefSetIndex modRefSetIndex,
      const rvsdg::Output & origin,
      std::optional<size_t> minTargetSize,
      bool mayMod);

  ModRefSetIndex
  AnnotateLoad(const rvsdg::SimpleNode & loadNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateStore(const rvsdg::SimpleNode & storeNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateAlloca(const rvsdg::SimpleNode & allocaNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateMalloc(const rvsdg::SimpleNode & mallocNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateFree(const rvsdg::SimpleNode & freeNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateMemcpy(const rvsdg::SimpleNode & memcpyNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateMemset(const rvsdg::SimpleNode & memsetNode, const rvsdg::LambdaNode & lambda);

  ModRefSetIndex
  AnnotateCall(const rvsdg::SimpleNode & callNode, const rvsdg::LambdaNode & lambda);

  /**
   * Uses the simple and complex constraints to propagate MemoryNodes between ModRefSets
   * until all constraints are satisfied.
   */
  void
  SolveModRefSetConstraintGraph();

  /**
   * For all ModRefSets where a blocklist is defined,
   * checks that none of the MemoryNodes from the blocklist have been added to the ModRefSet.
   * @return true if all blocklists are satisfied.
   */
  bool
  VerifyBlocklists() const;

  /**
   * After solving, the ModRefSet representing all external functions is used to determine
   * if any non-escaped variables in the module are only referenced, and never written to.
   * These memory nodes can be considered constant, and should be omitted from all ModRefSets.
   */
  void
  determineReadOnlyMemory();

  /**
   * Goes through the solved ModRefSets and materializes implicitly included
   */
  void
  materializeSets();

  /**
   *
   */
  void
  materializeSetsInFunction(const rvsdg::LambdaNode & lambda);

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
