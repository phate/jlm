/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP

#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

namespace jlm::llvm::aa
{

// The type used for indexing Mod/Ref sets during analysis and solving
using ModRefNodeIndex = uint32_t;

// The final result of running the summarizer
class RegionAwareModRefSummary;

/** \brief Region-aware mod/ref summarizer
 *
 * The key idea of the region-aware memory mod/ref summarizer is to only provide memory locations
 * for a structural node that are actually utilized within its regions. This ensures that no
 * superfluous states will be routed through structural nodes and renders them independent if they
 * do not reference the same memory location. The region-aware analysis proceeds as follows:
 *
 * 1. Call Graph Creation: creates a call graph by looking at all call operations.
 * This graph includes calls to external functions, and calls from external functions.
 * Each function is assigned to a strongly connected component.
 *
 * 2. Find allocas that are dead in each SCC:
 * For each SCC in the call graph, only allocas defined within the SCC,
 * or within one of its predecessors, can be live.
 * All other allocas are placed in the DeadAllocasInScc lists.
 *
 * 3. Simple Alloca Set Creation: An Alloca is "simple" if its address is never stored to
 * any memory location, except for other simple Allocas.
 * The PointsToGraph is used to determine which allocas are simple.
 *
 * 4. Create sets of Non-Reentrant allocas for each region.
 *
 * 5. ModRefGraph Building: Creates a graph containing nodes for loads, stores, calls,
 * regions and functions. Each ModRefNode tracks memory nodes that are loaded from or stored to.
 * The nodes also have flags to represent operations on all external available memory nodes.
 * Edges in the ModRefGraph propagate sets and flags between nodes.
 *
 * 6. ModRefGraph Solving: sets and flags are propagated along edges in the graph
 *
 * 7. MemoryNodeOrder creation: all relevant memory nodes in the program are ordered in a list
 *
 * 8. ModRefSummary creation: the ModRefGraph is converted into the final result data structure
 *
 * @see ModRefSummarizer
 * @see MemoryStateEncoder
 */
class RegionAwareModRefSummarizer final : public ModRefSummarizer
{
  class Statistics;
  struct ModRefNode;
  class ModRefGraph;
  struct MemoryNodeOrderingMetadata;
  struct Context;
  class RegionAwareModRefSummary;

public:
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
   * For each SCC in the call graph, determines which allocas can be known to not be live
   * when a function from the SCC is at the top of the call stack.
   */
  void
  findAllocasDeadInSccs();

  /**
   * Creates a set containing all simple Allocas is the PointsToGraph.
   * An Alloca is simple if it is only reachable from other simple Allocas,
   * or from RegisterNodes, in the PointsToGraph.
   */
  static util::HashSet<PointsToGraph::NodeIndex>
  createSimpleAllocaSet(const PointsToGraph & pointsToGraph);

  util::HashSet<PointsToGraph::NodeIndex>
  getSimpleAllocasReachableFromRegionArguments(const rvsdg::Region & region);

  /**
   * Uses the call graph to determine if the given function can ever be involved
   * in a recursive chain of function calls.
   *
   * @param lambda the function in question.
   * @return true if it is possible for lambda to be involved in recursion, false otherwise
   */
  bool
  isRecursionPossible(const rvsdg::LambdaNode & lambda);

  /**
   * Creates a set for each region that contains alloca definitions,
   * where the alloca fits the requirements for being non-reentrant.
   * @return the total number of non-reentrant alloca never involved in any recursion
    // the alloca is definitely non-reentrant.s in the module.
   */
  size_t
  createNonReentrantAllocaSets();

  /**
   * Creates ModRefSets for regions and nodes within the function.
   * The flow of MemoryNodes between sets is modeled by adding edges to the constraint graph.
   */
  void
  annotateFunction(const rvsdg::LambdaNode & lambda);

  /**
   * Recursive call used to create ModRefSets for the given region, its nodes and its sub-regions.
   * @param region the region to create ModRefSets for.
   * @param lambda the function this region belongs to
   */
  ModRefNodeIndex
  annotateRegion(const rvsdg::Region & region, const rvsdg::LambdaNode & lambda);

  ModRefNodeIndex
  annotateStructuralNode(
      const rvsdg::StructuralNode & structuralNode,
      const rvsdg::LambdaNode & lambda);

  std::optional<ModRefNodeIndex>
  annotateSimpleNode(const rvsdg::SimpleNode & simpleNode, const rvsdg::LambdaNode & lambda);

  /**
   * Helper function for filling ModRefSets based on the pointer being operated on
   * @tparam IsStore true if the operation in a store, false if it is a load
   * @param modRefNode the index of the ModRefSet representing the memory operation
   * @param origin the output producing the pointer value being operated on
   * @param minTargetSize an optional size requirement for targeted memory locations
   * @param lambda the function the operation is happening in
   */
  template<bool IsStore>
  void
  addPointerOriginTargets(
      ModRefNodeIndex modRefNode,
      const rvsdg::Output & origin,
      std::optional<size_t> minTargetSize,
      const rvsdg::LambdaNode & lambda);

  ModRefNodeIndex
  annotateLoad(const rvsdg::SimpleNode & loadNode, const rvsdg::LambdaNode & lambda);

  ModRefNodeIndex
  annotateStore(const rvsdg::SimpleNode & storeNode, const rvsdg::LambdaNode & lambda);

  ModRefNodeIndex
  annotateAlloca(const rvsdg::SimpleNode & allocaNode);

  ModRefNodeIndex
  annotateMalloc(const rvsdg::SimpleNode & mallocNode);

  ModRefNodeIndex
  annotateFree(const rvsdg::SimpleNode & freeNode, const rvsdg::LambdaNode & lambda);

  ModRefNodeIndex
  annotateMemcpy(const rvsdg::SimpleNode & memcpyNode, const rvsdg::LambdaNode & lambda);

  ModRefNodeIndex
  annotateCall(const rvsdg::SimpleNode & callNode, const rvsdg::LambdaNode & lambda);

  /**
   * Creates a single ModRefSet responsible for representing all reads and writes
   * that may happen in external functions, or due to calls made from external functions.
   */
  void
  createExternalModRefNode();

  /**
   * Uses the results of solving the ModRefGraph to define an ordering of all relevant Memory Nodes.
   * Stores the created MemoryNodeOrdering in the context,
   * while also creating a MemoryNodeOrderingMetadata with important indices in the ordering.
   */
  void
  createMemoryNodeOrdering();

  /**
   * Uses the solved ModRefGraph and MemoryNodeOrdering to build the final ModRefSummary.
   * @return the created ModRefSummary
   */
  [[nodiscard]] std::unique_ptr<RegionAwareModRefSummary>
  createModRefSummary();

  /**
   * Helper function for debugging, listing out all functions, grouped by call graph SCC.
   */
  [[nodiscard]] std::string
  callGraphSCCsToString() const;

  /**
   * Converts \p rvsdg to an annotated region tree. This method is very useful for debugging the
   * RegionAwareMemoryNodeProvider.
   *
   * @param rvsdg The RVSDG that is converted to a region tree.
   *
   * @return A string that contains the region tree.
   */
  [[nodiscard]] std::string
  dumpRegionTree(const rvsdg::Graph & rvsdg);

  std::unique_ptr<Context> Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP
