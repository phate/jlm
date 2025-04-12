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
class RegionSummary;

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
 * 2. These strongly connected components are then visited in reverse topological order.
 * This process is called Annotation, and creates summaries for all regions and calls.
 * These summaries contain the sets of memory locations that may be read from or written to
 * inside the region or by the target of the call.
 *
 * Since a call may target another function in the same SCC that has not been annotated yet,
 * a flag is set on all RegionSummary and CallSummary instances that may contain recursion.
 *
 * 3. Propagation: Once all functions have been annotated, we go back to all summaries that
 * were flagged as possibly containing recursion. Their sets of memory locations are expanded to
 * include all memory locations that may be affected by any function inside the SCC.
 *
 * @see ModRefSummarizer
 * @see MemoryStateEncoder
 */
class RegionAwareModRefSummarizer final : public ModRefSummarizer
{
public:
  class Statistics;

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
   * Creates a call graph including all functions in the module, and groups all functions into SCCs.
   * The resulting SCCs and topological order will be stored in the `FunctionSCCs_` field.
   *
   * @param rvsdgModule the module for which a mod/ref summary is computed.
   */
  void
  CreateCallGraph(const rvsdg::RvsdgModule & rvsdgModule);

  /**
   * Creates RegionSummaries and CallSummaries for the given function.
   * These summaries will contain the memory locations utilized in each region / call,
   * including in sub-regions and inside call targets.
   *
   * Assumes that all functions with lower sccIndex have already been annotated.
   * If regions or calls may contain recursion, a flag is set, to signal that the summaries are not
   * complete. By the time all functions in an SCC have been annotated, the field
   * SCCMemoryNodesAffected_ will contain the complete set of memory locations affected by the SCC.
   */
  void
  AnnotateFunction(const rvsdg::LambdaNode & lambda, size_t sccIndex);

  /**
   * Recursive call used to create summaries for the given region and all sub-regions.
   * @param region the region to create summaries for.
   * @param sccIndex the SCC index of the function being analyzed
   */
  RegionSummary &
  AnnotateRegion(rvsdg::Region & region, size_t sccIndex);

  /**
   * Recursively annotates all subregions in the given structural node.
   * Propagates memory locations to the parent region.
   * @param structuralNode the structural node
   * @param regionSummary the summary of the region containing the structural node
   */
  void
  AnnotateStructuralNode(
      const rvsdg::StructuralNode & structuralNode,
      RegionSummary & regionSummary);

  void
  AnnotateSimpleNode(const rvsdg::SimpleNode & simpleNode, RegionSummary & regionSummary);

  void
  AnnotateLoad(const rvsdg::SimpleNode & loadNode, RegionSummary & regionSummary);

  void
  AnnotateStore(const StoreNode & storeNode, RegionSummary & regionSummary);

  void
  AnnotateAlloca(const rvsdg::SimpleNode & allocaNode, RegionSummary & regionSummary);

  void
  AnnotateMalloc(const rvsdg::SimpleNode & mallocNode, RegionSummary & regionSummary);

  void
  AnnotateFree(const rvsdg::SimpleNode & freeNode, RegionSummary & regionSummary);

  void
  AnnotateMemcpy(const rvsdg::SimpleNode & memcpyNode, RegionSummary & regionSummary);

  void
  AnnotateCall(const CallNode & callNode, RegionSummary & regionSummary);

  /**
   * Revisits all Region- and Call-Summaries and adds utilized memory locations that were not
   * added during annotation, due to possibly being a part of a recursive call chain.
   * For summaries that are known to not contain recursion, nothing is done.
   */
  void
  PropagateRecursiveMemoryLocations();

  /**
   * Collects all lambda nodes defined in the given module, in an unspecified order.
   * @param rvsdgModule the module
   * @return a list of all lambda nodes in the module
   */
  static std::vector<const rvsdg::LambdaNode *>
  CollectLambdaNodes(const rvsdg::RvsdgModule & rvsdgModule);

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

  /**
   * Struct holding temporary data used during the creation of a single mod/ref summary
   */
  struct Context
  {
    /**
     * The set of functions belonging to each SCC in the call graph.
     * The SCCs are ordered in reverse topological order, so
     * if function a() calls b(), and they are not in the same SCC,
     * the SCC containing a() comes after the SCC containing b().
     */
    std::vector<util::HashSet<const rvsdg::LambdaNode *>> SccFunctions;

    /**
     * Which SCC contains the node representing external functions
     */
    size_t ExternalNodeSccIndex = 0;

    /**
     * A mapping from functions to the index of the SCC they belong to in the call graph
     */
    std::unordered_map<const rvsdg::LambdaNode *, size_t> FunctionToSccIndex;

    /**
     * The set of memory nodes that may be affected within a given SCC. Indexed by sccIndex.
     */
    std::vector<util::HashSet<const PointsToGraph::MemoryNode *>> SccSummaries;
  };

  Context Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_REGIONAWAREMODREFSUMMARIZER_HPP
