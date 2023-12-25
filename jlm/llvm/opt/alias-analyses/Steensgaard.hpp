/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/util/disjointset.hpp>

namespace jlm::llvm::aa
{

class Location;
class LocationSet;
class RegisterLocation;

/** \brief Steensgaard alias analysis
 *
 * This class implements a Steensgaard alias analysis. The analysis is inter-procedural,
 * field-insensitive, context-insensitive, flow-insensitive, and uses a static heap model. It is an
 * implementation corresponding to the algorithm presented in Bjarne Steensgaard - Points-to
 * Analysis in Almost Linear Time.
 */
class Steensgaard final : public AliasAnalysis
{
  class Statistics;

public:
  ~Steensgaard() override;

  Steensgaard();

  Steensgaard(const Steensgaard &) = delete;

  Steensgaard(Steensgaard &&) = delete;

  Steensgaard &
  operator=(const Steensgaard &) = delete;

  Steensgaard &
  operator=(Steensgaard &&) = delete;

  std::unique_ptr<PointsToGraph>
  Analyze(const RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector)
      override;

  /**
   * \brief Analyze RVSDG module
   *
   * \param module RVSDG module the analysis is performed on.
   *
   * \return A PointsTo graph.
   */
  std::unique_ptr<PointsToGraph>
  Analyze(const RvsdgModule & rvsdgModule);

private:
  void
  AnalyzeRvsdg(const jlm::rvsdg::graph & graph);

  void
  AnalyzeRegion(jlm::rvsdg::region & region);

  void
  AnalyzeLambda(const lambda::node & node);

  void
  AnalyzeDelta(const delta::node & node);

  void
  AnalyzePhi(const phi::node & node);

  void
  AnalyzeGamma(const jlm::rvsdg::gamma_node & node);

  void
  AnalyzeTheta(const jlm::rvsdg::theta_node & node);

  void
  AnalyzeSimpleNode(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeStructuralNode(const jlm::rvsdg::structural_node & node);

  void
  AnalyzeAlloca(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeMalloc(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeLoad(const LoadNode & loadNode);

  void
  AnalyzeStore(const StoreNode & storeNode);

  void
  AnalyzeCall(const CallNode & callNode);

  void
  AnalyzeDirectCall(const CallNode & callNode, const lambda::node & lambdaNode);

  void
  AnalyzeExternalCall(const CallNode & callNode);

  void
  AnalyzeIndirectCall(const CallNode & callNode);

  void
  AnalyzeGep(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeBitcast(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeBits2ptr(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeConstantPointerNull(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeUndef(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeMemcpy(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeConstantArray(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeConstantStruct(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeConstantAggregateZero(const jlm::rvsdg::simple_node & node);

  void
  AnalyzeExtractValue(const jlm::rvsdg::simple_node & node);

  /**
   * Construct a points-to graph from the Steensgaard analysis result.
   *
   * @return A points-to graph.
   */
  [[nodiscard]] std::unique_ptr<PointsToGraph>
  ConstructPointsToGraph() const;

  /**
   * Creates a points-to graph memory node for \p location.
   *
   * @param location The location for which a memory node is created.
   * @param pointsToGraph The points-to graph in which the memory node is created.
   *
   * @return A points-to graph memory node.
   *
   * \note This is a partial function as not all locations can be mapped to points-to graph memory
   * nodes.
   */
  static PointsToGraph::MemoryNode &
  CreatePointsToGraphMemoryNode(const Location & location, PointsToGraph & pointsToGraph);

  /**
   * Collects all memory nodes that escape the module. The \p escapingRegisterLocations
   * are the registers that were determined throughout the Steensgaard analysis to point to escaped
   * memory. They serve as starting point for the fix-point computation.
   *
   * @param escapingRegisterLocations All registers that point to escaped memory.
   * @param memoryNodesInSet The points-to memory nodes that are part of a location set.
   *
   * @return A set of memory nodes that escaped the module.
   */
  [[nodiscard]] util::HashSet<PointsToGraph::MemoryNode *>
  CollectEscapedMemoryNodes(
      const util::HashSet<RegisterLocation *> & escapingRegisterLocations,
      const std::unordered_map<
          const util::disjointset<Location *>::set *,
          std::vector<PointsToGraph::MemoryNode *>> & memoryNodesInSet) const;

  /**
   * Resolves all points-to graph nodes that were marked throughout the analysis as pointing
   * towards unknown memory locations, i.e., we have no idea where these marked nodes point to.
   * The resolving of these nodes simply happens by adding edges from them to \b all existing
   * points-to graph memory nodes. Note, that this strategy of resolving such nodes is extremely
   * expensive for points-to graphs with a lot of nodes marked as pointing towards unknown
   * memory locations as it leads to a quadratic number of edges.
   *
   * FIXME: We can do better than resolving these nodes by simply adding edges to all points-to
   * graph memory nodes. The requirement is that the users of these marked nodes, i.e,, stores,
   * loads, calls, etc., need to stay in the same place as dictated by the memory state given by
   * the original program. In other words, a store referencing an address that is marked as
   * unknown is not allowed to move before or after any other memory referencing operation.
   * However, this does not necessarily require that the address needs to point to all points-to
   * graph memory nodes in order to fulfill this requirement, it is just the simplest to go
   * about it. We simply speculate on the fact that unknown marked points-to graph nodes are
   * rare. If this turns out to be false, then we might want to invest in a better strategy.
   *
   * @param pointsToGraph The PointsToGraph constructed by ConstructPointsToGraph.
   */
  static void
  RedirectUnknownMemoryNodeSources(PointsToGraph & pointsToGraph);

  std::unique_ptr<LocationSet> LocationSet_;
};

}

#endif
