/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP

#include <jlm/llvm/opt/alias-analyses/PointsToAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/util/disjointset.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Graph;
class output;
class Region;
class SimpleNode;
class StructuralNode;
class ThetaNode;
}

namespace jlm::llvm
{

namespace delta
{
class node;
}

namespace lambda
{
class node;
}

namespace phi
{
class node;
}

class LoadNode;
class StoreNode;

namespace aa
{
class Location;
class RegisterLocation;

/** \brief Steensgaard alias analysis
 *
 * This class implements a Steensgaard alias analysis. The analysis is inter-procedural,
 * field-insensitive, context-insensitive, flow-insensitive, and uses a static heap model. It is an
 * implementation corresponding to the algorithm presented in Bjarne Steensgaard - Points-to
 * Analysis in Almost Linear Time.
 */
class Steensgaard final : public PointsToAnalysis
{
  class Context;
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
  Analyze(const rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
      override;

  /**
   * \brief Analyze RVSDG module
   *
   * \param rvsdgModule RVSDG module the analysis is performed on.
   *
   * \return A PointsTo graph.
   */
  std::unique_ptr<PointsToGraph>
  Analyze(const RvsdgModule & rvsdgModule);

private:
  void
  AnalyzeRvsdg(const rvsdg::Graph & graph);

  void
  AnalyzeImports(const rvsdg::Graph & graph);

  void
  AnalyzeExports(const rvsdg::Graph & graph);

  void
  AnalyzeRegion(rvsdg::Region & region);

  void
  AnalyzeLambda(const rvsdg::LambdaNode & node);

  void
  AnalyzeDelta(const delta::node & node);

  void
  AnalyzePhi(const rvsdg::PhiNode & node);

  void
  AnalyzeGamma(const rvsdg::GammaNode & node);

  void
  AnalyzeTheta(const rvsdg::ThetaNode & node);

  void
  AnalyzeSimpleNode(const rvsdg::SimpleNode & node);

  void
  AnalyzeStructuralNode(const rvsdg::StructuralNode & node);

  void
  AnalyzeAlloca(const rvsdg::SimpleNode & node);

  void
  AnalyzeMalloc(const rvsdg::SimpleNode & node);

  void
  AnalyzeLoad(const rvsdg::SimpleNode & node);

  void
  AnalyzeStore(const rvsdg::SimpleNode & node);

  void
  AnalyzeCall(const rvsdg::SimpleNode & callNode);

  void
  AnalyzeDirectCall(const rvsdg::SimpleNode & callNode, const rvsdg::LambdaNode & lambdaNode);

  void
  AnalyzeExternalCall(const rvsdg::SimpleNode & callNode);

  void
  AnalyzeIndirectCall(const rvsdg::SimpleNode & callNode);

  void
  AnalyzeGep(const rvsdg::SimpleNode & node);

  void
  AnalyzeBitcast(const rvsdg::SimpleNode & node);

  void
  AnalyzeBits2ptr(const rvsdg::SimpleNode & node);

  void
  AnalyzePtrToInt(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantPointerNull(const rvsdg::SimpleNode & node);

  void
  AnalyzeUndef(const rvsdg::SimpleNode & node);

  void
  AnalyzeMemcpy(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantArray(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantStruct(const rvsdg::SimpleNode & node);

  void
  AnalyzeConstantAggregateZero(const rvsdg::SimpleNode & node);

  void
  AnalyzeExtractValue(const rvsdg::SimpleNode & node);

  void
  AnalyzeVaList(const rvsdg::SimpleNode & node);

  void
  AnalyzePointerToFunction(const rvsdg::SimpleNode & node);

  void
  AnalyzeIOBarrier(const rvsdg::SimpleNode & node);

  void
  AnalyzeFunctionToPointer(const rvsdg::SimpleNode & node);

  /**
   * Marks register \p output as escaping the module. This indicates that the pointer in \p output
   * is going outside the module, where we do not know what happens with it. Consequently, the
   * locations that \p output points to need to be marked as points-to-external and
   * points-to-escaped.
   *
   * @param output The register that is marked as escaping.
   */
  void
  MarkAsEscaped(const rvsdg::output & output);

  /**
   * Propagates the points-to flags throughout the disjoint set location graph.
   *
   * AnalyzeRvsdg() builds a disjoint set location graph, where each set is annotated with points-to
   * flags. This method propagates these points-to flags throughout the graph as all of these flags
   * are contagious. An example would be a set A that is marked as PointsToEscapedMemory and points
   * to another set B. As set B would "originate" from a load operation from "set A", the
   * consequence would be that set B must be marked as PointsToEscapedMemory as well.
   */
  void
  PropagatePointsToFlags();

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
          const util::DisjointSet<Location *>::set *,
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

  std::unique_ptr<Context> Context_;
};

}
}

#endif
