/*
 * Copyright 2023 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ANDERSEN_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ANDERSEN_HPP

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointerObjectSet.hpp>

namespace jlm::llvm::aa
{

/**
 * class implementing Andersen's set constraint based pointer analysis, based on the Ph.D. thesis
 * Lars Ole Andersen - Program Analysis and Specialization for the C Programming Language
 * The analysis is inter-procedural, field-insensitive, context-insensitive,
 * flow-insensitive, and uses a static heap model.
 */
class Andersen final : public AliasAnalysis
{
  class Statistics;

public:
  Andersen() = default;

  ~Andersen() noexcept override = default;

  Andersen(const Andersen &) = delete;

  Andersen(Andersen &&) = delete;

  Andersen &
  operator=(const Andersen &) = delete;

  Andersen &
  operator=(Andersen &&) = delete;

  /**
   * Performs Andersen's alias analysis on the rvsdg \p module,
   * producing a PointsToGraph describing what memory objects exists,
   * and which values in the rvsdg program may point to them.
   * @param module the module to analyze
   * @param statisticsCollector the collector that will receive pass statistics
   * @return A PointsToGraph for the module
   */
  std::unique_ptr<PointsToGraph>
  Analyze(const RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  /**
   * @brief Shorthand for Analyze, ignoring collecting any statistics.
   * @see Analyze
   */
  std::unique_ptr<PointsToGraph>
  Analyze(const RvsdgModule & module);

  /**
   * Converts a PointerObjectSet into PointsToGraph nodes,
   * and points-to-graph set memberships into edges.
   *
   * Note that registers sharing PointerObject, become separate PointsToGraph nodes.
   *
   * In the PointerObjectSet, the PointsToExternal flag encodes pointing to an address from outside
   * the module. This may however be the address of a memory object within the module, that has
   * escaped. In the final graph, any node marked as pointing to external, will get an edge to the
   * special "external" node, as well as to every memory object node marked as escaped.
   *
   * @return the newly created PointsToGraph
   */
  static std::unique_ptr<PointsToGraph>
  ConstructPointsToGraphFromPointerObjectSet(const PointerObjectSet & set);

private:
  void
  AnalyzeRegion(rvsdg::region & region);

  void
  AnalyzeSimpleNode(const rvsdg::simple_node & node);

  void
  AnalyzeAlloca(const rvsdg::simple_node & node);

  void
  AnalyzeMalloc(const rvsdg::simple_node & node);

  void
  AnalyzeLoad(const LoadNode & loadNode);

  void
  AnalyzeStore(const StoreNode & storeNode);

  void
  AnalyzeCall(const CallNode & callNode);

  void
  AnalyzeGep(const rvsdg::simple_node & node);

  void
  AnalyzeBitcast(const rvsdg::simple_node & node);

  void
  AnalyzeBits2ptr(const rvsdg::simple_node & node);

  void
  AnalyzePtr2bits(const rvsdg::simple_node & node);

  void
  AnalyzeConstantPointerNull(const rvsdg::simple_node & node);

  void
  AnalyzeUndef(const rvsdg::simple_node & node);

  void
  AnalyzeMemcpy(const rvsdg::simple_node & node);

  void
  AnalyzeConstantArray(const rvsdg::simple_node & node);

  void
  AnalyzeConstantStruct(const rvsdg::simple_node & node);

  void
  AnalyzeConstantAggregateZero(const rvsdg::simple_node & node);

  void
  AnalyzeExtractValue(const rvsdg::simple_node & node);

  void
  AnalyzeValist(const rvsdg::simple_node & node);

  void
  AnalyzeStructuralNode(const rvsdg::structural_node & node);

  void
  AnalyzeLambda(const lambda::node & node);

  void
  AnalyzeDelta(const delta::node & node);

  void
  AnalyzePhi(const phi::node & node);

  void
  AnalyzeGamma(const rvsdg::gamma_node & node);

  void
  AnalyzeTheta(const rvsdg::theta_node & node);

  void
  AnalyzeRvsdg(const rvsdg::graph & graph);

  std::unique_ptr<PointerObjectSet> Set_;
  std::unique_ptr<PointerObjectConstraintSet> Constraints_;
};

} // namespace

#endif
