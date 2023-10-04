/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>

namespace jlm::llvm::aa
{

class LocationSet;

/** \brief Steensgaard alias analysis
 *
 * This class implements a Steensgaard alias analysis. The analysis is inter-procedural, field-insensitive,
 * context-insensitive, flow-insensitive, and uses a static heap model. It is an implementation corresponding to the
 * algorithm presented in Bjarne Steensgaard - Points-to Analysis in Almost Linear Time.
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
  Analyze(
    const RvsdgModule & module,
    jlm::util::StatisticsCollector & statisticsCollector) override;

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

  static std::unique_ptr<PointsToGraph>
  ConstructPointsToGraph(const LocationSet & locationSets);

  static void
  RedirectUnknownMemoryNodeSources(PointsToGraph & pointsToGraph);

  std::unique_ptr<LocationSet> LocationSet_;
};

}

#endif
