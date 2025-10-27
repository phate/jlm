/*
 * Copyright 2025 Lars Astrup Sundt <lars.astrup.sundt@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PartialRedundancyElimination_HPP
#define JLM_LLVM_OPT_PartialRedundancyElimination_HPP

#include "jlm/llvm/ir/operators/IntegerOperations.hpp"
#include "jlm/rvsdg/MatchType.hpp"
#include "jlm/rvsdg/traverser.hpp"
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <stdint.h>

namespace jlm::rvsdg
{
class DeltaNode;
class GammaNode;
class Graph;
class LambdaNode;
class Output;
class StructuralNode;
class ThetaNode;
class Region;
}

namespace jlm::llvm{


/** \brief Partial Redundancy Elimination
 *
 * A pass for doing partial redundancy analysis and elimination
 */
class PartialRedundancyElimination final : public jlm::rvsdg::Transformation
{
  class Context;
  class Statistics;

public:
  ~PartialRedundancyElimination() noexcept override;

  PartialRedundancyElimination();

  PartialRedundancyElimination(const PartialRedundancyElimination &) = delete;
  PartialRedundancyElimination(PartialRedundancyElimination &&) = delete;

  PartialRedundancyElimination &
  operator=(const PartialRedundancyElimination &) = delete;
  PartialRedundancyElimination &
  operator=(PartialRedundancyElimination &&) = delete;

  void
  Run(jlm::rvsdg::RvsdgModule & module, jlm::util::StatisticsCollector & statisticsCollector) override;



private:

  void TraverseTopDownRecursively(rvsdg::Region& reg,          void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node* node));

  static void dump_region(        PartialRedundancyElimination *pe, rvsdg::Node* node);
  static void dump_node(          PartialRedundancyElimination *pe, rvsdg::Node* node);
  static void initialize_interned_and_stats(PartialRedundancyElimination *pe, rvsdg::Node* node);

  size_t stat_theta_count;
  size_t stat_gamma_count;
  size_t stat_interned_literals_count;
  std::unordered_map< std::string, rvsdg::Node* >  interned_literals_;
  std::unordered_map< rvsdg::Node*, rvsdg::Node* > literals_nodes_;
};

}


#endif
