/*
 * Copyright 2025 Lars Astrup Sundt <lars.astrup.sundt@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_PartialRedundancyElimination_HPP
#define JLM_LLVM_OPT_PartialRedundancyElimination_HPP

#include "gvn.hpp"
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

struct ThetaData
{
  size_t stat_iteration_count;
  jlm::rvsdg::gvn::GVN_Val prism;
  jlm::rvsdg::gvn::BrittlePrism pre;
  jlm::rvsdg::gvn::BrittlePrism post;
};

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
  enum class ThetaMode{
    GVN_FIND_FIXED_POINT,
    GVN_FINALIZE,
  };
  ThetaMode gvn_mode_thetas_ = ThetaMode::GVN_FIND_FIXED_POINT;
  void TraverseTopDownRecursively(rvsdg::Region& reg,          void(*cb)(PartialRedundancyElimination* pe, rvsdg::Node* node));

  static void dump_region(          PartialRedundancyElimination *pe, rvsdg::Node* node);
  static void dump_node(            PartialRedundancyElimination *pe, rvsdg::Node* node);
  static void initialize_stats(     PartialRedundancyElimination *pe, rvsdg::Node* node);

  size_t stat_theta_count;
  size_t stat_gamma_count;
  std::unordered_map<rvsdg::Node*, ThetaData> thetas_;

  std::unordered_map< rvsdg::Output *, rvsdg::gvn::GVN_Val> output_to_gvn_;
  void RegisterGVN(rvsdg::Output * output, rvsdg::gvn::GVN_Val gvn){
    output_to_gvn_[output] = gvn; //overwrites old values
  }

  rvsdg::gvn::GVN_Manager gvn_;
  inline rvsdg::gvn::GVN_Val GVNOrZero(rvsdg::Output* edge){
    if (output_to_gvn_.find(edge) != output_to_gvn_.end()){
      return output_to_gvn_[edge];
    }
    return rvsdg::gvn::GVN_NO_VALUE;
  }

  inline rvsdg::gvn::GVN_Val GVNOrWarn(rvsdg::Output* edge, rvsdg::Node* ctx_node){
    if (output_to_gvn_.find(edge) != output_to_gvn_.end()){
      return output_to_gvn_[edge];
    }

    std::cout << "Logic error: missing input for edge" + ctx_node->DebugString() + std::to_string(ctx_node->GetNodeId());

    return rvsdg::gvn::GVN_NO_VALUE;
  }

  inline rvsdg::gvn::GVN_Val GVNOrPanic(rvsdg::Output* edge, rvsdg::Node* ctx_node){
    if (output_to_gvn_.find(edge) != output_to_gvn_.end()){
      return output_to_gvn_[edge];
    }
    throw std::runtime_error("Logic error: missing input for edge" + ctx_node->DebugString() + std::to_string(ctx_node->GetNodeId()));
  }

  /// -----------------------------------------------------------

  void GVN(rvsdg::Region& root);
  void GVN_VisitRegion(rvsdg::Region& reg);
  void GVN_VisitAllSubRegions(rvsdg::Node* node);
  void GVN_VisitNode(rvsdg::Node* node);
  void GVN_VisitGammaNode(rvsdg::Node* node);
  void GVN_VisitThetaNode(rvsdg::Node* tn);
  void GVN_FinalizeThetaNode(rvsdg::Node * node);
  void GVN_VisitLambdaNode(rvsdg::Node* ln);
  void GVN_VisitLeafNode(rvsdg::Node* node);

  void PassDownRegion(rvsdg::Region& reg);
  void PassDownAllSubRegions(rvsdg::Node* node);
  void PassDownNode(rvsdg::Node* node);
  void PassDownGammaNode(rvsdg::Node* gn);
  void PassDownThetaNode(rvsdg::Node* tn);
  void PassDownLambdaNode(rvsdg::Node* ln);
  void PassDownLeafNode(rvsdg::Node* node);
};

}


#endif
