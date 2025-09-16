/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSUMMARY_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSUMMARY_HPP

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::llvm::aa
{

/** \brief Mod/Ref Summary
 *
 * Contains the memory nodes that are required at the entry and exit of a region, and for call
 * nodes.
 */
class ModRefSummary
{
public:
  virtual ~ModRefSummary() noexcept = default;

  [[nodiscard]] virtual const PointsToGraph &
  GetPointsToGraph() const noexcept = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForRegion(const rvsdg::Region & region) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForNode(const rvsdg::Node & node) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForLambda(const rvsdg::LambdaNode & node) const = 0;
  /*
  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForLoad(const rvsdg::SimpleNode & loadNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForStore(const rvsdg::SimpleNode & storeNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForMemcpy(const rvsdg::SimpleNode & memcpyNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForFree(const rvsdg::SimpleNode & freeNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForCall(const rvsdg::SimpleNode & callNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForTheta(const rvsdg::ThetaNode & thetaNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForGamma(const rvsdg::GammaNode & gammaNode) const = 0;

  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefForLambda(const rvsdg::LambdaNode & lambdaNode) const = 0;
  */
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSUMMARY_HPP
