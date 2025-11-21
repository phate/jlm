/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
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
 * Contains the memory nodes that are required to be routed into nodes and function bodies.
 */
class ModRefSummary
{
public:
  virtual ~ModRefSummary() noexcept = default;

  [[nodiscard]] virtual const PointsToGraph &
  GetPointsToGraph() const noexcept = 0;

  /**
   * Provides the set of memory nodes that represent memory locations that may be
   * modified or referenced by the given simple node.
   *
   * The simple node can be any operation that reads from memory, or produces value of memory, e.g.:
   *  - \ref LoadOperation and \ref StoreOperation nodes
   *  - \ref MemCpyOperation nodes
   *  - \ref FreeOperation nodes
   *  - \ref CallOperation nodes, i.e., function calls
   *  - \ref AllocaOperation and \ref MallocOperation nodes, which produce memory states
   *
   * @param node the node operating on memory
   * @return the Mod/Ref set of the node.
   */
  [[nodiscard]] virtual const util::HashSet<PointsToGraph::NodeIndex> &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const = 0;

  /**
   * Provides the set of memory nodes that should be routed into a given gamma node
   * @param gamma the gamma node
   * @return the entry Mod/Ref set for the gamma
   */
  [[nodiscard]] virtual const util::HashSet<PointsToGraph::NodeIndex> &
  GetGammaEntryModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of memory nodes that should be routed out of a given gamma node
   * @param gamma the gamma node
   * @return the exit Mod/Ref set for the gamma
   */
  [[nodiscard]] virtual const util::HashSet<PointsToGraph::NodeIndex> &
  GetGammaExitModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of memory nodes that should be routed in and out of a theta node
   * @param theta the theta node
   * @return the Mod/Ref set for the theta
   */
  [[nodiscard]] virtual const util::HashSet<PointsToGraph::NodeIndex> &
  GetThetaModRef(const rvsdg::ThetaNode & theta) const = 0;

  /**
   * Provides the set of memory nodes that are routed in to the given lambda's subregion
   * @param lambda the lambda node
   * @return the entry Mod/Ref set for the lambda
   */
  [[nodiscard]] virtual const util::HashSet<PointsToGraph::NodeIndex> &
  GetLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const = 0;

  /**
   * Provides the set of memory nodes that are routed out of the given lambda's subregion
   * @param lambda the lambda node
   * @return the exit Mod/Ref set for the lambda
   */
  [[nodiscard]] virtual const util::HashSet<PointsToGraph::NodeIndex> &
  GetLambdaExitModRef(const rvsdg::LambdaNode & lambda) const = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSUMMARY_HPP
