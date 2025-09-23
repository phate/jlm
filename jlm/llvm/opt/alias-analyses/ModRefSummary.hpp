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
 * Contains the memory nodes that are required to be routed into nodes and function bodies.
 */
class ModRefSummary
{
public:
  virtual ~ModRefSummary() noexcept = default;

  [[nodiscard]] virtual const PointsToGraph &
  GetPointsToGraph() const noexcept = 0;

  /**
   * Provides the set of MemoryNodes that represent memory locations that may be
   * modified or referenced by the given simple node.
   *
   * The simple node can be any operation that affects memory:
   *  - Load, Store, Memcpy, Free, Call, which have memory states routed through them
   *  - Alloca or Malloc, which produce memory states
   *
   * @param node the node operating on memory
   * @return the Mod/Ref set of the node.
   */
  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const = 0;

  /**
   * Provides the set of MemoryNodes that should be routed into a given Gamma node
   * @param gamma the Gamma node
   * @return the entry Mod/Ref set for the Gamma
   */
  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaEntryModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of MemoryNodes that should be routed out of a given Gamma node
   * @param gamma the Gamma node
   * @return the exit Mod/Ref set for the Gamma
   */
  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaExitModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of MemoryNodes that should be routed in and out of a Theta node
   * @param theta the Theta node
   * @return the Mod/Ref set for the Theta
   */
  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetThetaModRef(const rvsdg::ThetaNode & theta) const = 0;

  /**
   * Provides the set of MemoryNodes that are routed in to the given Lambda's subregion
   * @param lambda the Lambda node
   * @return the entry Mod/Ref set for the Lambda
   */
  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const = 0;

  /**
   * Provides the set of MemoryNodes that are routed out of the given Lambda's subregion
   * @param lambda the Lambda node
   * @return the exit Mod/Ref set for the Lambda
   */
  [[nodiscard]] virtual const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaExitModRef(const rvsdg::LambdaNode & lambda) const = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSUMMARY_HPP
