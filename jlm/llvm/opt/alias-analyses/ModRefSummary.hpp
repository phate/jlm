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

#include <unordered_map>

namespace jlm::llvm::aa
{

/**
 * Class representing the set of memory nodes that may be referenced and/or modified
 * by some operation, or inside a structural node.
 *
 * Memory nodes that are marked constant, or memory that is provably never stored to,
 * can be omitted from all ModRefSets.
 *
 * Memory nodes can also be compressed into the external node.
 * Let A be a memory node. If the following implications hold in every ModRefSet in a function F:
 *  - A is marked "Ref" -> External is marked either "Mod" or "Ref"
 *  - A is marked "Mod" -> External is marked "Mod"
 * Then A can be omitted from all ModRefSets in F.
 * Operations on A will anyways be sequentialized properly using the external edge.
 *
 * Implementations of ModRefSummarizers should create subclasses of this class.
 */
class ModRefSet
{
public:
  [[nodiscard]] const std::unordered_map<PointsToGraph::NodeIndex, bool> &
  getModRefNodes() const
  {
    return modRefNodes_;
  }

protected:
  // Prevent users of the ModRefSummary from accidentally copying sets by value
  ModRefSet() = default;
  ModRefSet(const ModRefSet & other) = default;

  /**
   * The set of memory nodes in the ModRefSet, indexed by their index in a points to graph.
   * The boolean key indicates reference vs modification:
   *  - false means the memory nodes is only referenced
   *  - true means the memory nodes is possibly stored to
   */
  std::unordered_map<PointsToGraph::NodeIndex, bool> modRefNodes_;
};

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
   * Provides the ModRefSet containing memory nodes that may be modified or referenced
   * by the given simple node.
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
  [[nodiscard]] virtual const ModRefSet &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const = 0;

  /**
   * Provides the set of memory nodes that should be routed into a given gamma node
   * @param gamma the gamma node
   * @return the entry Mod/Ref set for the gamma
   */
  [[nodiscard]] virtual const ModRefSet &
  GetGammaEntryModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of memory nodes that should be routed out of a given gamma node
   * @param gamma the gamma node
   * @return the exit Mod/Ref set for the gamma
   */
  [[nodiscard]] virtual const ModRefSet &
  GetGammaExitModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of memory nodes that should be routed in and out of a theta node
   * @param theta the theta node
   * @return the Mod/Ref set for the theta
   */
  [[nodiscard]] virtual const ModRefSet &
  GetThetaModRef(const rvsdg::ThetaNode & theta) const = 0;

  /**
   * Provides the set of memory nodes that are routed in to the given lambda's subregion
   * @param lambda the lambda node
   * @return the entry Mod/Ref set for the lambda
   */
  [[nodiscard]] virtual const ModRefSet &
  GetLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const = 0;

  /**
   * Provides the set of memory nodes that are routed out of the given lambda's subregion
   * @param lambda the lambda node
   * @return the exit Mod/Ref set for the lambda
   */
  [[nodiscard]] virtual const ModRefSet &
  GetLambdaExitModRef(const rvsdg::LambdaNode & lambda) const = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSUMMARY_HPP
