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

#include <type_traits>
#include <unordered_map>

namespace jlm::llvm::aa
{

/**
 * Enum representing the ways in which a ModRefSet may affect a memory node.
 * A Ref means the memory object may be loaded from.
 * A Mod means the memory object may we written to.
 *
 * The enum values are chosen such that bitwise OR results in the union of effects.
 */
enum ModRefEffect : uint8_t
{
  NoEffect = 0,
  RefOnly = 0b1,
  ModOnly = 0b10,
  ModRef = 0b11
};

/**
 * @return true if the given \p effect includes possibly referencing memory, false otherwise
 */
[[nodiscard]] inline bool
mayEffectReference(ModRefEffect effect)
{
  return effect & ModRefEffect::RefOnly;
}

/**
 * @return true if the given \p effect includes possibly modifying memory, false otherwise
 */
[[nodiscard]] inline bool
mayEffectModify(ModRefEffect effect)
{
  return effect & ModRefEffect::ModOnly;
}

/**
 * Union operator between two \ref ModRefEffect values.
 * @param a the first effect
 * @param b the second effect
 * @return a ModRefEffect representing both \p a and \p b
 */
[[nodiscard]] inline ModRefEffect
operator|(ModRefEffect a, ModRefEffect b)
{
  return static_cast<ModRefEffect>(
      static_cast<std::underlying_type_t<ModRefEffect>>(a)
      | static_cast<std::underlying_type_t<ModRefEffect>>(b));
}

/**
 * Assignment operator version of union of two \ref ModRefEffect values.
 */
inline ModRefEffect &
operator|=(ModRefEffect & a, ModRefEffect b)
{
  return a = (a | b);
}

/**
 * Checks if the effects represented by \p subset are all contained within the effects
 * represented by \p superset.
 * @param subset the subset effects to check
 * @param superset the superset effects to check against
 * @return true if \p subset is a subset of \p superset, false otherwise
 */
[[nodiscard]] inline bool
isEffectSubset(ModRefEffect subset, ModRefEffect superset)
{
  return (subset | superset) == superset;
}

/**
 * Class representing the set of memory nodes that may
 * be referenced and/or modified by some operation.
 *
 * Memory nodes that are marked constant, or memory that is provably never stored to,
 * can be omitted from all ModRefSets.
 *
 * Memory nodes can also be compressed into the external node.
 * Let A be a memory node. If the following implications hold in every ModRefSet in a function
 * F:
 *  - A is marked "Ref" -> External is marked "Ref"
 *  - A is marked "Mod" -> External is marked "Mod"
 * Then A can be omitted from all ModRefSets in F.
 * Operations on A will still be sequentialized by the state edge representing the external
 * node. Implementations of ModRefSummarizers should create subclasses of this class.
 */
class ModRefSet
{
public:
  [[nodiscard]] const std::unordered_map<PointsToGraph::NodeIndex, ModRefEffect> &
  getModRefNodes() const
  {
    return modRefNodes_;
  }

protected:
  // Prevent users of the ModRefSummary base class from accidentally copying sets by value,
  // or constructing empty ModRefSets. Instances should always be summarizer-specific subclasses.
  ModRefSet() = default;
  ModRefSet(const ModRefSet & other) = default;
  ModRefSet(ModRefSet && other) = default;
  ModRefSet &
  operator=(const ModRefSet & other) = default;
  ModRefSet &
  operator=(ModRefSet && other) = default;

  /**
   * The set of memory nodes in the ModRefSet, indexed by their index in a points to graph.
   */
  std::unordered_map<PointsToGraph::NodeIndex, ModRefEffect> modRefNodes_;
};

/** \brief Mod/Ref Summary
 *
 * Provides sets of memory nodes that may be referenced or modified in nodes and functions.
 */
class ModRefSummary
{
public:
  virtual ~ModRefSummary() noexcept = default;

  [[nodiscard]] virtual const PointsToGraph &
  GetPointsToGraph() const noexcept = 0;

  /**
   * Provides the \ref ModRefSet containing memory nodes that may be modified or referenced
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
