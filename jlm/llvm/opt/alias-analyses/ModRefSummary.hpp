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
#include <jlm/util/common.hpp>

namespace jlm::llvm::aa
{

using MemoryNodeOrderingIndex = uint32_t;

/**
 * An ordering of all memory nodes in a PointsToGraph that are relevant to some ModRefSummary.
 */
class MemoryNodeOrdering
{
public:
  MemoryNodeOrdering(
      const PointsToGraph & pointsToGraph,
      std::vector<PointsToGraph::NodeIndex> memoryNodeOrder)
      : pointsToGraph_(pointsToGraph),
        memoryNodeOrder_(std::move(memoryNodeOrder))
  {}

  [[nodiscard]] const PointsToGraph &
  getPointsToGraph() const noexcept
  {
    return pointsToGraph_;
  }

  [[nodiscard]] MemoryNodeOrderingIndex
  numMemoryNodes() const noexcept
  {
    return memoryNodeOrder_.size();
  };

  [[nodiscard]] PointsToGraph::NodeIndex
  getMemoryNodeAt(MemoryNodeOrderingIndex index) const noexcept
  {
    JLM_ASSERT(index < memoryNodeOrder_.size());
    return memoryNodeOrder_[index];
  }

  [[nodiscard]] std::string
  getDebugString() const;

private:
  const PointsToGraph & pointsToGraph_;

  std::vector<PointsToGraph::NodeIndex> memoryNodeOrder_;
};

struct MemoryNodeInterval
{
  // Start index in the MemoryNodeOrdering. Inclusive
  MemoryNodeOrderingIndex start;
  // End index in the MemoryNodeOrdering. Exclusive
  MemoryNodeOrderingIndex end;

  // Default ctor needed for vector resize
  MemoryNodeInterval()
      : start(0),
        end(0)
  {}

  /**
   * Creates an interval containing the single memory node with the given index
   * @param index the position of the memory node in the MemoryNodeOrdering
   */
  explicit MemoryNodeInterval(MemoryNodeOrderingIndex index)
      : start(index),
        end(index + 1)
  {}

  /**
   * Creates an interval contaning every memory node with index in the range [start, end).
   * @param start the position of the first memory node in the MemoryNodeOrdering
   * @param end the position one after the last memory node in the MemoryNodeOrdering
   */
  MemoryNodeInterval(MemoryNodeOrderingIndex start, MemoryNodeOrderingIndex end)
      : start(start),
        end(end)
  {}

  [[nodiscard]] bool
  operator<(const MemoryNodeInterval & other) const noexcept
  {
    if (start != other.start)
      return start < other.start;
    return end < other.end;
  }
};

class MemoryNodeIntervalSet
{
public:
  MemoryNodeIntervalSet() = default;

  /**
   * Creates an interval set using the given \p intervals.
   * The intervals will be sorted, merged when possible, and empty intervals will be removed.
   * @param intervals the intervals to include in the set
   */
  explicit MemoryNodeIntervalSet(std::vector<MemoryNodeInterval> intervals)
      : intervals_(std::move(intervals))
  {
    sortAndCompact();
  }

  [[nodiscard]] const std::vector<MemoryNodeInterval> &
  getIntervals() const noexcept
  {
    return intervals_;
  }

  [[nodiscard]] size_t
  numIntervals() const noexcept
  {
    return intervals_.size();
  }

  [[nodiscard]] std::string
  getDebugString() const;

private:
  /**
   * Sorts intervals and replaces overlapping/adjacent intervals with their union.
   */
  void
  sortAndCompact();

  std::vector<MemoryNodeInterval> intervals_;
};

/**
 * Helper class for iterating over a memory node interval set
 */
class MemoryNodeIntervalSetIterator
{
public:
  explicit MemoryNodeIntervalSetIterator(const MemoryNodeIntervalSet & set);

  std::optional<MemoryNodeInterval>
  peek() const;

  void
  next();

private:
  const MemoryNodeIntervalSet & set_;
  size_t index_;
};

/**
 * Helper class for iterating over the union of two MemoryNodeIntervalSets
 */
class MemoryNodeIntervalSetUnionIterator
{
public:
  MemoryNodeIntervalSetUnionIterator(
      MemoryNodeIntervalSetIterator a,
      MemoryNodeIntervalSetIterator b);

  std::optional<MemoryNodeInterval>
  peek() const;

  void
  next();

private:
  MemoryNodeIntervalSetIterator a_;
  MemoryNodeIntervalSetIterator b_;
  std::optional<MemoryNodeInterval> current_;
};

/**
 * Helper class for iterating over intervals that are stored to,
 * and intervals that are loaded from but NOT stored to.
 */
class MemoryNodeIntervalSetDifferenceIterator
{
public:
  MemoryNodeIntervalSetDifferenceIterator(
      MemoryNodeIntervalSetIterator loads,
      MemoryNodeIntervalSetIterator stores);

  /**
   * @return a pair representing the current interval, if there is one. Otherwise noneopt.
   * The returned pair contains the interval itself, as well as a boolean that is true if the
   * interval gets stored to.
   */
  std::optional<std::pair<MemoryNodeInterval, bool>>
  peek() const;

  void
  next();

private:
  MemoryNodeIntervalSetIterator loads_;
  MemoryNodeIntervalSetIterator stores_;
  std::optional<std::pair<MemoryNodeInterval, bool>> current_;

  // Due to the possibility of a store interval cutting off a load interval,
  // we might need to restart in the middle of the load interval.
  // This field gives the earliest possible start for the next interval.
  MemoryNodeOrderingIndex lastEnd_;
};

class ModRefSet
{
public:
  ModRefSet(MemoryNodeIntervalSet loadIntervals, MemoryNodeIntervalSet storeIntervals)
      : loadIntervals_(std::move(loadIntervals)),
        storeIntervals_(std::move(storeIntervals))
  {}

  /**
   * Checks if the ModRefSet represents doing absolutely nothing with any memory
   * @return true if the ModRefSet does nothing to memory, false otherwise
   */
  [[nodiscard]] bool
  isEmpty() const noexcept
  {
    return loadIntervals_.numIntervals() == 0 && storeIntervals_.numIntervals() == 0;
  }

  const MemoryNodeIntervalSet &
  getLoadIntervals() const noexcept
  {
    return loadIntervals_;
  }

  [[nodiscard]] MemoryNodeIntervalSetIterator
  getLoadIntervalIterator() const noexcept
  {
    return MemoryNodeIntervalSetIterator(loadIntervals_);
  }

  const MemoryNodeIntervalSet &
  getStoreIntervals() const noexcept
  {
    return storeIntervals_;
  }

  [[nodiscard]] MemoryNodeIntervalSetIterator
  getStoreIntervalIterator() const noexcept
  {
    return MemoryNodeIntervalSetIterator(storeIntervals_);
  }

  /**
   * @return an iterator providing intervals containing all memory locations that are either
   * stored to or loaded from.
   */
  [[nodiscard]] MemoryNodeIntervalSetUnionIterator
  getLoadStoreIntervalIterator() const noexcept
  {
    return MemoryNodeIntervalSetUnionIterator(
        getLoadIntervalIterator(),
        getStoreIntervalIterator());
  }

  /**
   * @return an iterator providing intervals containing all memory locations that are stored to
   * or loaded from, or both.
   * For each interval provided by the stream, a boolean indicates if the memory locations are
   * being stored to, or only loaded from.
   */
  [[nodiscard]] MemoryNodeIntervalSetDifferenceIterator
  getLoadStoreIntervalDifferenceIterator() const noexcept
  {
    return MemoryNodeIntervalSetDifferenceIterator(
        getLoadIntervalIterator(),
        getStoreIntervalIterator());
  }

  [[nodiscard]] std::string
  getDebugString() const;

private:
  // The set of memory nodes that are loaded from (ref)
  MemoryNodeIntervalSet loadIntervals_;
  // The set of memory nodes that are stored to (mod)
  MemoryNodeIntervalSet storeIntervals_;
};

/** \brief Mod/Ref Summary
 *
 * Contains ModRefSets for all operations and structural nodes that perform loads and/or
 * stores to memory.
 */
class ModRefSummary
{
public:
  virtual ~ModRefSummary() noexcept = default;

  [[nodiscard]] virtual const MemoryNodeOrdering &
  getMemoryNodeOrdering() const = 0;

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
  [[nodiscard]] virtual const ModRefSet &
  getSimpleNodeModRef(const rvsdg::SimpleNode & node) const = 0;

  /**
   * Provides the set of MemoryNodes that should be routed into a given Gamma node
   * @param gamma the Gamma node
   * @return the entry Mod/Ref set for the Gamma
   */
  [[nodiscard]] virtual const ModRefSet &
  getGammaEntryModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of MemoryNodes that should be routed out of a given Gamma node
   * @param gamma the Gamma node
   * @return the exit Mod/Ref set for the Gamma
   */
  [[nodiscard]] virtual const ModRefSet &
  getGammaExitModRef(const rvsdg::GammaNode & gamma) const = 0;

  /**
   * Provides the set of MemoryNodes that should be routed in and out of a Theta node
   * @param theta the Theta node
   * @return the Mod/Ref set for the Theta
   */
  [[nodiscard]] virtual const ModRefSet &
  getThetaModRef(const rvsdg::ThetaNode & theta) const = 0;

  /**
   * Provides the set of MemoryNodes that are routed in to the given Lambda's subregion
   * @param lambda the Lambda node
   * @return the entry Mod/Ref set for the Lambda
   */
  [[nodiscard]] virtual const ModRefSet &
  getLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const = 0;

  /**
   * Provides the set of MemoryNodes that are routed out of the given Lambda's subregion
   * @param lambda the Lambda node
   * @return the exit Mod/Ref set for the Lambda
   */
  [[nodiscard]] virtual const ModRefSet &
  getLambdaExitModRef(const rvsdg::LambdaNode & lambda) const = 0;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_MODREFSUMMARY_HPP
