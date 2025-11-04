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

  // Default ctor needed for vector resize. Do not use
  MemoryNodeInterval() : start(0), end(0) {}

  explicit MemoryNodeInterval(MemoryNodeOrderingIndex start) : start(start), end(start + 1) {}

  MemoryNodeInterval(MemoryNodeOrderingIndex start, MemoryNodeOrderingIndex end) : start(start), end(end) {}

  /**
   * Intervals are sorted primarily by start index.
   * If the start indices are identical, the largest interval comes first.
   */
  bool
  operator<(const MemoryNodeInterval & other) const noexcept
  {
    if (start < other.start)
      return true;
    if (start == other.start)
      return end > other.end;
    return false;
  }
};

struct MemoryNodeIntervalSet
{
  // The set of intervals, which should be sorted according to the interval ordering rules.
  std::vector<MemoryNodeInterval> intervals;

  MemoryNodeIntervalSet() = default;

  explicit MemoryNodeIntervalSet(std::vector<MemoryNodeInterval> intervals)
      : intervals(std::move(intervals))
  {}

  /**
   * Sorts intervals and replaces overlapping/adjacent intervals with their union.
   */
  void
  sortAndCompact()
  {
    if (intervals.empty())
      return;

    std::sort(intervals.begin(), intervals.end());
    size_t currentInterval = 0;

    // Check if intervals can be merged into currentInterval instead of being new intervals
    for (size_t i = 1; i < intervals.size(); i++)
    {
      if (intervals[currentInterval].end >= intervals[i].start)
      {
        // The intervals are mergeable!
        intervals[currentInterval].end = std::max(intervals[currentInterval].end, intervals[i].end);
      }
      else
      {
        // The intervals are not mergeable, make i be the next currentInterval

        // If currentInterval is an empty interval, override it instead of incrementing
        if (intervals[currentInterval].start < intervals[currentInterval].end)
          currentInterval++;
        if (currentInterval != i)
          intervals[currentInterval] = intervals[i];
      }
    }

    // Move past the final interval, if it is non-empty
    if (intervals[currentInterval].start < intervals[currentInterval].end)
      currentInterval++;

    // Discard any intervals after the final interval
    intervals.resize(currentInterval);
  }
};

struct ModRefSet
{
  ModRefSet(MemoryNodeIntervalSet loads, MemoryNodeIntervalSet stores)
      : loads(std::move(loads)),
        stores(std::move(stores))
  {}

  // The set of memory nodes that are loaded from (ref)
  MemoryNodeIntervalSet loads;
  // The set of memory nodes that are stored to (mod)
  MemoryNodeIntervalSet stores;
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
  getMemoryNodeOrdering() = 0;

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
