/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/LambdaMemoryState.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryStateEncoder.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm::aa
{

/**
 * \brief Helper struct for counting up MemoryNodes, among some set of entities that use them
 */
struct MemoryStateTypeCounter final
{
  // The number of entities that have been counted
  uint64_t NumEntities = 0;

  // How many memory nodes have been counted, in total
  uint64_t NumMemoryNodes = 0;

  // How many intervals have been counted in total across all entities
  uint64_t NumIntervals = 0;
  // Remember the single entity with the highest number of intervals
  uint64_t MaxIntervals = 0;

  void
  CountEntity(uint64_t numMemoryNodes, uint16_t numIntervals)
  {
    NumEntities++;
    NumMemoryNodes += numMemoryNodes;
    NumIntervals += numIntervals;
    if (numIntervals > MaxIntervals)
      MaxIntervals = numIntervals;
  }

  void
  CountEntity(const ModRefSet & modRefSet)
  {
    uint64_t numMemoryNodes = 0;
    uint64_t numIntervals = 0;

    auto intervals = modRefSet.getLoadStoreIntervalIterator();
    while (const auto interval = intervals.peek())
    {
      numMemoryNodes += interval->end - interval->start;
      numIntervals++;
      intervals.next();
    }

    CountEntity(numMemoryNodes, numIntervals);
  }
};

/** \brief Statistics class for memory state encoder encoding
 *
 */
class EncodingStatistics final : public util::Statistics
{
  // These are prefixes for statistics that count MemoryNode types
  static constexpr auto NumTotalMemoryNodes_ = "#TotalMemoryNodes";
  static constexpr auto NumTotalIntervals_ = "#TotalIntervals";
  static constexpr auto MaxIntervals_ = "#MaxIntervals";

  // The number of regions that are inside lambda nodes (including the lambda subregion itself)
  static constexpr auto NumIntraProceduralRegions_ = "#IntraProceduralRegions";
  // Suffix used when counting region state arguments (or LambdaEntrySplit for lambda subregions)
  static constexpr auto RegionArgumentStateSuffix_ = "Arguments";

  // Counting both volatile and non-volatile loads
  static constexpr auto NumLoadOperations_ = "#LoadOperations";
  // Suffix used when counting memory states routed through loads
  static constexpr auto LoadStateSuffix_ = "sThroughLoad";

  // Counting both volatile and non-volatile stores
  static constexpr auto NumStoreOperations_ = "#StoreOperations";
  // Suffix used when counting memory states routed through stores
  static constexpr auto StoreStateSuffix_ = "sThroughStore";

  // Counting call entry merges
  static constexpr auto NumCallOperations_ = "#CallOperations";
  // Suffix used when counting memory states routed into call entry merges
  static constexpr auto CallStateSuffix_ = "sThroughCall";

public:
  ~EncodingStatistics() override = default;

  explicit EncodingStatistics(const util::FilePath & sourceFile)
      : Statistics(Statistics::Id::MemoryStateEncoder, sourceFile)
  {}

  void
  Start(const rvsdg::Graph & graph)
  {
    AddMeasurement(Label::NumRvsdgNodesBefore, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  Stop()
  {
    GetTimer(Label::Timer).stop();
  }

  void
  AddIntraProceduralRegionMemoryStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumIntraProceduralRegions_, counter.NumEntities);
    AddMemoryStateTypeCounter(RegionArgumentStateSuffix_, counter);
  }

  void
  AddLoadMemoryStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumLoadOperations_, counter.NumEntities);
    AddMemoryStateTypeCounter(LoadStateSuffix_, counter);
  }

  void
  AddStoreMemoryStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumStoreOperations_, counter.NumEntities);
    AddMemoryStateTypeCounter(StoreStateSuffix_, counter);
  }

  void
  AddCallMemoryStateCounts(const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumCallOperations_, counter.NumEntities);
    AddMemoryStateTypeCounter(CallStateSuffix_, counter);
  }

  static std::unique_ptr<EncodingStatistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<EncodingStatistics>(sourceFile);
  }

private:
  void
  AddMemoryStateTypeCounter(const std::string & suffix, const MemoryStateTypeCounter & counter)
  {
    AddMeasurement(NumTotalMemoryNodes_ + suffix, counter.NumMemoryNodes);
    AddMeasurement(NumTotalIntervals_ + suffix, counter.NumIntervals);
    AddMeasurement(MaxIntervals_ + suffix, counter.MaxIntervals);
  }
};

/** \brief Context for the memory state encoder
 */
class MemoryStateEncoder::Context final
{
public:
  explicit Context(const ModRefSummary & modRefSummary)
      : ModRefSummary_(modRefSummary)
  {}

  Context(const Context &) = delete;

  Context(Context &&) = delete;

  Context &
  operator=(const Context &) = delete;

  Context &
  operator=(Context &&) = delete;

  const ModRefSummary &
  GetModRefSummary() const noexcept
  {
    return ModRefSummary_;
  }

  MemoryStateTypeCounter &
  GetInterProceduralRegionCounter()
  {
    return InterProceduralRegionCounter_;
  }

  MemoryStateTypeCounter &
  GetLoadCounter()
  {
    return LoadCounter_;
  }

  MemoryStateTypeCounter &
  GetStoreCounter()
  {
    return StoreCounter_;
  }

  MemoryStateTypeCounter &
  GetCallCounter()
  {
    return CallCounter_;
  }

  static std::unique_ptr<Context>
  Create(const ModRefSummary & modRefSummary)
  {
    return std::make_unique<Context>(modRefSummary);
  }

private:
  const ModRefSummary & ModRefSummary_;

  // Counters used for producing statistics about memory states
  MemoryStateTypeCounter InterProceduralRegionCounter_;
  MemoryStateTypeCounter LoadCounter_;
  MemoryStateTypeCounter StoreCounter_;
  MemoryStateTypeCounter CallCounter_;
};

/**
 * Represents a single interval that was stored to or loaded from by some operation
 */
struct LiveInterval
{
  // The interval of memory nodes
  MemoryNodeInterval interval;

  // If true, the operation represents the modification of the memory nodes in its interval
  bool isStore;

  // The node containing the operation.
  // Alternatively, nullptr if the live interval represents the region entry memory state
  rvsdg::Node * node;

  // The memory state output of the node.
  // Alternatively, the region argument if this is the memory at the start of the region.
  rvsdg::Output * memoryStateOutput;
};

/**
 * Class representing a set of (possibly overlapping) intervals and which memory state outputs
 * produced it and used it.
 */
class RegionIntervalOutputMapping
{
public:
  RegionIntervalOutputMapping(rvsdg::Region & region)
      : region_(region)
  {}

  /**
   * Registers the region's memory state argument as the source of all MemoryNodes in its ModRefSet.
   * This must be the first method called on the instance.
   * @param memoryStateArgument the memory state argument of the region
   * @param modRefSet the mod ref set of the region
   */
  void
  createRegionEntry(rvsdg::Output & memoryStateArgument, const ModRefSet & modRefSet)
  {
    JLM_ASSERT(memoryStateArgument.region() == &region_);
    JLM_ASSERT(liveIntervals_.empty());

    auto intervals = modRefSet.getLoadStoreIntervalIterator();
    while (const auto interval = intervals.peek())
    {
      liveIntervals_.push_back(LiveInterval{ *interval, true, nullptr, &memoryStateArgument });
      intervals.next();
    }

    JLM_ASSERT(isValid());
  }

  /**
   * Adds the intervals from the given ModRefSet to the live intervals,
   * without removing any other live intervals.
   * @param node the node that creates the memory state
   * @param memoryStateOutput the memory state output
   * @param modRefSet the ModRefSet of the node
   */
  void
  addModRefSet(rvsdg::Node & node, rvsdg::Output & memoryStateOutput, const ModRefSet & modRefSet)
  {
    // Move the current live intervals into old live intervals
    std::swap(oldLiveIntervals_, liveIntervals_);
    // This method copies all surviving old intervals, as well as adding the new intervals.
    liveIntervals_.clear();

    // The index of the old interval being processed next, in the oldLiveIntervals_ array.
    size_t oldLiveIntervalIndex = 0;
    // The stream of new intervals
    auto newIntervals = modRefSet.getLoadStoreIntervalDifferenceIterator();

    while (const auto newIntervalPair = newIntervals.peek())
    {
      const auto [newInterval, newIntervalIsStore] = *newIntervalPair;

      while (oldLiveIntervalIndex < oldLiveIntervals_.size())
      {
        const auto oldInterval = oldLiveIntervals_[oldLiveIntervalIndex];
        if (oldInterval.interval.start > newInterval.start)
        {
          // We can not process this old interval yet
          break;
        }

        liveIntervals_.push_back(oldInterval);
        oldLiveIntervalIndex++;
      }

      liveIntervals_.push_back(
          LiveInterval{ newInterval, newIntervalIsStore, &node, &memoryStateOutput });
      newIntervals.next();
    }

    // Add any remaining old intervals
    while (oldLiveIntervalIndex < oldLiveIntervals_.size())
    {
      liveIntervals_.push_back(oldLiveIntervals_[oldLiveIntervalIndex]);
      oldLiveIntervalIndex++;
    }

    JLM_ASSERT(isValid());
  }

  /**
   * Uses the current set of live loads and stores to attach a node with the given \p modRefSet.
   * It consumes memory state outputs from operations it depends on.
   * It also updates the sets of live loads and stores
   *
   * @param node the node being attached
   * @param memoryStateOutput the node's memory state output
   * @param modRefSet the ModRefSet of the node
   * @return the MemoryStateSetNode created to provide the memory state for the node.
   */
  rvsdg::SimpleNode &
  attachNode(rvsdg::Node & node, rvsdg::Output & memoryStateOutput, const ModRefSet & modRefSet)
  {
    // Keep track of which memory state outputs the attached node will depend on
    std::vector<rvsdg::Output *> memoryStateOutputs;
    // Outputs in this set are the ones we depend on, used to avoid duplication
    util::HashSet<rvsdg::Output *> dependentOutputs;
    // Outputs in this set have been confirmed to be safe to NOT depend on
    util::HashSet<rvsdg::Output *> independentOutputs;

    // Move the current live intervals into old live intervals
    std::swap(oldLiveIntervals_, liveIntervals_);
    // This method copies all surviving old intervals, as well as adding the new intervals.
    liveIntervals_.clear();

    // The index of the old interval being processed next, in the oldLiveIntervals_ array.
    size_t oldLiveIntervalIndex = 0;

    // When an old interval is partially removed, the part that is left over is placed here
    std::vector<LiveInterval> leftoverOldIntervals;
    // Double-buffering of leftoverOldIntervals
    std::vector<LiveInterval> oldLeftoverOldIntervals;

    // Importantly, while oldLiveIntervals are allowed to contain overlapping intervals,
    // the new intervals never overlap!
    auto newIntervals = modRefSet.getLoadStoreIntervalDifferenceIterator();
    while (const auto newIntervalPair = newIntervals.peek())
    {
      const auto newInterval = newIntervalPair->first;
      const auto newIntervalIsStore = newIntervalPair->second;

      // Set to true once all old intervals that start earlier than newInterval have been processed,
      // and the newInterval itself has been added
      bool newIntervalAdded = false;

      // Processes an old interval. Will possibly split up the interval and add what remains of it
      // to the right of the newInterval into the leftoverOldIntervals.
      // Every time it is called the start index must be at least as large as the last call.
      // The start index of the old interval must be lower than the end of the new interval
      const auto processOldInterval = [&](const LiveInterval & oldInterval)
      {
        JLM_ASSERT(oldInterval.interval.start < newInterval.end);

        if (oldInterval.interval.end <= newInterval.start)
        {
          // The old interval finishes fully before newInterval, add is as it
          liveIntervals_.push_back(oldInterval);
          return;
        }

        // Make sure the new interval gets its proper place in the liveIntervals list
        if (oldInterval.interval.start >= newInterval.start && !newIntervalAdded)
        {
          liveIntervals_.push_back({ newInterval, newIntervalIsStore, &node, &memoryStateOutput });
          newIntervalAdded = true;
        }

        // If we get here there is overlap between the new and old interval
        bool eraseOverlap = false;
        if (newIntervalIsStore || oldInterval.isStore)
        {
          if (dependentOutputs.Contains(oldInterval.memoryStateOutput))
          {
            eraseOverlap = newIntervalIsStore;
          }
          else if (independentOutputs.Contains(oldInterval.memoryStateOutput))
          {
            eraseOverlap = false;
          }
          else
          {
            // TODO: Add pairwise alias analysis checking
            dependentOutputs.insert(oldInterval.memoryStateOutput);
            memoryStateOutputs.push_back(oldInterval.memoryStateOutput);
            eraseOverlap = newIntervalIsStore;
          }
        }

        if (eraseOverlap)
        {
          // We must erase overlap, which means keeping any part of oldInterval that precedes
          // or comes after newInterval
          if (oldInterval.interval.start < newInterval.start)
          {
            // There is a part of oldInterval before the newInterval
            auto oldIntervalBeforeNew = oldInterval;
            oldIntervalBeforeNew.interval.end = newInterval.start;
            liveIntervals_.push_back({ oldIntervalBeforeNew });
          }
          if (oldInterval.interval.end > newInterval.end)
          {
            // There is a part of oldInterval after the newInterval, add it to the leftover list
            auto leftoverInterval = oldInterval;
            leftoverInterval.interval.start = newInterval.end;
            leftoverOldIntervals.push_back(leftoverInterval);
          }
        }
        else
        {
          // If we are not required to replace the overlap, we can just add the old interval as is
          liveIntervals_.push_back(oldInterval);
        }
      };

      // Any intervals left over after processing the last newInterval should be handled first
      std::swap(leftoverOldIntervals, oldLeftoverOldIntervals);
      leftoverOldIntervals.clear();

      for (auto leftover : oldLeftoverOldIntervals)
      {
        processOldInterval(leftover);
      }

      // Process all intervals that start before the end of newInterval
      while (oldLiveIntervalIndex < oldLiveIntervals_.size())
      {
        const auto oldInterval = oldLiveIntervals_[oldLiveIntervalIndex];
        if (oldInterval.interval.start >= newInterval.end)
        {
          // oldInterval starts after the current newInterval. We are not ready to process it yet
          break;
        }

        processOldInterval(oldInterval);
        oldLiveIntervalIndex++;
      }

      // If the newInterval was never added during processing of old intervals, add it now
      if (!newIntervalAdded)
      {
        liveIntervals_.push_back(
            LiveInterval{ newInterval, newIntervalIsStore, &node, &memoryStateOutput });
      }
      newIntervals.next();
    }

    // All new intervals have been processed, add any remaining old intervals
    for (auto leftover : leftoverOldIntervals)
    {
      liveIntervals_.push_back(leftover);
    }
    while (oldLiveIntervalIndex < oldLiveIntervals_.size())
    {
      liveIntervals_.push_back(oldLiveIntervals_[oldLiveIntervalIndex]);
      oldLiveIntervalIndex++;
    }

    JLM_ASSERT(isValid());

    return MemoryStateSetOperation::CreateNode(region_, memoryStateOutputs);
  }

  /**
   * Creates a memory state set node for the end of the region,
   * joining together every live operation that has loaded from or stored to memory nodes that
   * are included in the given \p modRefSet.
   * This must be the last method called on the instance.
   * @param modRefSet the ModRefSet of the region
   * @return the MemoryStateSetNode created to provide the final memory state output
   */
  rvsdg::SimpleNode &
  createSetNodeForRegionExit(const ModRefSet & modRefSet)
  {
    std::vector<rvsdg::Output *> memoryStateOutputs;
    // Used to avoid duplicates
    util::HashSet<rvsdg::Output *> outputSet;

    size_t liveIntervalIndex = 0;

    auto intervals = modRefSet.getLoadStoreIntervalIterator();
    while (const auto interval = intervals.peek())
    {
      while (liveIntervalIndex < liveIntervals_.size())
      {
        auto & liveInterval = liveIntervals_[liveIntervalIndex];
        if (liveInterval.interval.end <= interval->start)
        {
          // We can move past this live interval, as it is fully behind the current interval
          liveIntervalIndex++;
          continue;
        }
        if (liveInterval.interval.start >= interval->end)
        {
          // We can not process this live interval yet, as it comes after the current interval
          break;
        }

        // We have found a live interval that overlaps with the current interval
        if (outputSet.insert(liveInterval.memoryStateOutput))
          memoryStateOutputs.push_back(liveInterval.memoryStateOutput);
        liveIntervalIndex++;
      }

      intervals.next();
    }

    return MemoryStateSetOperation::CreateNode(region_, memoryStateOutputs);
  }

  /**
   * Checks that the set of live intervals does not contain any empty intervals,
   * and that the intervals are sorted by increasing start index.
   * @return false if validation failed
   */
  bool
  isValid() const
  {
    for (size_t i = 0; i < liveIntervals_.size(); ++i)
    {
      if (liveIntervals_[i].interval.start >= liveIntervals_[i].interval.end)
        return false;
      if (i >= 1 && liveIntervals_[i].interval.start < liveIntervals_[i - 1].interval.start)
        return false;
    }

    return true;
  }

  /**
   * @return a string containing all live intervals
   */
  [[nodiscard]] std::string
  getDebugString() const
  {
    std::ostringstream ss;
    ss << "LiveIntervals for region " << &region_ << ":" << std::endl;
    for (const auto & liveInterval : liveIntervals_)
    {
      ss << "[" << liveInterval.interval.start << ", " << (liveInterval.interval.end - 1) << "]";
      if (liveInterval.isStore)
      {
        ss << " store";
      }
      else
      {
        ss << " load";
      }

      if (liveInterval.node)
      {
        ss << " (node " << liveInterval.node << ")";
      }

      ss << " (output " << liveInterval.memoryStateOutput << ")" << std::endl;
    }

    return ss.str();
  }

private:
  rvsdg::Region & region_;

  // Live intervals, always stored in order of increasing start index
  std::vector<LiveInterval> liveIntervals_;

  // Used to "double-buffer" the set of live intervals, by swapping current and old.
  // This avoids making new heap allocations for every traversal
  std::vector<LiveInterval> oldLiveIntervals_;
};

MemoryStateEncoder::~MemoryStateEncoder() noexcept = default;

MemoryStateEncoder::MemoryStateEncoder() = default;

void
MemoryStateEncoder::Encode(
    rvsdg::RvsdgModule & rvsdgModule,
    const ModRefSummary & modRefSummary,
    util::StatisticsCollector & statisticsCollector)
{
  Context_ = Context::Create(modRefSummary);
  auto statistics = EncodingStatistics::Create(rvsdgModule.SourceFilePath().value());

  std::cerr << modRefSummary.getMemoryNodeOrdering().getDebugString() << std::endl;

  statistics->Start(rvsdgModule.Rvsdg());
  EncodeInterProceduralRegion(rvsdgModule.Rvsdg().GetRootRegion());
  statistics->Stop();

  statistics->AddIntraProceduralRegionMemoryStateCounts(
      Context_->GetInterProceduralRegionCounter());
  statistics->AddLoadMemoryStateCounts(Context_->GetLoadCounter());
  statistics->AddStoreMemoryStateCounts(Context_->GetStoreCounter());
  statistics->AddCallMemoryStateCounts(Context_->GetCallCounter());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done with the encoding
  Context_.reset();

  // Remove all nodes that became dead throughout the encoding.
  DeadNodeElimination deadNodeElimination;
  deadNodeElimination.Run(rvsdgModule, statisticsCollector);
}

void
MemoryStateEncoder::EncodeInterProceduralRegion(rvsdg::Region & region)
{
  rvsdg::TopDownTraverser traverser(&region);
  for (const auto node : traverser)
  {
    MatchTypeOrFail(
        *node,
        [&](rvsdg::PhiNode & phiNode)
        {
          EncodePhi(phiNode);
        },
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          EncodeLambda(lambdaNode);
        },
        [&]([[maybe_unused]] rvsdg::DeltaNode & deltaNode)
        {
          // Nothing to be done for global variable definitions
        });
  }
}

void
MemoryStateEncoder::EncodePhi(rvsdg::PhiNode & phiNode)
{
  EncodeInterProceduralRegion(*phiNode.subregion());
}

void
MemoryStateEncoder::EncodeLambda(rvsdg::LambdaNode & lambdaNode)
{
  RegionIntervalOutputMapping liveIntervals(*lambdaNode.subregion());

  EncodeLambdaEntry(lambdaNode, liveIntervals);
  EncodeIntraProceduralRegion(*lambdaNode.subregion(), liveIntervals);
  EncodeLambdaExit(lambdaNode, liveIntervals);
}

void
MemoryStateEncoder::EncodeLambdaEntry(
    rvsdg::LambdaNode & lambdaNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  const auto & modRefSummary = Context_->GetModRefSummary();
  auto & memoryStateArgument = GetMemoryStateRegionArgument(lambdaNode);
  const auto & lambdaModRefSet = modRefSummary.getLambdaEntryModRef(lambdaNode);
  liveIntervals.createRegionEntry(memoryStateArgument, lambdaModRefSet);

  std::cerr << "Encoding lamda region entry for " << lambdaNode.DebugString() << std::endl;
  std::cerr << "ModRefSet:" << lambdaModRefSet.getDebugString() << std::endl;
  std::cerr << liveIntervals.getDebugString() << std::endl;

  Context_->GetInterProceduralRegionCounter().CountEntity(lambdaModRefSet);
}

void
MemoryStateEncoder::EncodeLambdaExit(
    rvsdg::LambdaNode & lambdaNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  const auto & modRefSummary = Context_->GetModRefSummary();
  auto & memoryStateResult = GetMemoryStateRegionResult(lambdaNode);
  const auto & lambdaModRefSet = modRefSummary.getLambdaExitModRef(lambdaNode);

  // Group together all memory nodes that appear loaded from or stored to from outside the function
  auto & joinNode = liveIntervals.createSetNodeForRegionExit(lambdaModRefSet);
  memoryStateResult.divert_to(joinNode.output(0));

  std::cerr << "Encoding lamda region exit for " << lambdaNode.DebugString() << std::endl;
  std::cerr << "ModRefSet:" << lambdaModRefSet.getDebugString() << std::endl;
  std::cerr << liveIntervals.getDebugString() << std::endl;
}

void
MemoryStateEncoder::EncodeIntraProceduralRegion(
    rvsdg::Region & region,
    RegionIntervalOutputMapping & liveIntervals)
{
  rvsdg::TopDownTraverser traverser(&region);
  for (const auto node : traverser)
  {
    MatchTypeOrFail(
        *node,
        [&](rvsdg::SimpleNode & simpleNode)
        {
          EncodeSimpleNode(simpleNode, liveIntervals);
        },
        [&](rvsdg::GammaNode & gammaNode)
        {
          EncodeGamma(gammaNode, liveIntervals);
        },
        [&](rvsdg::ThetaNode & thetaNode)
        {
          EncodeTheta(thetaNode, liveIntervals);
        });
  }
}

void
MemoryStateEncoder::EncodeSimpleNode(
    rvsdg::SimpleNode & simpleNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  if (is<AllocaOperation>(&simpleNode))
  {
    EncodeAlloca(simpleNode, liveIntervals);
  }
  else if (is<MallocOperation>(&simpleNode))
  {
    EncodeMalloc(simpleNode, liveIntervals);
  }
  else if (is<LoadOperation>(&simpleNode))
  {
    EncodeLoad(simpleNode, liveIntervals);
  }
  else if (is<StoreOperation>(&simpleNode))
  {
    EncodeStore(simpleNode, liveIntervals);
  }
  else if (is<CallOperation>(&simpleNode))
  {
    EncodeCall(simpleNode, liveIntervals);
  }
  else if (is<FreeOperation>(&simpleNode))
  {
    EncodeFree(simpleNode, liveIntervals);
  }
  else if (is<MemCpyOperation>(&simpleNode))
  {
    EncodeMemcpy(simpleNode, liveIntervals);
  }
  else if (is<MemoryStateOperation>(&simpleNode))
  {
    // Nothing needs to be done
  }
}

void
MemoryStateEncoder::EncodeAlloca(
    rvsdg::SimpleNode & allocaNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  JLM_ASSERT(is<AllocaOperation>(&allocaNode));

  const auto & allocaModRefSet = Context_->GetModRefSummary().getSimpleNodeModRef(allocaNode);
  auto & allocaMemoryStateOutput = *allocaNode.output(1);
  liveIntervals.addModRefSet(allocaNode, allocaMemoryStateOutput, allocaModRefSet);

  std::cerr << "Encoding alloca " << &allocaNode << std::endl;
  std::cerr << "ModRefSet:" << allocaModRefSet.getDebugString() << std::endl;
  std::cerr << liveIntervals.getDebugString() << std::endl;
}

void
MemoryStateEncoder::EncodeMalloc(
    rvsdg::SimpleNode & mallocNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  JLM_ASSERT(is<MallocOperation>(&mallocNode));

  const auto & mallocModRefSet = Context_->GetModRefSummary().getSimpleNodeModRef(mallocNode);
  auto & mallocMemoryStateOutput = *mallocNode.output(1);
  liveIntervals.addModRefSet(mallocNode, mallocMemoryStateOutput, mallocModRefSet);
}

void
MemoryStateEncoder::EncodeLoad(
    rvsdg::SimpleNode & node,
    RegionIntervalOutputMapping & liveIntervals)
{
  JLM_ASSERT(is<LoadOperation>(&node));

  const auto & loadModRefSet = Context_->GetModRefSummary().getSimpleNodeModRef(node);
  Context_->GetLoadCounter().CountEntity(loadModRefSet);

  auto memoryStateInputs = LoadOperation::MemoryStateInputs(node);
  JLM_ASSERT(std::distance(memoryStateInputs.begin(), memoryStateInputs.end()) == 1);
  auto & loadMemoryStateInput = *memoryStateInputs.begin();

  auto memoryStateOutputs = LoadOperation::MemoryStateOutputs(node);
  JLM_ASSERT(std::distance(memoryStateOutputs.begin(), memoryStateOutputs.end()) == 1);
  auto & loadMemoryStateOutput = *memoryStateOutputs.begin();

  auto & setNode = liveIntervals.attachNode(node, loadMemoryStateOutput, loadModRefSet);
  loadMemoryStateInput.divert_to(setNode.output(0));

  std::cerr << "Encoding load " << &node << std::endl;
  std::cerr << "ModRefSet:" << loadModRefSet.getDebugString() << std::endl;
  std::cerr << liveIntervals.getDebugString() << std::endl;
}

void
MemoryStateEncoder::EncodeStore(
    rvsdg::SimpleNode & node,
    RegionIntervalOutputMapping & liveIntervals)
{
  JLM_ASSERT(is<StoreOperation>(&node));

  const auto & storeModRefSet = Context_->GetModRefSummary().getSimpleNodeModRef(node);
  Context_->GetStoreCounter().CountEntity(storeModRefSet);

  auto memoryStateInputs = StoreOperation::MemoryStateInputs(node);
  JLM_ASSERT(std::distance(memoryStateInputs.begin(), memoryStateInputs.end()) == 1);
  auto & storeMemoryStateInput = *memoryStateInputs.begin();

  auto memoryStateOutputs = StoreOperation::MemoryStateOutputs(node);
  JLM_ASSERT(std::distance(memoryStateOutputs.begin(), memoryStateOutputs.end()) == 1);
  auto & storeMemoryStateOutput = *memoryStateOutputs.begin();

  auto & setNode = liveIntervals.attachNode(node, storeMemoryStateOutput, storeModRefSet);
  storeMemoryStateInput.divert_to(setNode.output(0));

  std::cerr << "Encoding store " << &node << std::endl;
  std::cerr << "ModRefSet:" << storeModRefSet.getDebugString() << std::endl;
  std::cerr << liveIntervals.getDebugString() << std::endl;
}

void
MemoryStateEncoder::EncodeFree(
    rvsdg::SimpleNode & freeNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  JLM_ASSERT(is<FreeOperation>(&freeNode));

  const auto & freeModRefSet = Context_->GetModRefSummary().getSimpleNodeModRef(freeNode);

  // TODO: Use proper accessors
  JLM_ASSERT(freeNode.ninputs() == 2);
  JLM_ASSERT(freeNode.noutputs() == 1);
  auto & memoryStateInput = *freeNode.input(1);
  auto & memoryStateOutput = *freeNode.output(0);

  auto & setNode = liveIntervals.attachNode(freeNode, memoryStateOutput, freeModRefSet);
  memoryStateInput.divert_to(setNode.output(0));
}

void
MemoryStateEncoder::EncodeCall(
    rvsdg::SimpleNode & callNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  const auto & callModRefSet = Context_->GetModRefSummary().getSimpleNodeModRef(callNode);
  Context_->GetCallCounter().CountEntity(callModRefSet);

  auto & memoryStateInput = CallOperation::GetMemoryStateInput(callNode);
  auto & memoryStateOutput = CallOperation::GetMemoryStateOutput(callNode);

  auto & setNode = liveIntervals.attachNode(callNode, memoryStateOutput, callModRefSet);
  memoryStateInput.divert_to(setNode.output(0));

  std::cerr << "Encoding call node " << &callNode << std::endl;
  std::cerr << "ModRefSet:" << callModRefSet.getDebugString() << std::endl;
  std::cerr << liveIntervals.getDebugString() << std::endl;
}

void
MemoryStateEncoder::EncodeMemcpy(
    rvsdg::SimpleNode & memcpyNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  const auto & memcpyModRefSet = Context_->GetModRefSummary().getSimpleNodeModRef(memcpyNode);

  const auto & op = *util::assertedCast<const MemCpyOperation>(&memcpyNode.GetOperation());
  JLM_ASSERT(op.NumMemoryStates() == 1);

  // TODO: Use proper accessors
  auto & memoryStateInput = *memcpyNode.input(memcpyNode.ninputs() - 1);
  auto & memoryStateOutput = *memcpyNode.output(memcpyNode.noutputs() - 1);

  auto & setNode = liveIntervals.attachNode(memcpyNode, memoryStateOutput, memcpyModRefSet);
  memoryStateInput.divert_to(setNode.output(0));
}

void
MemoryStateEncoder::EncodeGamma(
    rvsdg::GammaNode & gammaNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  const auto & gammaModRefSet = Context_->GetModRefSummary().getGammaEntryModRef(gammaNode);

  // Find a memory state entry variable we can use
  std::optional<rvsdg::GammaNode::EntryVar> memoryStateEntryVar;
  const auto entryVars = gammaNode.GetEntryVars();
  for (auto & entryVar : entryVars)
  {
    if (is<MemoryStateType>(entryVar.input->Type()))
    {
      if (memoryStateEntryVar.has_value())
        throw std::logic_error("Multiple memory state entry variables found");
      memoryStateEntryVar = entryVar;
    }
  }

  std::optional<rvsdg::GammaNode::ExitVar> memoryStateExitVar;
  const auto exitVars = gammaNode.GetExitVars();
  for (auto & exitVar : exitVars)
  {
    if (is<MemoryStateType>(exitVar.output->Type()))
    {
      if (memoryStateExitVar.has_value())
        throw std::logic_error("Multiple memory state exit variables found");
      memoryStateExitVar = exitVar;
    }
  }

  if (!gammaModRefSet.isEmpty())
  {
    if (!memoryStateEntryVar.has_value() || !memoryStateExitVar.has_value())
    {
      throw std::logic_error(
          "Gamma node with ModRefSet must have a memory state entry and exit variable");
    }

    auto & setNode =
        liveIntervals.attachNode(gammaNode, *memoryStateExitVar->output, gammaModRefSet);
    memoryStateEntryVar->input->divert_to(setNode.output(0));
  }

  for (size_t i = 0; i < gammaNode.nsubregions(); ++i)
  {
    auto & subregion = *gammaNode.subregion(i);
    RegionIntervalOutputMapping subregionLiveIntervals(subregion);
    if (memoryStateEntryVar.has_value())
    {
      subregionLiveIntervals.createRegionEntry(
          *memoryStateEntryVar->branchArgument[i],
          gammaModRefSet);
    }

    Context_->GetInterProceduralRegionCounter().CountEntity(gammaModRefSet);
    EncodeIntraProceduralRegion(subregion, subregionLiveIntervals);

    if (memoryStateExitVar.has_value())
    {
      auto & setNode = subregionLiveIntervals.createSetNodeForRegionExit(gammaModRefSet);
      memoryStateExitVar->branchResult[i]->divert_to(setNode.output(0));
    }
  }
}

void
MemoryStateEncoder::EncodeTheta(
    rvsdg::ThetaNode & thetaNode,
    RegionIntervalOutputMapping & liveIntervals)
{
  const auto & thetaModRefSet = Context_->GetModRefSummary().getThetaModRef(thetaNode);

  // Find a memory state loop variable we can use
  std::optional<rvsdg::ThetaNode::LoopVar> memoryStateLoopVar;
  const auto loopVars = thetaNode.GetLoopVars();
  for (auto & loopVar : loopVars)
  {
    if (is<MemoryStateType>(loopVar.input->Type()))
    {
      if (memoryStateLoopVar.has_value())
        throw std::logic_error("Multiple memory state loop variables found");
      memoryStateLoopVar = loopVar;
    }
  }

  if (!thetaModRefSet.isEmpty())
  {
    if (!memoryStateLoopVar.has_value())
    {
      throw std::logic_error("Theta node with ModRefSet must have a memory state loop variable");
    }

    auto & setNode =
        liveIntervals.attachNode(thetaNode, *memoryStateLoopVar->output, thetaModRefSet);
    memoryStateLoopVar->input->divert_to(setNode.output(0));
  }

  RegionIntervalOutputMapping subregionLiveIntervals(*thetaNode.subregion());
  if (memoryStateLoopVar.has_value())
  {
    subregionLiveIntervals.createRegionEntry(*memoryStateLoopVar->pre, thetaModRefSet);
  }

  Context_->GetInterProceduralRegionCounter().CountEntity(thetaModRefSet);
  EncodeIntraProceduralRegion(*thetaNode.subregion(), subregionLiveIntervals);

  if (memoryStateLoopVar.has_value())
  {
    auto & setNode = subregionLiveIntervals.createSetNodeForRegionExit(thetaModRefSet);
    memoryStateLoopVar->post->divert_to(setNode.output(0));
  }
}

bool
MemoryStateEncoder::ShouldHandle(const rvsdg::SimpleNode & simpleNode) const noexcept
{
  for (size_t n = 0; n < simpleNode.ninputs(); n++)
  {
    auto input = simpleNode.input(n);
    if (is<MemoryStateType>(input->Type()))
    {
      return true;
    }
  }

  for (size_t n = 0; n < simpleNode.noutputs(); n++)
  {
    auto output = simpleNode.output(n);
    if (is<MemoryStateType>(output->Type()))
    {
      return true;
    }
  }

  return false;
}

}
