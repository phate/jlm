/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/LocalAliasAnalysis.hpp>
#include <jlm/llvm/opt/StoreValueForwarding.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/Phi.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <memory>
#include <queue>
#include <unordered_map>

namespace jlm::llvm
{

/**
 * \brief Store Value Forwarding Statistics class
 */
class StoreValueForwarding::Statistics final : public util::Statistics
{
  static constexpr auto NumTotalLoads_ = "#TotalLoads";
  static constexpr auto NumLoadsForwarded_ = "#LoadsForwarded";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::StoreValueForwarding, sourceFile)
  {}

  void
  StartStatistics() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  StopStatistics(size_t numTotalLoads, size_t numLoadsForwarded) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(NumTotalLoads_, numTotalLoads);
    AddMeasurement(NumLoadsForwarded_, numLoadsForwarded);
  }

  static std::unique_ptr<Statistics>
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

/**
 * Class keeping track of the internal state during Store Value Forwarding.
 */
struct StoreValueForwarding::Context final
{
  Context() = default;

  // Counters used for statistics
  size_t numTotalLoads = 0;
  size_t numLoadsForwarded = 0;
};

StoreValueForwarding::~StoreValueForwarding() noexcept = default;

StoreValueForwarding::StoreValueForwarding()
    : Transformation("StoreValueForwarding")
{}

void
StoreValueForwarding::traverseInterProceduralRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&](rvsdg::PhiNode & phiNode)
        {
          traverseInterProceduralRegion(*phiNode.subregion());
        },
        [&](rvsdg::LambdaNode & lambdaNode)
        {
          traverseIntraProceduralRegion(*lambdaNode.subregion());
        },
        [&]([[maybe_unused]] rvsdg::DeltaNode & deltaNode)
        {
          // Do nothing about delta nodes
        });
  }
}

void
StoreValueForwarding::traverseIntraProceduralRegion(rvsdg::Region & region)
{
  rvsdg::TopDownTraverser traverser(&region);
  for (auto node : traverser)
  {
    rvsdg::MatchTypeOrFail(
        *node,
        [&](rvsdg::GammaNode & gammaNode)
        {
          for (auto & subregion : gammaNode.Subregions())
          {
            traverseIntraProceduralRegion(subregion);
          }
        },
        [&](rvsdg::ThetaNode & thetaNode)
        {
          traverseIntraProceduralRegion(*thetaNode.subregion());
        },
        [&](rvsdg::SimpleNode & simpleNode)
        {
          if (is<LoadNonVolatileOperation>(&simpleNode))
          {
            processLoadNode(simpleNode);
          }

          // For other node types, we don't need to do anything for store value forwarding
        });
  }

  // Any forwarded loads are dead at this point, so remove them
  region.prune(false);
}

/**
 * Helper class for tracing state edges back to the store node(s) relevant to a given load node.
 * Can trace into structural nodes, and keep track of separate store nodes for separate regions.
 * If all branches lead to a store that can be forwarded, forwarding can be performed.
 */
class StoreTracer
{
  enum class StoreNodeInfo
  {
    ValueForwarding,  // The store can be forwarded to the \ref loadNode_
    ClobberNoForward, // The store may clobber the loaded value, but can not be forwarded
    NoClobber         // The store is guaranteed to not clobber the loaded value
  };

  // When tracing backwards in time through memory state edges, we store the first point
  // at which a store writes to the loadNode, or a structural node causes the loaded value
  // to have multiple possible origins.
  enum class StoreValueOriginKind
  {
    Unknown,         // Tracing does not lead to known stored values in all branches
    StoreNode,       // Tracing leads to exactly one store node, and it is not inside a subregion
    GammaNodeOutput, // Tracing leads to the output of a gamma, but all branches trace to stores
    ThetaNodeOutput, // Tracing leads to the output of a theta, with a store inside
    ThetaNodePre     // Tracing leads to the pre value of a loop variable
  };

  // Struct representing the last change in stored value before a given point along a memory state
  struct StoreValueOrigin
  {
    StoreValueOriginKind kind;
    rvsdg::Node * node;

    [[nodiscard]] bool
    isKnown() const
    {
      return kind != StoreValueOriginKind::Unknown;
    }

    [[nodiscard]] bool
    operator==(const StoreValueOrigin & other) const noexcept
    {
      return kind == other.kind && node == other.node;
    }

    [[nodiscard]] bool
    operator!=(const StoreValueOrigin & other) const noexcept
    {
      return !(*this == other);
    }

    static StoreValueOrigin
    unknown()
    {
      return StoreValueOrigin{ StoreValueOriginKind::Unknown, nullptr };
    }

    static StoreValueOrigin
    storeNode(rvsdg::SimpleNode & storeNode)
    {
      return StoreValueOrigin{ StoreValueOriginKind::StoreNode, &storeNode };
    }

    static StoreValueOrigin
    gammaNodeOutput(rvsdg::GammaNode & gammaNode)
    {
      return StoreValueOrigin{ StoreValueOriginKind::GammaNodeOutput, &gammaNode };
    }

    static StoreValueOrigin
    thetaNodeOutput(rvsdg::ThetaNode & thetaNode)
    {
      return StoreValueOrigin{ StoreValueOriginKind::ThetaNodeOutput, &thetaNode };
    }

    static StoreValueOrigin
    thetaNodePre(rvsdg::ThetaNode & thetaNode)
    {
      return StoreValueOrigin{ StoreValueOriginKind::ThetaNodePre, &thetaNode };
    }
  };

public:
  StoreTracer(rvsdg::SimpleNode & loadNode)
      : loadNode_(loadNode)
  {
    loadedAddress_ = &llvm::traceOutput(*LoadOperation::AddressInput(loadNode).origin());
    loadedType_ = LoadOperation::LoadedValueOutput(loadNode).Type();
    loadedTypeSize_ = GetTypeStoreSize(*loadedType_);
  }

  /**
   * Performs tracing of memory states to find store node(s) that can be forwarded to the load.
   * If tracing leads to nodes that may or may not clobber the load,
   * or if different memory state inputs lead to different store node(s),
   * store value forwarding is not possible, and false is returned.
   * Different branches of structural nodes can lead to different store nodes,
   * as long as all memory states lead to the same store node in each branch.
   * @return true iff all memory state inputs could be traced to the same store node(s).
   */
  bool
  traceAllMemoryStateInputs()
  {
    // First find out what region and nodes may clobber, and which store nodes can be forwarded
    const auto clobbersMarked = markAllClobbers();
    // If the marking phase found clobbering nodes that can not be forwarded, give up early
    if (!clobbersMarked)
      return false;

    bool anyMemoryStates = false;

    // Perform tracing from each memory state input to find exactly what store it leads to
    for (auto & memoryStateInput : LoadNonVolatileOperation::MemoryStateInputs(loadNode_))
    {
      anyMemoryStates = true;

      // If the memory state input can not be traced back to store nodes,
      // or different memory state inputs lead to different store nodes, we give up
      auto lastStoredValue = getLastStoreBeforeInput(memoryStateInput);
      if (!lastStoredValue.isKnown())
        return false;
    }

    // Not possible to to forwarding if there are no memory states
    if (!anyMemoryStates)
      return false;

    // During tracing, loop back-edges are never followed, but instead added to a list.
    // Go through the list to ensure all back-edges have been traced as well.
    while (!loopVarPostsToTrace_.IsEmpty())
    {
      auto loopVarPost = *loopVarPostsToTrace_.Items().begin();
      auto lastStoredValue = getLastStoreBeforeInput(*loopVarPost);
      if (!lastStoredValue.isKnown())
        return false;

      loopVarPostsToTrace_.Remove(loopVarPost);
    }

    return true;
  }

  /**
   * Performs Store Value Forwarding to the loadNode.
   * @pre traceAllMemoryStateInputs() returned true.
   */
  void
  forwardStoredValues()
  {
    auto & loadedValueOutput = LoadOperation::LoadedValueOutput(loadNode_);
    auto & loadRegion = *loadNode_.region();

    // There must be a node providing the last value stored before the load
    const auto lastStoreNode = lastStoreBeforeNode_[&loadNode_];
    JLM_ASSERT(lastStoreNode.isKnown());
    auto & storedValueOutput = getStoredValueOutput(lastStoreNode);

    // Route the stored value to the load's region
    // TODO: Memoize routed values to prevent duplicated edges
    auto & routedStoredValue = rvsdg::RouteToRegion(storedValueOutput, loadRegion);

    // Divert users of the load to the routed stored value
    loadedValueOutput.divert_users(&routedStoredValue);

    // Make the load node dead by routing all memory state users around it
    for (auto & memoryStateOutput : LoadNonVolatileOperation::MemoryStateOutputs(loadNode_))
    {
      auto & memoryStateInput =
          LoadNonVolatileOperation::MapMemoryStateOutputToInput(memoryStateOutput);
      memoryStateOutput.divert_users(memoryStateInput.origin());
    }
  }

private:
  /**
   * Traces from all memory state inputs and checks if the reached nodes
   * may clobber the value loaded by the \ref loadNode_, and if they do,
   * if it is possible to perform store value forwarding.
   */
  bool
  markAllClobbers()
  {
    util::HashSet<rvsdg::Output *> seen;
    std::queue<rvsdg::Output *> queue;

    llvm::OutputTracer tracer;
    tracer.setTraceThroughStructuralNodes(true);
    tracer.setTraceThroughLoadedStates(true);

    const auto enqueue = [&](rvsdg::Output & output)
    {
      auto & tracedOutput = tracer.trace(output);
      if (seen.insert(&tracedOutput))
        queue.push(&tracedOutput);
    };

    // Start by enqueing all memory state inputs
    for (auto & memoryStateInput : LoadNonVolatileOperation::MemoryStateInputs(loadNode_))
    {
      enqueue(*memoryStateInput.origin());
    }

    // Handle all outputs reachable by following memory state edges
    while (!queue.empty())
    {
      rvsdg::Output & output = *queue.front();
      queue.pop();

      // If this is a store node, check if it clobbers
      if (auto [storeNode, storeOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<StoreOperation>(output);
          storeNode && storeOp)
      {
        // Lookup or create a store node info entry
        auto [it, inserted] = storeNodeInfo_.emplace(storeNode, StoreNodeInfo::ClobberNoForward);

        // If the store has not been encountered before, determine forwarding / clobbering
        if (inserted)
        {
          const auto aliasReponse = queryAliasAnalysis(*storeNode);
          switch (aliasReponse)
          {
          case aa::AliasAnalysis::MayAlias:
            it->second = StoreNodeInfo::ClobberNoForward;
            break;
          case aa::AliasAnalysis::NoAlias:
            it->second = StoreNodeInfo::NoClobber;
            break;
          case aa::AliasAnalysis::MustAlias:
          {
            // MustAlias means a store forwarding candidate was found,
            // but forwarding is only possible if the type matches
            auto storedType = StoreOperation::StoredValueInput(*storeNode).Type();
            if (*storedType == *loadedType_)
              it->second = StoreNodeInfo::ValueForwarding;
            else
              it->second = StoreNodeInfo::ClobberNoForward;
            break;
          }
          default:
            JLM_UNREACHABLE("Unknown AliasAnalysis response");
          }

          // If the store may clobber, mark the region
          if (it->second != StoreNodeInfo::NoClobber)
            markRegionAsClobbering(storeNode->region());

          // For the time being we do not handle any clobbering stores, so give up early
          if (it->second == StoreNodeInfo::ClobberNoForward)
          {
            return false;
          }
        }

        // If the store was a not a clobber, continue tracing on the other side
        if (it->second == StoreNodeInfo::NoClobber)
        {
          auto & memoryStateInput = StoreOperation::MapMemoryStateOutputToInput(output);
          enqueue(*memoryStateInput.origin());
        }
        continue;
      }

      if (auto [joinNode, joinOp] =
              rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(output);
          joinNode && joinOp)
      {
        for (auto & input : joinNode->Inputs())
          enqueue(*input.origin());
        continue;
      }

      if (auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(output))
      {
        // output is the exit variable of a gamma node. Trace all its branch results.
        auto exitVar = gammaNode->MapOutputExitVar(output);
        for (auto branchResult : exitVar.branchResult)
          enqueue(*branchResult->origin());
        continue;
      }

      if (auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
      {
        // output is the output of a loop variable. Keep tracing into the region.
        auto loopVar = thetaNode->MapOutputLoopVar(output);
        enqueue(*loopVar.post->origin());
        continue;
      }

      if (auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
      {
        // output is the pre of a loop variable, keep tracing from the post and input
        auto loopVar = thetaNode->MapPreLoopVar(output);
        enqueue(*loopVar.input->origin());
        enqueue(*loopVar.post->origin());
        continue;
      }

      // The output belongs to an unhandled node type, such as a function call.
      // This disqualifies the load from store value forwarding.
      markRegionAsClobbering(output.region());
      return false;
    }

    return true;
  }

  /**
   * Marks the given region as containing operations that clobber the \ref loadNode.
   * Also marks all parent regions.
   * @param region the region to mark
   */
  void
  markRegionAsClobbering(rvsdg::Region * region)
  {
    if (clobberingRegions_.insert(region))
    {
      if (!region->IsRootRegion())
        markRegionAsClobbering(region->node()->region());
    }
  }

  /**
   * Attempts to trace the given memory state input back to a store node that can be forwarded.
   * If a store node that "MayAlias" the load is encountered, nullptr is returned.
   * If structural nodes are encountered, all branches are traced.
   * If any of the branches are not traceable, nullptr is returned.
   * If the branches lead to different store nodes, the structural node itself is returned.
   *
   * During tracing, the class keeps track of the last store before nodes and region exits.
   * If tracing different memory state chains lead to different "last store node before X",
   * store forwarding is not possible, and nullptr is returned to terminate early.
   *
   * @pre markNonInvariantRegion() must have been called.
   *
   * @param input the memory state input to trace from
   * @return the last node that stores to the memory loaded by the loadNode, before the input.
   */
  StoreValueOrigin
  getLastStoreBeforeInput(rvsdg::Input & input)
  {
    // If the input has already been traced, return the last result
    if (const auto it = lastStoreBeforeInput_.find(&input); it != lastStoreBeforeInput_.end())
      return it->second;

    auto result = getLastStoreBeforeInputInternal(input);

    // Add the result to the tracing maps
    lastStoreBeforeInput_[&input] = result;

    // If the input is on a node, add the result to the node map
    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(input))
    {
      const auto [it, inserted] = lastStoreBeforeNode_.emplace(node, result);

      // If the node already had a different last store value, give up
      if (!inserted && it->second != result)
        return StoreValueOrigin::unknown();
    }

    // If the input is a region exit, add the result to the region exit map
    if (auto regionResult = dynamic_cast<rvsdg::RegionResult *>(&input))
    {
      const auto region = regionResult->region();
      const auto [it, inserted] = lastStoreInRegion_.emplace(region, result);

      // If the region already had a different last store value, give up
      if (!inserted && it->second != result)
        return StoreValueOrigin::unknown();
    }

    return result;
  }

  StoreValueOrigin
  getLastStoreBeforeInputInternal(rvsdg::Input & input)
  {
    // Create a tracer, allow it to go through loads, as we only care about stores
    llvm::OutputTracer tracer;
    tracer.setTraceThroughStructuralNodes(true);
    tracer.setTraceThroughLoadedStates(true);
    auto & tracedOutput = tracer.trace(*input.origin());

    // If tracing reached a store operation, look up its info
    if (auto [storeNode, storeOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<StoreOperation>(tracedOutput);
        storeNode && storeOp)
    {
      // The store node we reached must already have been seen during markAllClobbers
      JLM_ASSERT(storeNodeInfo_.count(storeNode));

      const auto info = storeNodeInfo_[storeNode];
      switch (info)
      {
      case StoreNodeInfo::ValueForwarding:
        return StoreValueOrigin::storeNode(*storeNode);

      case StoreNodeInfo::ClobberNoForward:
        return StoreValueOrigin::unknown();

      case StoreNodeInfo::NoClobber:
      {
        // If the store is not clobbering, keep tracing along the memory state chain
        auto & memoryStateInput = StoreOperation::MapMemoryStateOutputToInput(tracedOutput);
        return getLastStoreBeforeInput(memoryStateInput);
      }

      default:
        JLM_UNREACHABLE("Unknown StoreNodeInfo");
      }
    }

    // For join operations, all the inputs must lead to the same last store
    if (auto [joinNode, joinOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateJoinOperation>(tracedOutput);
        joinNode && joinOp)
    {
      if (joinNode->ninputs() == 0)
        return StoreValueOrigin::unknown();

      for (auto & input : joinNode->Inputs())
      {
        auto result = getLastStoreBeforeInput(input);
        if (!result.isKnown())
          return StoreValueOrigin::unknown();
      }

      // If none of the calls returned nullptr, there must a shared last store before the join
      const auto sharedLastStore = lastStoreBeforeNode_[joinNode];
      JLM_ASSERT(sharedLastStore.isKnown());
      return sharedLastStore;
    }

    // If we found an exit variable of a gamma node, trace each of its subregions
    if (auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(tracedOutput))
    {
      const auto exitVar = gammaNode->MapOutputExitVar(tracedOutput);

      // If all branches lead to the same store node, return it directly.
      // If different last store nodes have been observed, this becomes nullptr
      std::optional<StoreValueOrigin> commonStoreValueOrigin;

      for (auto branchResult : exitVar.branchResult)
      {
        auto lastStoreNode = getLastStoreBeforeInput(*branchResult);

        // If any of the gamma branches is impossible to trace back to a last store,
        // give up on forwarding entirely
        if (!lastStoreNode.isKnown())
          return StoreValueOrigin::unknown();

        // Keep track if there is a single shared last store in all branches
        if (!commonStoreValueOrigin.has_value())
          commonStoreValueOrigin = lastStoreNode;
        else if (commonStoreValueOrigin.value() != lastStoreNode)
          commonStoreValueOrigin = StoreValueOrigin::unknown();
      }

      JLM_ASSERT(commonStoreValueOrigin.has_value());
      if (commonStoreValueOrigin->isKnown())
        return *commonStoreValueOrigin;

      // If the last store node depends on the branch taken, return the gamma node itself
      return StoreValueOrigin::gammaNodeOutput(*gammaNode);
    }

    // If we found an exit variable of a theta node, continue tracing on the inside
    if (auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(tracedOutput))
    {
      const auto loopVar = thetaNode->MapOutputLoopVar(tracedOutput);
      auto lastStoreNode = getLastStoreBeforeInput(*loopVar.post);

      if (!lastStoreNode.isKnown())
        return StoreValueOrigin::unknown();

      // if the reached store node is inside the theta, it must be routed out of it
      if (lastStoreNode.node->region() == thetaNode->subregion())
        return StoreValueOrigin::thetaNodeOutput(*thetaNode);

      // The last store node is outside the theta, so point to it directly
      return lastStoreNode;
    }

    // If we found a loop pre variable in a theta node, trace both inside and outside
    if (auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(tracedOutput))
    {
      const auto loopVar = thetaNode->MapPreLoopVar(tracedOutput);

      // Trace from the theta input first
      auto inputLastStoreNode = getLastStoreBeforeInput(*loopVar.input);
      if (!inputLastStoreNode.isKnown())
        return StoreValueOrigin::unknown();

      // Check if the loop body may clobber the loaded value
      bool mayClobber = clobberingRegions_.Contains(thetaNode->subregion());

      // If the loop body doesn't clobber, just return the store value of the input
      if (!mayClobber)
        return inputLastStoreNode;

      // If the loop variables' post result is not traced, add it to the list.
      if (lastStoreBeforeInput_.count(loopVar.post) == 0)
        loopVarPostsToTrace_.insert(loopVar.post);

      return StoreValueOrigin::thetaNodePre(*thetaNode);
    }

    // Tracing reached something that is not handled, such as a function call
    return StoreValueOrigin::unknown();
  }

  /**
   * Performs a simple alias analysis query between the class' loadNode and the given store node,
   * to determine if they read and write from the same exact address,
   * possibly interfere, or if the operations are guaranteed to be fully independent.
   * @param storeNode the StoreOperation node
   */
  aa::AliasAnalysis::AliasQueryResponse
  queryAliasAnalysis(rvsdg::SimpleNode & storeNode)
  {
    // Use the local alias analysis, but limited to a single traced origin.
    // This causes the analysis to give up as soon as a pointer has multiple possible origins,
    // or an unknown offset.
    aa::LocalAliasAnalysis localAA;
    localAA.setMaxTraceCollectionSize(1);

    const auto & storeAddress = *StoreOperation::AddressInput(storeNode).origin();
    const auto storeType = StoreOperation::StoredValueInput(storeNode).Type();
    const auto storedSize = GetTypeStoreSize(*storeType);

    // Query the alias analysis
    return localAA.Query(*loadedAddress_, loadedTypeSize_, storeAddress, storedSize);
  }

  /**
   * Gets an output containing the last value that was stored to \ref loadNode_'s memory,
   * for a given storeValueOrigin. This operation may involve routing values into structural nodes.
   */
  rvsdg::Output &
  getStoredValueOutput(StoreValueOrigin storeValueOrigin)
  {
    JLM_ASSERT(storeValueOrigin.isKnown());

    if (storeValueOrigin.kind == StoreValueOriginKind::StoreNode)
    {
      // For store nodes, the stored value is the origin of the node's value input
      return *StoreOperation::StoredValueInput(*storeValueOrigin.node).origin();
    }

    // For gamma nodes, create an exit variable by finding the stored value in each of its regions
    if (storeValueOrigin.kind == StoreValueOriginKind::GammaNodeOutput)
    {
      auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(storeValueOrigin.node);
      JLM_ASSERT(gammaNode);

      // TODO: Memoize across all loads to avoid duplicated routing

      std::vector<rvsdg::Output *> lastStorePerSubregion;

      for (auto & subregion : gammaNode->Subregions())
      {
        auto lastStoreValue = lastStoreInRegion_[&subregion];
        auto & storedValueOutput = getStoredValueOutput(lastStoreValue);

        // TODO: Memoize routing
        auto & routedStoredValue = rvsdg::RouteToRegion(storedValueOutput, subregion);
        lastStorePerSubregion.push_back(&routedStoredValue);
      }

      auto exitVar = gammaNode->AddExitVar(lastStorePerSubregion);
      return *exitVar.output;
    }

    // For theta nodes, create a loop variable
    if (storeValueOrigin.kind == StoreValueOriginKind::ThetaNodeOutput
        || storeValueOrigin.kind == StoreValueOriginKind::ThetaNodePre)
    {
      auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(storeValueOrigin.node);
      JLM_ASSERT(thetaNode);

      // If the loop variable has not yet been created in this theta, create it now
      auto loopVarSlot = createdLoopVars_.find(thetaNode);
      if (loopVarSlot == createdLoopVars_.end())
      {
        // We must create the loop variable before continuing, so give it an Undef input initially
        auto undef = UndefValueOperation::Create(*thetaNode->region(), loadedType_);
        auto loopVar = thetaNode->AddLoopVar(undef);
        auto [it, inserted] = createdLoopVars_.emplace(thetaNode, loopVar);
        JLM_ASSERT(inserted);
        loopVarSlot = it;

        // If during tracing, the pre argument of memory state loop variable was reached,
        // we must route in the last value stored before the start of the loop.
        auto lastStoreBeforeTheta = lastStoreBeforeNode_.find(thetaNode);
        if (lastStoreBeforeTheta != lastStoreBeforeNode_.end())
        {
          JLM_ASSERT(lastStoreBeforeTheta->second.isKnown());
          auto & lastStore = getStoredValueOutput(lastStoreBeforeTheta->second);
          loopVar.input->divert_to(&rvsdg::RouteToRegion(lastStore, *thetaNode->region()));
        }

        // Divert the loop variable post origin to the last store inside the loop
        const auto subregion = thetaNode->subregion();
        const auto lastStoreInRegion = lastStoreInRegion_[subregion];
        JLM_ASSERT(lastStoreInRegion.isKnown() && lastStoreInRegion.node->region() == subregion);
        auto & storedValueOutput = getStoredValueOutput(lastStoreInRegion);
        loopVar.post->divert_to(&storedValueOutput);
      }

      // Return the correct output, depending on the query kind
      switch (storeValueOrigin.kind)
      {
      case StoreValueOriginKind::ThetaNodePre:
        return *loopVarSlot->second.pre;
      case StoreValueOriginKind::ThetaNodeOutput:
        return *loopVarSlot->second.output;
      default:
        JLM_UNREACHABLE("Unknown StoreValueOrigin kind");
      }
    }

    JLM_UNREACHABLE("Unknown StoreValueOriginKind");
  }

  rvsdg::SimpleNode & loadNode_;
  rvsdg::Output * loadedAddress_;
  std::shared_ptr<const rvsdg::Type> loadedType_;
  size_t loadedTypeSize_;

  // Map containing info about each store node relevant to store value forwarding.
  // The map is filled during the \ref markAllClobbers() phase.
  std::unordered_map<rvsdg::SimpleNode *, StoreNodeInfo> storeNodeInfo_;
  // If it is possible to trace a memory state from the \ref loadNode_ back to an
  // operation that may clobber the load, the region containing the clobber is marked,
  // and all its ancestor regions are marked as clobbering.
  util::HashSet<rvsdg::Region *> clobberingRegions_;

  // The last store value origin on the memory state chain before the given input.
  std::unordered_map<rvsdg::Input *, StoreValueOrigin> lastStoreBeforeInput_;
  // The last store value origin before the given node.
  std::unordered_map<rvsdg::Node *, StoreValueOrigin> lastStoreBeforeNode_;
  // The last store value origin node before the end of the given region.
  // It can be outside the region if no store occurs inside the region.
  std::unordered_map<rvsdg::Region *, StoreValueOrigin> lastStoreInRegion_;

  // When tracing reaches a loop var pre argument, tracing does not continue through the post.
  // The loop var post result is instead added to this set, to ensure that tracing happens later.
  // Only loop vars that have yet to be traced are added here.
  util::HashSet<rvsdg::Input *> loopVarPostsToTrace_;

  // During routing, at most one loop variable needs to be created per theta.
  std::unordered_map<rvsdg::ThetaNode *, rvsdg::ThetaNode::LoopVar> createdLoopVars_;
};

void
StoreValueForwarding::processLoadNode(rvsdg::SimpleNode & loadNode)
{
  context_->numTotalLoads++;

  // Only non-volatile loads are candidates for being forwarded to
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));

  StoreTracer storeTracer(loadNode);
  const bool success = storeTracer.traceAllMemoryStateInputs();

  if (success)
  {
    context_->numLoadsForwarded++;
    storeTracer.forwardStoredValues();
  }
}

void
StoreValueForwarding::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  context_ = std::make_unique<Context>();

  auto statistics = Statistics::Create(module.SourceFilePath().value());
  statistics->StartStatistics();

  auto & rvsdg = module.Rvsdg();
  traverseInterProceduralRegion(rvsdg.GetRootRegion());

  statistics->StopStatistics(context_->numTotalLoads, context_->numLoadsForwarded);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  context_.reset();
}
}
