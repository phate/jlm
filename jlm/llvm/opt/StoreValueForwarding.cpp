/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
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
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <memory>
#include <queue>

namespace jlm::llvm
{

/**
 * \brief Store Value Forwarding Statistics class
 */
class StoreValueForwarding::Statistics final : public util::Statistics
{
  static constexpr auto NumTotalLoads_ = "#TotalLoads";
  static constexpr auto NumLoadsForwarded_ = "#LoadsForwarded";
  static constexpr auto NumAAQueries_ = "#AliasAnalysisQueries";
  static constexpr auto TracingLabel_ = "TracingTime";
  static constexpr auto ForwardingLabel_ = "ForwardingTime";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Id::StoreValueForwarding, sourceFile)
  {
    AddTimer(TracingLabel_);
    AddTimer(ForwardingLabel_);
  }

  void
  StartStatistics() noexcept
  {
    AddTimer(Label::Timer).start();
  }

  void
  StopStatistics(
      size_t numTotalLoads,
      size_t numLoadsForwarded,
      size_t numAliasAnalysisQueries) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(NumTotalLoads_, numTotalLoads);
    AddMeasurement(NumLoadsForwarded_, numLoadsForwarded);
    AddMeasurement(NumAAQueries_, numAliasAnalysisQueries);
  }

  void
  startTracing() noexcept
  {
    GetTimer(TracingLabel_).start();
  }

  void
  stopTracing() noexcept
  {
    GetTimer(TracingLabel_).stop();
  }

  void
  startForwarding() noexcept
  {
    GetTimer(ForwardingLabel_).start();
  }

  void
  stopForwarding() noexcept
  {
    GetTimer(ForwardingLabel_).stop();
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
  explicit Context(Statistics & statistics) noexcept
      : outputTracer(true),
        statistics(statistics)
  {
    outputTracer.setTraceThroughStructuralNodes(true);
    outputTracer.setTraceThroughLoadedStates(true);
  }

  // Counters used for statistics
  size_t numTotalLoads = 0;
  size_t numLoadsForwarded = 0;
  size_t numAliasAnalysisQueries = 0;

  // Memoization of outputs that have been routed into regions
  struct OutputRegionHash
  {
    std::size_t
    operator()(const std::pair<rvsdg::Output *, rvsdg::Region *> & value) const
    {
      return std::hash<rvsdg::Output *>()(value.first) ^ std::hash<rvsdg::Region *>()(value.second);
    }
  };

  std::unordered_map<std::pair<rvsdg::Output *, rvsdg::Region *>, rvsdg::Output *, OutputRegionHash>
      routedOutputs{};

  OutputTracer outputTracer;

  Statistics & statistics;
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
          // Output tracing is only done intra-procedural in this pass, and we are about to process
          // a new lambda node. Clear the tracing cache to free up the memory from the last lambda
          // we processed.
          context_->outputTracer.clearCache();

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

// Enum containing the possible relationships between a load operation and a store node
enum class StoreNodeInfo
{
  ValueForwarding,  // The store can be forwarded to the load node
  ClobberNoForward, // The store may clobber the load, but the stored value can not be forwarded.
                    // This can for example be because the addresses are not MustAlias,
                    // or that the type of the stored value and the loaded value are not identical.
  NoClobber         // The store is guaranteed to not clobber the loaded value
};

// When tracing backwards from a load node through memory state edges, we store points
// at which a store writes to the load node, or a structural node causes the loaded value
// to have multiple possible origins. These points are known as StoreValueOrigins
struct StoreValueOrigin
{
  enum class Kind
  {
    Unknown,         // Tracing does not lead to known stored values in all branches
    Uninitialized,   // Tracing leads to uninitialized memory, like an alloca
    StoreNode,       // Tracing leads to exactly one store node, and it is not inside a subregion
    GammaNodeOutput, // Tracing leads to the output of a gamma, but all branches trace to stores
    ThetaNodeOutput, // Tracing leads to the output of a theta, with a store inside
    ThetaNodePre     // Tracing leads to the pre value of a loop variable
  };

  Kind kind;
  rvsdg::Node * node;

  [[nodiscard]] bool
  isKnown() const
  {
    return kind != Kind::Unknown;
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
  createUnknown()
  {
    return StoreValueOrigin{ Kind::Unknown, nullptr };
  }

  static StoreValueOrigin
  createUninitialized()
  {
    return StoreValueOrigin{ Kind::Uninitialized, nullptr };
  }

  static StoreValueOrigin
  createStoreNode(rvsdg::SimpleNode & storeNode)
  {
    return StoreValueOrigin{ Kind::StoreNode, &storeNode };
  }

  static StoreValueOrigin
  createGammaNodeOutput(rvsdg::GammaNode & gammaNode)
  {
    return StoreValueOrigin{ Kind::GammaNodeOutput, &gammaNode };
  }

  static StoreValueOrigin
  createThetaNodeOutput(rvsdg::ThetaNode & thetaNode)
  {
    return StoreValueOrigin{ Kind::ThetaNodeOutput, &thetaNode };
  }

  static StoreValueOrigin
  createThetaNodePre(rvsdg::ThetaNode & thetaNode)
  {
    return StoreValueOrigin{ Kind::ThetaNodePre, &thetaNode };
  }
};

/**
 * Helper class holding info during tracing of the memory states going to a given load node.
 * Can trace into structural nodes, and keep track of separate store nodes for separate regions.
 * If all branches lead to a store that can be forwarded, forwarding can be performed.
 */
class LoadTracingInfo
{
public:
  LoadTracingInfo(rvsdg::SimpleNode & loadNode, StoreValueForwarding::Context & context)
      : loadNode(loadNode),
        context(context)
  {
    JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));
    loadedAddress = &llvm::traceOutput(*LoadOperation::AddressInput(loadNode).origin());
    loadedType = LoadOperation::LoadedValueOutput(loadNode).Type();
    loadedTypeSize = GetTypeStoreSize(*loadedType);
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
    // Forwarding of loads with no memory states is not possible
    // TODO: We could load values from constant globals
    if (LoadOperation::numMemoryStates(loadNode) == 0)
      return false;

    // Perform tracing from each memory state input to find exactly what store it leads to
    for (auto & memoryStateInput : LoadOperation::MemoryStateInputs(loadNode))
    {
      // If the memory state input can not be traced back to store nodes,
      // or different memory state inputs lead to different store nodes, we give up
      auto lastStoredValue = getLastStoreBeforeInput(memoryStateInput);
      if (!lastStoredValue.isKnown())
        return false;
    }

    // During tracing, loop back-edges are never followed, but instead added to a list.
    // Go through the list to ensure all back-edges have been traced as well.
    while (!loopVarPostsToTrace.IsEmpty())
    {
      auto loopVarPost = *loopVarPostsToTrace.Items().begin();
      auto lastStoredValue = getLastStoreBeforeInput(*loopVarPost);
      if (!lastStoredValue.isKnown())
        return false;

      loopVarPostsToTrace.Remove(loopVarPost);
    }

    return true;
  }

private:
  /**
   * Performs a simple alias analysis query between the \ref loadNode and the given store node,
   * to determine if they read and write from the same exact address,
   * possibly interfere, or if the operations are guaranteed to be fully independent.
   * @param storeNode the StoreOperation node
   */
  aa::AliasAnalysis::AliasQueryResponse
  queryAliasAnalysis(rvsdg::SimpleNode & storeNode)
  {
    JLM_ASSERT(is<StoreOperation>(&storeNode));

    // Use the local alias analysis, but limited to a single traced origin.
    // This causes the analysis to give up as soon as a pointer has multiple possible origins,
    // or an unknown offset.
    aa::LocalAliasAnalysis localAA;
    localAA.setMaxTraceCollectionSize(1);

    context.numAliasAnalysisQueries++;

    const auto & storeAddress = *StoreOperation::AddressInput(storeNode).origin();
    const auto storeType = StoreOperation::StoredValueInput(storeNode).Type();
    const auto storedSize = GetTypeStoreSize(*storeType);

    // Query the alias analysis
    return localAA.Query(*loadedAddress, loadedTypeSize, storeAddress, storedSize);
  }

  /**
   * Attempts to trace the given memory state input back to a store node that can be forwarded.
   * If a store node that may alias the load is encountered, unknown is returned.
   * If structural nodes are encountered, all branches are traced.
   * If any of the branches are not traceable, unknown is returned.
   * If the branches lead to different store nodes, the structural node itself is returned.
   *
   * During tracing, the class keeps track of the last store before nodes and region exits.
   * If tracing different memory state chains lead to different "last store node before X",
   * store forwarding is not possible, and unknown is returned to terminate early.
   *
   * @param input the memory state input to trace from
   * @return the last node that stores to the memory loaded by the loadNode, before the input.
   */
  StoreValueOrigin
  getLastStoreBeforeInput(rvsdg::Input & input)
  {
    // If the input has already been traced, return the last result
    if (const auto it = lastStoreBeforeInput.find(&input); it != lastStoreBeforeInput.end())
      return it->second;

    auto result = getLastStoreBeforeInputInternal(input);

    // Add the result to the tracing maps
    lastStoreBeforeInput[&input] = result;

    // If the input is on a node, add the result to the node map
    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(input))
    {
      const auto [it, inserted] = lastStoreBeforeNode.emplace(node, result);

      // If the node already had a different last store value, give up
      if (!inserted && it->second != result)
        return StoreValueOrigin::createUnknown();
    }

    // If the input is a region exit, add the result to the region exit map
    if (auto regionResult = dynamic_cast<rvsdg::RegionResult *>(&input))
    {
      const auto region = regionResult->region();
      const auto [it, inserted] = lastStoreInRegion.emplace(region, result);

      // If the region already had a different last store value, give up
      if (!inserted && it->second != result)
        return StoreValueOrigin::createUnknown();
    }

    return result;
  }

  StoreValueOrigin
  getLastStoreBeforeInputInternal(rvsdg::Input & input)
  {
    auto & tracedOutput = context.outputTracer.trace(*input.origin());

    // If tracing reached a store operation, look up its info
    if (auto [storeNode, storeOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<StoreOperation>(tracedOutput);
        storeNode && storeOp)
    {
      // Lookup or create a store node info entry
      auto [it, inserted] = storeNodeInfo.emplace(storeNode, StoreNodeInfo::ClobberNoForward);

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
          if (*storedType == *loadedType)
            it->second = StoreNodeInfo::ValueForwarding;
          else
            it->second = StoreNodeInfo::ClobberNoForward;
          break;
        }
        default:
          JLM_UNREACHABLE("Unknown AliasAnalysis response");
        }
      }

      switch (it->second)
      {
      case StoreNodeInfo::ValueForwarding:
        return StoreValueOrigin::createStoreNode(*storeNode);

      case StoreNodeInfo::ClobberNoForward:
        return StoreValueOrigin::createUnknown();

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
        return StoreValueOrigin::createUnknown();

      for (auto & input : joinNode->Inputs())
      {
        auto result = getLastStoreBeforeInput(input);
        if (!result.isKnown())
          return StoreValueOrigin::createUnknown();
      }

      // If none of the calls returned nullptr, there must a shared last store before the join
      const auto sharedLastStore = lastStoreBeforeNode[joinNode];
      JLM_ASSERT(sharedLastStore.isKnown());
      return sharedLastStore;
    }

    // if tracing reaches an alloca, the value is uninitialized, so we can pick our own value
    if (auto [allocaNode, allocaOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(tracedOutput);
        allocaNode && allocaOp)
    {
      return StoreValueOrigin::createUninitialized();
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
          return StoreValueOrigin::createUnknown();

        // Keep track if there is a single shared last store in all branches
        if (!commonStoreValueOrigin.has_value())
          commonStoreValueOrigin = lastStoreNode;
        else if (commonStoreValueOrigin.value() != lastStoreNode)
          commonStoreValueOrigin = StoreValueOrigin::createUnknown();
      }

      JLM_ASSERT(commonStoreValueOrigin.has_value());
      if (commonStoreValueOrigin->isKnown())
        return *commonStoreValueOrigin;

      // If the last store node depends on the branch taken, return the gamma node itself
      return StoreValueOrigin::createGammaNodeOutput(*gammaNode);
    }

    // If we found an exit variable of a theta node, continue tracing on the inside
    if (auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(tracedOutput))
    {
      const auto loopVar = thetaNode->MapOutputLoopVar(tracedOutput);
      auto lastStoreNode = getLastStoreBeforeInput(*loopVar.post);

      if (!lastStoreNode.isKnown())
        return StoreValueOrigin::createUnknown();

      // if the last store before the end of the theta subregion is the pre of the same theta,
      // the loaded memory is loop invariant, and we can load from the last store before the theta.
      if (lastStoreNode.kind == StoreValueOrigin::Kind::ThetaNodePre
          && lastStoreNode.node == thetaNode)
      {
        return getLastStoreBeforeInput(*loopVar.input);
      }

      // if the reached store node is inside the theta, it must be routed out of it
      if (lastStoreNode.node->region() == thetaNode->subregion())
        return StoreValueOrigin::createThetaNodeOutput(*thetaNode);

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
        return StoreValueOrigin::createUnknown();

      // If the loop variables' post result is not traced, add it to the list.
      // Using a list prevents visiting the loop body multiple times during recursion.
      if (lastStoreBeforeInput.count(loopVar.post) == 0)
        loopVarPostsToTrace.insert(loopVar.post);

      return StoreValueOrigin::createThetaNodePre(*thetaNode);
    }

    // Tracing reached something that is not handled, such as a function call
    return StoreValueOrigin::createUnknown();
  }

public:
  rvsdg::SimpleNode & loadNode;
  rvsdg::Output * loadedAddress;
  std::shared_ptr<const rvsdg::Type> loadedType;
  size_t loadedTypeSize;

  StoreValueForwarding::Context & context;

  // Map containing info about each store node relevant to store value forwarding.
  std::unordered_map<rvsdg::SimpleNode *, StoreNodeInfo> storeNodeInfo;

  // The last store value origin on the memory state chain before the given input.
  std::unordered_map<rvsdg::Input *, StoreValueOrigin> lastStoreBeforeInput;
  // The last store value origin before the given node.
  std::unordered_map<rvsdg::Node *, StoreValueOrigin> lastStoreBeforeNode;
  // The last store value origin node before the end of the given region.
  // It can be outside the region if no store occurs inside the region.
  std::unordered_map<rvsdg::Region *, StoreValueOrigin> lastStoreInRegion;

  // When tracing reaches a loop var pre argument, tracing does not continue through the post.
  // The loop var post result is instead added to this set, to ensure that tracing happens later.
  // Only loop vars that have yet to be traced are added here.
  util::HashSet<rvsdg::Input *> loopVarPostsToTrace;

  // During routing, at most one exit variable need to be created per gamma
  std::unordered_map<rvsdg::GammaNode *, rvsdg::Output *> createdExitVars;
  // During routing, at most one loop variable needs to be created per theta.
  std::unordered_map<rvsdg::ThetaNode *, rvsdg::ThetaNode::LoopVar> createdLoopVars;
  // During routing, loop variable posts are not routed immediately, but added to this queue
  std::queue<rvsdg::Input *> unroutedLoopVarPosts;
};

void
StoreValueForwarding::processLoadNode(rvsdg::SimpleNode & loadNode)
{
  context_->numTotalLoads++;

  // Only non-volatile loads are candidates for being forwarded to
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));

  context_->statistics.startTracing();
  LoadTracingInfo loadTracingInfo(loadNode, *context_);
  const bool success = loadTracingInfo.traceAllMemoryStateInputs();
  context_->statistics.stopTracing();

  if (success)
  {
    context_->statistics.startForwarding();
    forwardStoredValues(loadTracingInfo);
    context_->statistics.stopForwarding();
  }
}

// Performs StoreValueForwarding to the load node represented by the tracingInfo.
void
StoreValueForwarding::forwardStoredValues(LoadTracingInfo & tracingInfo)
{
  context_->numLoadsForwarded++;

  auto & loadNode = tracingInfo.loadNode;
  auto & loadedValueOutput = LoadOperation::LoadedValueOutput(loadNode);
  auto & loadRegion = *loadNode.region();

  // There must be a node providing the last value stored before the load
  const auto lastStoreNode = tracingInfo.lastStoreBeforeNode[&loadNode];
  JLM_ASSERT(lastStoreNode.isKnown());
  auto & storedValueOutput = getStoredValueOrigin(lastStoreNode, loadRegion, tracingInfo);

  // Fixup all loop variables that were created during the above routing
  connectUnroutedLoopPosts(tracingInfo);

  // Divert users of the load to the routed stored value
  loadedValueOutput.divert_users(&storedValueOutput);

  // Make the load node dead by routing all memory state users around it
  for (auto & memoryStateOutput : LoadNonVolatileOperation::MemoryStateOutputs(loadNode))
  {
    auto & memoryStateInput =
        LoadNonVolatileOperation::MapMemoryStateOutputToInput(memoryStateOutput);
    memoryStateOutput.divert_users(memoryStateInput.origin());
  }
}

// Gets an rvsdg output providing the last value stored relative to the storeValueOrigin.
rvsdg::Output &
StoreValueForwarding::getStoredValueOrigin(
    StoreValueOrigin storeValueOrigin,
    rvsdg::Region & targetRegion,
    LoadTracingInfo & tracingInfo)
{
  JLM_ASSERT(storeValueOrigin.isKnown());

  if (storeValueOrigin.kind == StoreValueOrigin::Kind::Uninitialized)
  {
    // When forwarding uninitialized memory, create an undef node
    return *UndefValueOperation::Create(targetRegion, tracingInfo.loadedType);
  }

  if (storeValueOrigin.kind == StoreValueOrigin::Kind::StoreNode)
  {
    // For store nodes, the stored value is the origin of the node's value input
    auto & storedValue = *StoreOperation::StoredValueInput(*storeValueOrigin.node).origin();
    return routeOutputToRegion(storedValue, targetRegion);
  }

  // For gamma nodes, create an exit variable by finding the stored value in each of its regions
  if (storeValueOrigin.kind == StoreValueOrigin::Kind::GammaNodeOutput)
  {
    auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(storeValueOrigin.node);
    JLM_ASSERT(gammaNode);

    // We only need to create at most one exit variable per gamma, so memoize it
    auto [it, inserted] = tracingInfo.createdExitVars.emplace(gammaNode, nullptr);
    if (inserted)
    {
      std::vector<rvsdg::Output *> lastStorePerSubregion;
      for (auto & subregion : gammaNode->Subregions())
      {
        auto lastStoreValue = tracingInfo.lastStoreInRegion[&subregion];
        auto & storedValueOutput = getStoredValueOrigin(lastStoreValue, subregion, tracingInfo);
        lastStorePerSubregion.push_back(&storedValueOutput);
      }

      auto exitVar = gammaNode->AddExitVar(lastStorePerSubregion);
      it->second = exitVar.output;
    }
    JLM_ASSERT(it->second);
    return routeOutputToRegion(*it->second, targetRegion);
  }

  // For theta nodes, create a loop variable
  if (storeValueOrigin.kind == StoreValueOrigin::Kind::ThetaNodeOutput
      || storeValueOrigin.kind == StoreValueOrigin::Kind::ThetaNodePre)
  {
    auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(storeValueOrigin.node);
    JLM_ASSERT(thetaNode);

    // If the theta output is being requested, and no clobber happens inside the theta,
    // skip making a loop variable and return the last store before the theta instead
    if (storeValueOrigin.kind == StoreValueOrigin::Kind::ThetaNodeOutput)
    {
      auto lastStoreInRegion = tracingInfo.lastStoreInRegion.find(thetaNode->subregion());
      JLM_ASSERT(lastStoreInRegion != tracingInfo.lastStoreInRegion.end());
      JLM_ASSERT(lastStoreInRegion->second.isKnown());

      // If the last store is the theta pre, the loop body never clobbers
      if (lastStoreInRegion->second.kind == StoreValueOrigin::Kind::ThetaNodePre)
      {
        auto lastStoreBeforeTheta = tracingInfo.lastStoreBeforeNode.find(thetaNode);
        JLM_ASSERT(lastStoreBeforeTheta != tracingInfo.lastStoreBeforeNode.end());
        return getStoredValueOrigin(lastStoreBeforeTheta->second, targetRegion, tracingInfo);
      }
    }

    // If the loop variable has not yet been created in this theta, create it now
    auto loopVarSlot = tracingInfo.createdLoopVars.find(thetaNode);
    if (loopVarSlot == tracingInfo.createdLoopVars.end())
    {
      rvsdg::Output * initialValue = nullptr;

      if (auto lastStoreBeforeTheta = tracingInfo.lastStoreBeforeNode.find(thetaNode);
          lastStoreBeforeTheta != tracingInfo.lastStoreBeforeNode.end())
      {
        JLM_ASSERT(lastStoreBeforeTheta->second.isKnown());
        auto & outerRegion = *thetaNode->region();
        initialValue =
            &getStoredValueOrigin(lastStoreBeforeTheta->second, outerRegion, tracingInfo);
      }
      else
      {
        // the loop variable is only used as a theta output, so give it undef as input
        initialValue = UndefValueOperation::Create(*thetaNode->region(), tracingInfo.loadedType);
      }

      // Create the loop variable and add it to the map
      JLM_ASSERT(initialValue);
      auto loopVar = thetaNode->AddLoopVar(initialValue);
      auto [it, inserted] = tracingInfo.createdLoopVars.emplace(thetaNode, loopVar);
      JLM_ASSERT(inserted);
      loopVarSlot = it;

      // To prevent looping during routing, the created loop variable's post is added to a
      // queue of loop variable posts that are routed properly later.
      tracingInfo.unroutedLoopVarPosts.push(loopVar.post);
    }

    // Return the correct output, depending on the query kind
    switch (storeValueOrigin.kind)
    {
    case StoreValueOrigin::Kind::ThetaNodePre:
      return routeOutputToRegion(*loopVarSlot->second.pre, targetRegion);
    case StoreValueOrigin::Kind::ThetaNodeOutput:
      return routeOutputToRegion(*loopVarSlot->second.output, targetRegion);
    default:
      JLM_UNREACHABLE("Unknown StoreValueOrigin kind");
    }
  }

  JLM_UNREACHABLE("Unknown StoreValueOriginKind");
}

void
StoreValueForwarding::connectUnroutedLoopPosts(LoadTracingInfo & tracingInfo)
{
  // The process of handling all created loop variables may also create more loop variables,
  // so keep going until the queue is empty.
  while (!tracingInfo.unroutedLoopVarPosts.empty())
  {
    auto post = tracingInfo.unroutedLoopVarPosts.front();
    tracingInfo.unroutedLoopVarPosts.pop();

    auto lastStore = tracingInfo.lastStoreInRegion.find(post->region());
    JLM_ASSERT(lastStore != tracingInfo.lastStoreInRegion.end());
    auto & origin = getStoredValueOrigin(lastStore->second, *post->region(), tracingInfo);
    post->divert_to(&origin);
  }
}

rvsdg::Output &
StoreValueForwarding::routeOutputToRegion(rvsdg::Output & output, rvsdg::Region & region)
{
  if (output.region() == &region)
    return output;

  if (region.IsRootRegion())
    JLM_UNREACHABLE("root region reached during attempt at routing output into region");

  if (auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(region.node()))
  {
    // Route the output all the way to just outside the gamma first
    auto & outerOutput = routeOutputToRegion(output, *gammaNode->region());

    // If the outer output already has a corresponding EntryVar, return it
    if (auto it = context_->routedOutputs.find({ &outerOutput, &region });
        it != context_->routedOutputs.end())
      return *it->second;

    // Create an EntryVar for the output, add all branch arguments to the cache
    auto entryVar = gammaNode->AddEntryVar(&outerOutput);
    for (auto branchArgument : entryVar.branchArgument)
    {
      context_->routedOutputs[{ &outerOutput, branchArgument->region() }] = branchArgument;
    }

    return *entryVar.branchArgument[region.index()];
  }

  if (auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(region.node()))
  {
    // Route the output all the way to just outside the theta first
    auto & outerOutput = routeOutputToRegion(output, *thetaNode->region());

    // If the outer output already has a corresponding invariant loop variable, return it
    if (auto it = context_->routedOutputs.find({ &outerOutput, &region });
        it != context_->routedOutputs.end())
      return *it->second;

    // Create an invariant LoopVar for the output and add it to the cache
    auto loopVar = thetaNode->AddLoopVar(&outerOutput);
    context_->routedOutputs[{ &outerOutput, &region }] = loopVar.pre;
    return *loopVar.pre;
  }

  JLM_UNREACHABLE("routeOutputToRegion reached unhandled structural node");
}

void
StoreValueForwarding::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(module.SourceFilePath().value());
  context_ = std::make_unique<Context>(*statistics);

  statistics->StartStatistics();

  auto & rvsdg = module.Rvsdg();
  traverseInterProceduralRegion(rvsdg.GetRootRegion());

  statistics->StopStatistics(
      context_->numTotalLoads,
      context_->numLoadsForwarded,
      context_->numAliasAnalysisQueries);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  context_.reset();
}
}
