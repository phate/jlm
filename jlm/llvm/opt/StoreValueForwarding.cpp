/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/llvm/ir/operators/MemoryStateOperations.hpp"
#include "jlm/llvm/ir/operators/operators.hpp"
#include "jlm/rvsdg/region.hpp"
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
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
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <memory>
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
   * If tracing leads to nodes that possibly clobber the load,
   * or if different memory state inputs lead to different store node(s),
   * store value forwarding is not possible, and false is returned.
   * @return true iff all memory state inputs could be traced to the same store node(s).
   */
  bool
  traceAllMemoryStateInputs()
  {
    for (auto & memoryStateInput : LoadNonVolatileOperation::MemoryStateInputs(loadNode_))
    {
      // If the memory state input can not be traced back to store nodes,
      // or different memory state inputs lead to different store nodes, we give up
      auto lastStoredValue = getLastStoreBeforeInput(memoryStateInput);
      if (!lastStoredValue.isKnown())
        return false;
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
   * @param input the memory state input to trace from
   * @return the last node that stores to the memory loaded by the loadNode, before the input.
   */
  StoreValueOrigin
  getLastStoreBeforeInput(rvsdg::Input & input)
  {
    // If the input has already been traced, return the last result
    if (const auto it = lastStoreBeforeInput_.find(&input); it != lastStoreBeforeInput_.end())
      return it->second;

    // Avoid infinite loops by taking note when tracing from a loop post variable
    if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(input))
    {
      const auto inserted = loopVarPostsStarted_.insert(&input);
      JLM_ASSERT(inserted);
    }

    auto result = getLastStoreBeforeInputInternal(input);

    // Add the result to the tracing maps
    lastStoreBeforeInput_[&input] = result;

    // If the input is on a node, add the result to the node map
    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(input))
    {
      const auto [it, existed] = lastStoreBeforeNode_.insert({ node, result });

      // If the node already had a different last store value, give up
      if (existed && it->second != result)
        return StoreValueOrigin::unknown();
    }

    // If the input is a region exit, add the result to the region exit map
    if (auto regionResult = dynamic_cast<rvsdg::RegionResult *>(&input))
    {
      const auto region = regionResult->region();
      const auto [it, existed] = lastStoreInRegion_.insert({ region, result });

      // If the region already had a different last store value, give up
      if (existed && it->second != result)
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

    // If tracing reached a store operation, perform alias analysis
    if (auto [storeNode, storeOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<StoreOperation>(tracedOutput);
        storeNode && storeOp)
    {
      const auto aliasReponse = queryAliasAnalysis(*storeNode);
      switch (aliasReponse)
      {
      case aa::AliasAnalysis::MayAlias:
        return StoreValueOrigin::unknown(); // May alias responses always disqualify
      case aa::AliasAnalysis::NoAlias:
      {
        // NoAlias means the store node can be ignored.
        // Tracing continues from the memory state input corresponding to the memory state output.
        auto & memoryStateInput = StoreOperation::MapMemoryStateOutputToInput(tracedOutput);
        return getLastStoreBeforeInput(memoryStateInput);
      }
      case aa::AliasAnalysis::MustAlias:
        // MustAlias means a store forwarding candidate was found,
        // but forwarding is only possible if the type matches
        auto storedType = StoreOperation::StoredValueInput(*storeNode).Type();
        if (*storedType != *loadedType_)
          return StoreValueOrigin::unknown();
        return StoreValueOrigin::storeNode(*storeNode);
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

      // Trace out of the theta first
      auto inputLastStoreNode = getLastStoreBeforeInput(*loopVar.input);
      if (!inputLastStoreNode.isKnown())
        return StoreValueOrigin::unknown();

      // If tracing has gone through the entire loop body back to the pre,
      // the loaded value is loop invariant, and we return the input directly.
      if (loopVarPostsStarted_.Contains(loopVar.post))
        return inputLastStoreNode;

      // Check what store value is reached from the loop post variable
      auto postLastStoreNode = getLastStoreBeforeInput(*loopVar.post);
      if (!postLastStoreNode.isKnown())
        return StoreValueOrigin::unknown();

      // If the store is invariant inside the loop, return the outside origin
      if (postLastStoreNode == inputLastStoreNode)
        return inputLastStoreNode;

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
   * Gets an output containing the last value that was stored to the memory loaded by loadNode,
   * right before / inside the given \p node.
   * The returned output is in the same region as \p node, or in a parent region.
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

    // For theta node exits, create a new loop variable
    if (storeValueOrigin.kind == StoreValueOriginKind::ThetaNodeOutput)
    {
      auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(storeValueOrigin.node);
      JLM_ASSERT(thetaNode);

      // TODO: Memoize across all loads to avoid duplicated routing

      const auto subregion = thetaNode->subregion();
      const auto lastStoreInRegion = lastStoreInRegion_[subregion];

      // The ThetaNodeOutput kind is only used when the StoreValueOrigin is in the theta subregion
      JLM_ASSERT(lastStoreInRegion.isKnown() && lastStoreInRegion.node->region() == subregion);
      auto & storedValueOutput = getStoredValueOutput(lastStoreInRegion);

      // Create an undef node for the loop variable's input. Only its output is used.
      const auto undef = UndefValueOperation::Create(*thetaNode->region(), loadedType_);

      // Create the loop variable and divert the post result to the store value inside the theta
      auto loopVar = thetaNode->AddLoopVar(undef);
      loopVar.post->divert_to(&storedValueOutput);
      return *loopVar.output;
    }

    // For theta node pre, create a new loop variable
    if (storeValueOrigin.kind == StoreValueOriginKind::ThetaNodePre)
    {
      auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(storeValueOrigin.node);
      JLM_ASSERT(thetaNode);

      // TODO: Memoize across all loads to avoid duplicated routing

      // Create a loop variable whose input is the last stored value before the theta
      const auto lastStoreBeforeTheta = lastStoreBeforeNode_[thetaNode];
      JLM_ASSERT(lastStoreBeforeTheta.isKnown());
      auto & storedValueBeforeTheta = getStoredValueOutput(lastStoreBeforeTheta);
      auto & routedInput = rvsdg::RouteToRegion(storedValueBeforeTheta, *thetaNode->region());
      auto loopVar = thetaNode->AddLoopVar(&routedInput);

      // Check if there are any new stores values inside the theta
      const auto subregion = thetaNode->subregion();
      const auto lastStoreInRegion = lastStoreInRegion_[subregion];
      JLM_ASSERT(lastStoreInRegion.isKnown());

      // If the stored value is not loop invariant, divert the loop var post
      if (lastStoreBeforeTheta != lastStoreInRegion)
      {
        // If the stored value is not loop invariant, there must be a store in the region
        JLM_ASSERT(lastStoreInRegion.node->region() == subregion);
        auto & storedValueBeforeExit = getStoredValueOutput(lastStoreInRegion);
        loopVar.post->divert_to(&storedValueBeforeExit);
      }

      return *loopVar.pre;
    }

    JLM_UNREACHABLE("Unknown StoreValueOriginKind");
  }

  rvsdg::SimpleNode & loadNode_;
  rvsdg::Output * loadedAddress_;
  std::shared_ptr<const rvsdg::Type> loadedType_;
  size_t loadedTypeSize_;

  // The last store value origin on the memory state chain before the given input.
  std::unordered_map<rvsdg::Input *, StoreValueOrigin> lastStoreBeforeInput_;
  // The last store value origin before the given node.
  std::unordered_map<rvsdg::Node *, StoreValueOrigin> lastStoreBeforeNode_;
  // The last store value origin node before the end of the given region.
  // It can be outside the region if no store occurs inside the region.
  std::unordered_map<rvsdg::Region *, StoreValueOrigin> lastStoreInRegion_;

  // When tracing starts from a loop post var, it gets added here.
  // If tracing reaches the corresponding loop pre var,
  // it can continue out through the loop input var.
  util::HashSet<rvsdg::Input *> loopVarPostsStarted_;
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
