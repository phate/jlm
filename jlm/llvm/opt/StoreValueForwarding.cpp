/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/rvsdg/RegionPredicateTrace.hpp"
#include "jlm/util/Hash.hpp"
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/Andersen.hpp>
#include <jlm/llvm/opt/alias-analyses/LocalAliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraphAliasAnalysis.hpp>
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

// Makes the LocalAA try harder, by tracing along all possible paths,
// and checking if allocas provably do not escape the function.
static const bool ENABLE_AGGRESSIVE_LOCALAA = std::getenv("JLM_ENABLE_SVF_AGGRESSIVE_LOCALAA");

// Enables the use of the PointsToGraphAliasAnalysis.
// Runs Andersen to make the PointsToGraph, and queries it if LocalAA yields MayAlias.
static const bool ENABLE_PTGAA = std::getenv("JLM_ENABLE_SVF_PTGAA");

// Enables the use of region predication checking when tracing origins of loaded values
static const bool ENABLE_REGION_PREDICATE_CHECK =
    !std::getenv("JLM_DISABLE_REGION_PREDICATE_CHECK");

// By default, loads whose memory states can be traced to other loads attempt to forward
// the previously loaded value, if the types match, and the addresses are the same (MustAlias).
// When disabled, loads are skipped during tracing, and never considered for value forwarding.
static const bool DISABLE_LOAD_LOAD_FORWARDING = std::getenv("JLM_DISABLE_LOAD_LOAD_FORWARDING");

/**
 * Helper for counting alias analysis responses
 */
struct AliasQueryResponseCounter
{
  size_t numNoAliasAnalysisQueries = 0;
  size_t numMayAliasAnalysisQueries = 0;
  size_t numMustAliasAnalysisQueries = 0;

  void
  addResponse(aa::AliasAnalysis::AliasQueryResponse response)
  {
    switch (response)
    {
    case aa::AliasAnalysis::NoAlias:
      numNoAliasAnalysisQueries++;
      break;
    case aa::AliasAnalysis::MayAlias:
      numMayAliasAnalysisQueries++;
      break;
    case aa::AliasAnalysis::MustAlias:
      numMustAliasAnalysisQueries++;
      break;
    default:
      throw std::logic_error("Unhandled alias analysis query response!");
    }
  }

  void
  addFromCounter(const AliasQueryResponseCounter & other)
  {
    numNoAliasAnalysisQueries += other.numNoAliasAnalysisQueries;
    numMayAliasAnalysisQueries += other.numMayAliasAnalysisQueries;
    numMustAliasAnalysisQueries += other.numMustAliasAnalysisQueries;
  }
};

/**
 * \brief Store Value Forwarding Statistics class
 */
class StoreValueForwarding::Statistics final : public util::Statistics
{
  static constexpr auto NumLoadsWithMemoryStateLabel_ = "#LoadsWithMemoryState";
  static constexpr auto NumLoadsWithoutMemoryStateLabel_ = "#LoadsWithoutMemoryState";
  static constexpr auto NumLoadsTracedToDeltaNodeLabel_ = "#LoadsTracedToDeltaNode";
  static constexpr auto NumForwardedLoadsWithMemoryStateLabel_ = "#ForwardedLoadsWithMemoryState";
  static constexpr auto NumForwardedLoadsWithoutMemoryStateLabel_ =
      "#ForwardedLoadsWithoutMemoryState";
  static constexpr auto numNoAliasStoreLabel_ = "#NoAliasStore";
  static constexpr auto numMayAliasStoreLabel_ = "#MayAliasStore";
  static constexpr auto numMustAliasStoreLabel_ = "#MustAliasStore";
  static constexpr auto numNoAliasLoadLabel_ = "#NoAliasLoad";
  static constexpr auto numMayAliasLoadLabel_ = "#MayAliasLoad";
  static constexpr auto numMustAliasLoadLabel_ = "#MustAliasLoad";
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
      const size_t numLoadsWithMemoryState,
      const size_t numLoadsWithoutMemoryState,
      const size_t numLoadsTracedtoDeltaNode,
      const size_t numForwardedLoadsWithMemoryState,
      const size_t numForwardedLoadsWithoutMemoryState,
      const AliasQueryResponseCounter & storeAAResponses,
      const AliasQueryResponseCounter & loadAAResponses) noexcept
  {
    GetTimer(Label::Timer).stop();
    AddMeasurement(NumLoadsWithMemoryStateLabel_, numLoadsWithMemoryState);
    AddMeasurement(NumLoadsWithoutMemoryStateLabel_, numLoadsWithoutMemoryState);
    AddMeasurement(NumLoadsTracedToDeltaNodeLabel_, numLoadsTracedtoDeltaNode);
    AddMeasurement(NumForwardedLoadsWithMemoryStateLabel_, numForwardedLoadsWithMemoryState);
    AddMeasurement(NumForwardedLoadsWithoutMemoryStateLabel_, numForwardedLoadsWithoutMemoryState);
    AddMeasurement(numNoAliasStoreLabel_, storeAAResponses.numNoAliasAnalysisQueries);
    AddMeasurement(numMayAliasStoreLabel_, storeAAResponses.numMayAliasAnalysisQueries);
    AddMeasurement(numMustAliasStoreLabel_, storeAAResponses.numMustAliasAnalysisQueries);
    AddMeasurement(numNoAliasLoadLabel_, loadAAResponses.numNoAliasAnalysisQueries);
    AddMeasurement(numMayAliasLoadLabel_, loadAAResponses.numMayAliasAnalysisQueries);
    AddMeasurement(numMustAliasLoadLabel_, loadAAResponses.numMustAliasAnalysisQueries);
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
  explicit Context(aa::AliasAnalysis & aliasAnalysis, Statistics & statistics) noexcept
      : outputTracer(true),
        aliasAnalysis(aliasAnalysis),
        statistics(statistics)
  {
    outputTracer.setTraceThroughStructuralNodes(true);
    // If load/load forwarding is disabled, make the tracer skip loads
    outputTracer.setTraceThroughLoadedStates(DISABLE_LOAD_LOAD_FORWARDING);
  }

  // Counters used for statistics
  size_t numLoadsWithMemoryState = 0;
  size_t numLoadsWithoutMemoryState = 0;
  size_t numLoadsTracedToDeltaNode = 0;
  size_t numForwardedLoadsWithMemoryState = 0;
  size_t numForwardedLoadsWithoutMemoryState = 0;
  AliasQueryResponseCounter storeAAResponses;
  AliasQueryResponseCounter loadAAResponses;

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

  rvsdg::RegionPredicateTrace regionPredicateTrace;

  // The AliasAnalysis instance used for all alias queries
  aa::AliasAnalysis & aliasAnalysis;

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
            processLoad(simpleNode);
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

// Enum containing the possible relationships between a load operation and a previous load
enum class LoadNodeInfo
{
  ValueForwarding, // The load can be forwarded
  NoClobber,       // The load can not be forwarded, but it is not a clobber
};

// When tracing backwards from a load node through memory state edges, we store points
// at which a store writes to the load node, an aliasing load is performed,
// or a structural node causes the loaded value to have multiple possible origins.
// These points are known as ValueOrigins
struct ValueOrigin
{
  enum class Kind
  {
    Unknown,         // Tracing does not lead to a known value origin in all branches
    Uninitialized,   // Tracing leads to uninitialized memory, like an alloca
    LoadNode,        // Tracing leads to exactly one load node, and it is not inside a subregion
    StoreNode,       // Tracing leads to exactly one store node, and it is not inside a subregion
    GammaNodeOutput, // Tracing leads to a gamma output, but all branches trace to value origins
    ThetaNodeOutput, // Tracing leads to a theta output, with a value origin inside
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
  operator==(const ValueOrigin & other) const noexcept
  {
    return kind == other.kind && node == other.node;
  }

  [[nodiscard]] bool
  operator!=(const ValueOrigin & other) const noexcept
  {
    return !(*this == other);
  }

  static ValueOrigin
  createUnknown()
  {
    return ValueOrigin{ Kind::Unknown, nullptr };
  }

  static ValueOrigin
  createUninitialized()
  {
    return ValueOrigin{ Kind::Uninitialized, nullptr };
  }

  static ValueOrigin
  createStoreNode(rvsdg::SimpleNode & storeNode)
  {
    return ValueOrigin{ Kind::StoreNode, &storeNode };
  }

  static ValueOrigin
  createLoadNode(rvsdg::SimpleNode & loadNode)
  {
    return ValueOrigin{ Kind::LoadNode, &loadNode };
  }

  static ValueOrigin
  createGammaNodeOutput(rvsdg::GammaNode & gammaNode)
  {
    return ValueOrigin{ Kind::GammaNodeOutput, &gammaNode };
  }

  static ValueOrigin
  createThetaNodeOutput(rvsdg::ThetaNode & thetaNode)
  {
    return ValueOrigin{ Kind::ThetaNodeOutput, &thetaNode };
  }

  static ValueOrigin
  createThetaNodePre(rvsdg::ThetaNode & thetaNode)
  {
    return ValueOrigin{ Kind::ThetaNodePre, &thetaNode };
  }
};

/**
 * Helper class holding info during tracing of the memory states going to a given load node.
 * Can trace into structural nodes, and keep track of separate value origins for separate regions.
 * If all branches lead to a value origin that can be forwarded, forwarding can be performed.
 */
class LoadTracingInfo
{
public:
  LoadTracingInfo(
      rvsdg::SimpleNode & loadNode,
      OutputTracer & tracer,
      aa::AliasAnalysis & aliasAnalysis,
      rvsdg::RegionPredicateTrace & regionPredicateTrace)
      : loadNode(loadNode),
        tracer(tracer),
        aliasAnalysis(aliasAnalysis),
        regionPredicateTrace(regionPredicateTrace)
  {
    JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));
    loadedAddress = &llvm::traceOutput(*LoadOperation::AddressInput(loadNode).origin());
    loadedType = LoadOperation::LoadedValueOutput(loadNode).Type();
    loadedTypeSize = GetTypeStoreSize(*loadedType);
  }

  /**
   * Performs tracing of memory states to find value origin(s) that can be forwarded to the load.
   * If tracing leads to nodes that may or may not clobber the load,
   * or if different memory state inputs lead to different value origins,
   * value forwarding is not possible, and false is returned.
   * Different branches of structural nodes can lead to different value origins,
   * as long as all memory states lead to the same store node in each branch.
   * @return true iff all memory state inputs could be traced to the same value origin(s).
   */
  bool
  traceAllMemoryStateInputs()
  {
    JLM_ASSERT(LoadOperation::numMemoryStates(loadNode) != 0);

    // Perform tracing from each memory state input to find exactly what store it leads to
    for (auto & memoryStateInput : LoadOperation::MemoryStateInputs(loadNode))
    {
      // If the memory state input can not be traced back to store nodes,
      // or different memory state inputs lead to different store nodes, we give up
      // Since tracing starts at the load, no loop back-edges have been followed yet
      auto lastValueOrigin = getLastValueOriginBeforeInput(memoryStateInput, false);
      if (!lastValueOrigin.isKnown())
        return false;
    }

    // During tracing, loop back-edges are never followed, but instead added to a list.
    // Go through the list to ensure all back-edges have been traced as well.
    while (!loopVarPostsToTrace.IsEmpty())
    {
      auto loopVarPost = *loopVarPostsToTrace.Items().begin();
      // A loop back-edge has been followed, so pass in true
      auto lastValueOrigin = getLastValueOriginBeforeInput(*loopVarPost, true);
      if (!lastValueOrigin.isKnown())
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
  queryAliasAnalysisWithStore(rvsdg::SimpleNode & storeNode)
  {
    JLM_ASSERT(is<StoreOperation>(&storeNode));

    const auto & storeAddress = *StoreOperation::AddressInput(storeNode).origin();
    const auto storeType = StoreOperation::StoredValueInput(storeNode).Type();
    const auto storedSize = GetTypeStoreSize(*storeType);

    // Trace the store address now, to avoid duplicate work when multiple alias analyses are used
    const auto & tracedStoredAddress = llvm::traceOutput(storeAddress);

    // Query the alias analysis
    const auto response =
        aliasAnalysis.Query(*loadedAddress, loadedTypeSize, tracedStoredAddress, storedSize);
    storeAAResponses.addResponse(response);

    return response;
  }

  /**
   * Performs a simple alias analysis query between the LoadTracingInfo's \ref loadNode,
   * and the other load node given as a parameter, to determine if they read from the same address.
   * @param otherLoadNode the LoadOperation node
   */
  aa::AliasAnalysis::AliasQueryResponse
  queryAliasAnalysisWithLoad(rvsdg::SimpleNode & otherLoadNode)
  {
    JLM_ASSERT(is<LoadOperation>(&otherLoadNode));

    const auto & otherLoadAddress = *LoadOperation::AddressInput(otherLoadNode).origin();
    const auto otherLoadType = LoadOperation::LoadedValueOutput(otherLoadNode).Type();
    const auto otherLoadSize = GetTypeStoreSize(*otherLoadType);

    // Trace the store address now, to avoid duplicate work when multiple alias analyses are used
    const auto & tracedOtherLoadAddress = llvm::traceOutput(otherLoadAddress);

    // Query the alias analysis
    const auto response =
        aliasAnalysis.Query(*loadedAddress, loadedTypeSize, tracedOtherLoadAddress, otherLoadSize);
    loadAAResponses.addResponse(response);

    return response;
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
   * @param loopBackEdgeTaken true if any loop back edge has been traced on the way to this input
   * @return the last node that stores to the memory loaded by the loadNode, before the input.
   */
  ValueOrigin
  getLastValueOriginBeforeInput(rvsdg::Input & input, bool loopBackEdgeTaken)
  {
    // If the input has already been traced, return the last result
    if (const auto it = lastValueOriginBeforeInput.find({ &input, loopBackEdgeTaken });
        it != lastValueOriginBeforeInput.end())
      return it->second;

    auto result = getLastValueOriginBeforeInputInternal(input, loopBackEdgeTaken);

    // Add the result to the tracing maps
    lastValueOriginBeforeInput[{ &input, loopBackEdgeTaken }] = result;

    // If the input is on a node, add the result to the node map
    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(input))
    {
      const auto [it, inserted] =
          lastValueOriginBeforeNode.emplace(std::make_pair(node, loopBackEdgeTaken), result);

      // If the node already had a different last store value, give up
      if (!inserted && it->second != result)
        return ValueOrigin::createUnknown();
    }

    // If the input is a region exit, add the result to the region exit map
    if (auto regionResult = dynamic_cast<rvsdg::RegionResult *>(&input))
    {
      const auto region = regionResult->region();
      const auto [it, inserted] =
          lastValueOriginInRegion.emplace(std::make_pair(region, loopBackEdgeTaken), result);

      // If the region already had a different last store value, give up
      if (!inserted && it->second != result)
        return ValueOrigin::createUnknown();
    }

    return result;
  }

  ValueOrigin
  getLastValueOriginBeforeInputInternal(rvsdg::Input & input, bool loopBackEdgeTaken)
  {
    auto & tracedOutput = tracer.trace(*input.origin());

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
        const auto aliasReponse = queryAliasAnalysisWithStore(*storeNode);
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
        return ValueOrigin::createStoreNode(*storeNode);
      case StoreNodeInfo::ClobberNoForward:
        return ValueOrigin::createUnknown();
      case StoreNodeInfo::NoClobber:
      {
        // If the store is not clobbering, keep tracing along the memory state chain
        auto & memoryStateInput = StoreOperation::MapMemoryStateOutputToInput(tracedOutput);
        return getLastValueOriginBeforeInput(memoryStateInput, loopBackEdgeTaken);
      }

      default:
        JLM_UNREACHABLE("Unknown StoreNodeInfo");
      }
    }

    // If tracing reached a load operation, check if it is a perfect match
    if (auto [otherLoadNode, otherLoadOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<LoadOperation>(tracedOutput);
        otherLoadNode && otherLoadOp)
    {
      // Lookup or create a load node info entry
      auto [it, inserted] = loadNodeInfo.emplace(otherLoadNode, LoadNodeInfo::NoClobber);

      // If the load has not been encountered before, determine if forwarding is possible
      if (inserted)
      {
        const auto aliasReponse = queryAliasAnalysisWithLoad(*otherLoadNode);
        switch (aliasReponse)
        {
        case aa::AliasAnalysis::MayAlias:
        case aa::AliasAnalysis::NoAlias:
          it->second = LoadNodeInfo::NoClobber;
          break;
        case aa::AliasAnalysis::MustAlias:
        {
          // MustAlias means a forwarding candidate was found,
          // but forwarding is only possible if the type matches
          auto otherLoadedType = LoadOperation::LoadedValueOutput(*otherLoadNode).Type();
          if (*otherLoadedType == *loadedType)
            it->second = LoadNodeInfo::ValueForwarding;
          else
            it->second = LoadNodeInfo::NoClobber;
          break;
        }
        default:
          JLM_UNREACHABLE("Unknown AliasAnalysis response");
        }
      }

      switch (it->second)
      {
      case LoadNodeInfo::ValueForwarding:
        return ValueOrigin::createLoadNode(*otherLoadNode);
      case LoadNodeInfo::NoClobber:
      {
        // If the load can not be forwarded, keep tracing along the memory state chain
        auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(tracedOutput);
        return getLastValueOriginBeforeInput(memoryStateInput, loopBackEdgeTaken);
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
        return ValueOrigin::createUnknown();

      for (auto & input : joinNode->Inputs())
      {
        auto result = getLastValueOriginBeforeInput(input, loopBackEdgeTaken);
        if (!result.isKnown())
          return ValueOrigin::createUnknown();
      }

      // If none of the calls returned nullptr, there must a shared last store before the join
      const auto sharedLastStore = lastValueOriginBeforeNode[{ joinNode, loopBackEdgeTaken }];
      JLM_ASSERT(sharedLastStore.isKnown());
      return sharedLastStore;
    }

    // if tracing reaches an alloca, the value is uninitialized, so we can pick our own value
    if (auto [allocaNode, allocaOp] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(tracedOutput);
        allocaNode && allocaOp)
    {
      return ValueOrigin::createUninitialized();
    }

    // If we found an exit variable of a gamma node, trace each of its subregions
    if (auto gammaNode = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(tracedOutput))
    {
      const auto exitVar = gammaNode->MapOutputExitVar(tracedOutput);

      // If all branches lead to the same store node, return it directly.
      // If different last store nodes have been observed, this becomes nullptr
      std::optional<ValueOrigin> commonStoreValueOrigin;

      for (auto branchResult : exitVar.branchResult)
      {
        // Check if this gamma subregion was provably not taken before reaching the load node
        // We can only do this check if no back-edges have been taken
        if (ENABLE_REGION_PREDICATE_CHECK && !loopBackEdgeTaken)
        {
          auto & fromRegion = *branchResult->region();
          if(!regionPredicateTrace.CheckPredicatesSatisfiable(fromRegion, *loadNode.region()))
          {
            // Mark the region as providing uninitialized memory, since it is never reached
            lastValueOriginInRegion[{&fromRegion, loopBackEdgeTaken}] = ValueOrigin::createUninitialized();
            continue;
          }
        }

        auto lastStoreNode = getLastValueOriginBeforeInput(*branchResult, loopBackEdgeTaken);

        // If any of the gamma branches is impossible to trace back to a last store,
        // give up on forwarding entirely
        if (!lastStoreNode.isKnown())
          return ValueOrigin::createUnknown();

        // If this gamma branch leads to an uninitialized value,
        // it can be ignored when checking if all gamma branches lead to the same value.
        if (lastStoreNode.kind == ValueOrigin::Kind::Uninitialized)
          continue;

        // Keep track if there is a single shared last store in all branches
        if (!commonStoreValueOrigin.has_value())
          commonStoreValueOrigin = lastStoreNode;
        else if (commonStoreValueOrigin.value() != lastStoreNode)
          commonStoreValueOrigin = ValueOrigin::createUnknown();
      }

      if (!commonStoreValueOrigin.has_value())
        return ValueOrigin::createUninitialized();

      if (commonStoreValueOrigin->isKnown())
        return *commonStoreValueOrigin;

      // If the last store node depends on the branch taken, return the gamma node itself
      return ValueOrigin::createGammaNodeOutput(*gammaNode);
    }

    // If we found an exit variable of a theta node, continue tracing on the inside
    if (auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(tracedOutput))
    {
      const auto loopVar = thetaNode->MapOutputLoopVar(tracedOutput);
      // We continue tracing from the loop var post, but we have not taken a back-edge to get there,
      // so we keep passing the loopBackEdgeTaken parameter unmodified.
      const auto lastStoreNode = getLastValueOriginBeforeInput(*loopVar.post, loopBackEdgeTaken);

      if (!lastStoreNode.isKnown())
        return ValueOrigin::createUnknown();

      // if the last store before the end of the theta subregion is the pre of the same theta,
      // the loaded memory may be loop invariant, and tracing can continue from before the theta.
      if (lastStoreNode.kind == ValueOrigin::Kind::ThetaNodePre && lastStoreNode.node == thetaNode)
      {
        // A trace that assumes no back-edges have been taken may skip regions,
        // so unless loopBackEdgeTaken=true, we must do an additional check

        // No additional check needed
        if (loopBackEdgeTaken)
          return getLastValueOriginBeforeInput(*loopVar.input, loopBackEdgeTaken);

        // Trace again, this time with loopBackEdgeTaken=true
        const auto anyStoreInTheta = getLastValueOriginBeforeInput(*loopVar.post, true);
        if (anyStoreInTheta.kind == ValueOrigin::Kind::ThetaNodePre && anyStoreInTheta.node == thetaNode)
        {
          // The theta was determined to not affect the loaded value, so keep tracing.
          // Since the entire theta was effectively skipped, leave loopBackEdgeTaken unchanged.
          return getLastValueOriginBeforeInput(*loopVar.input, loopBackEdgeTaken);
        }
      }

      // if the reached store node is inside the theta, it must be routed out of it
      if (lastStoreNode.node->region() == thetaNode->subregion())
        return ValueOrigin::createThetaNodeOutput(*thetaNode);

      // The last store node is outside the theta, so point to it directly
      JLM_UNREACHABLE("Do we ever get here?");
      return lastStoreNode;
    }

    // If we found a loop pre variable in a theta node, trace both inside and outside
    if (auto thetaNode = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(tracedOutput))
    {
      const auto loopVar = thetaNode->MapPreLoopVar(tracedOutput);

      // Trace from the theta input first.
      // Since tracing is leaving a theta, set loopBackEdgeTaken=true
      auto inputLastStoreNode = getLastValueOriginBeforeInput(*loopVar.input, true);
      if (!inputLastStoreNode.isKnown())
        return ValueOrigin::createUnknown();

      // Since the loop value may also originte from a back-edge, add the back-edge to the list.
      // Using a list prevents visiting the loop body multiple times during recursion.
      loopVarPostsToTrace.insert(loopVar.post);

      return ValueOrigin::createThetaNodePre(*thetaNode);
    }

    // Tracing reached something that is not handled, such as a function call
    return ValueOrigin::createUnknown();
  }

public:
  rvsdg::SimpleNode & loadNode;
  rvsdg::Output * loadedAddress;
  std::shared_ptr<const rvsdg::Type> loadedType;
  size_t loadedTypeSize;

  OutputTracer & tracer;
  aa::AliasAnalysis & aliasAnalysis;
  rvsdg::RegionPredicateTrace & regionPredicateTrace;

  AliasQueryResponseCounter storeAAResponses;
  AliasQueryResponseCounter loadAAResponses;

  // Map containing info about each store node relevant to value forwarding.
  std::unordered_map<rvsdg::SimpleNode *, StoreNodeInfo> storeNodeInfo;
  // Map containing info about each load node relevant to value forwarding.
  std::unordered_map<rvsdg::SimpleNode *, LoadNodeInfo> loadNodeInfo;

  // The last value origin on the memory state chain before the given input.
  // The boolean in the key is true if any loop back-edges have been taken.
  std::unordered_map<
      std::pair<rvsdg::Input *, bool>,
      ValueOrigin,
      util::Hash<std::pair<rvsdg::Input *, bool>>>
      lastValueOriginBeforeInput;
  // The last value origin before the given node.
  // The boolean in the key is true if any loop back-edges have been taken.
  std::unordered_map<
      std::pair<rvsdg::Node *, bool>,
      ValueOrigin,
      util::Hash<std::pair<rvsdg::Node *, bool>>>
      lastValueOriginBeforeNode;
  // The last value origin node before the end of the given region.
  // It can also be outside the region if no clobber occurs inside the region.
  // The boolean in the key is true if any loop back-edges have been taken.
  std::unordered_map<
      std::pair<rvsdg::Region *, bool>,
      ValueOrigin,
      util::Hash<std::pair<rvsdg::Region *, bool>>>
      lastValueOriginInRegion;

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
StoreValueForwarding::processLoad(rvsdg::SimpleNode & loadNode)
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));

  if (LoadOperation::numMemoryStates(loadNode) == 0)
  {
    context_->numLoadsWithoutMemoryState++;
    processLoadWithoutMemoryStates(loadNode);
  }
  else
  {
    context_->numLoadsWithMemoryState++;
    processLoadWithMemoryStates(loadNode);
  }
}

void
StoreValueForwarding::processLoadWithMemoryStates(rvsdg::SimpleNode & loadNode)
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));
  JLM_ASSERT(LoadOperation::numMemoryStates(loadNode) != 0);

  context_->statistics.startTracing();
  LoadTracingInfo loadTracingInfo(
      loadNode,
      context_->outputTracer,
      context_->aliasAnalysis,
      context_->regionPredicateTrace);
  const auto shouldForwardValueOrigins = loadTracingInfo.traceAllMemoryStateInputs();
  context_->statistics.stopTracing();

  context_->storeAAResponses.addFromCounter(loadTracingInfo.storeAAResponses);
  context_->loadAAResponses.addFromCounter(loadTracingInfo.loadAAResponses);

  if (shouldForwardValueOrigins)
  {
    context_->statistics.startForwarding();
    forwardValueOrigins(loadTracingInfo);
    context_->statistics.stopForwarding();
  }
}

void
StoreValueForwarding::processLoadWithoutMemoryStates(rvsdg::SimpleNode & loadNode)
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));
  JLM_ASSERT(LoadOperation::numMemoryStates(loadNode) == 0);

  context_->statistics.startTracing();
  const auto tracedDelta = traceLoadWithoutMemoryStates(loadNode);
  context_->statistics.stopTracing();
  if (!tracedDelta.has_value())
  {
    return;
  }

  context_->statistics.startForwarding();
  forwardLoadWithoutMemoryStates(loadNode, tracedDelta.value());
  context_->statistics.stopForwarding();
}

std::optional<StoreValueForwarding::TracedDelta>
StoreValueForwarding::traceLoadWithoutMemoryStates(const rvsdg::SimpleNode & loadNode)
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));
  JLM_ASSERT(LoadOperation::numMemoryStates(loadNode) == 0);

  const auto & loadAddress = *LoadOperation::AddressInput(loadNode).origin();
  const auto [basePointer, gepConstantsOpt] = TracePointerOriginPrecise(loadAddress);
  if (!gepConstantsOpt.has_value())
  {
    return std::nullopt;
  }

  const auto deltaNode = rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(*basePointer);
  if (!deltaNode)
  {
    return std::nullopt;
  }

  context_->numLoadsTracedToDeltaNode++;
  return std::optional<TracedDelta>({ deltaNode, gepConstantsOpt.value() });
}

static size_t
getConstantDataArrayElementIndex(
    const ConstantDataArrayOperation & constantDataArray,
    const std::vector<GetElementPtrOperation::Constant> & gepConstants)
{
  if (gepConstants.empty())
    return 0;

  JLM_ASSERT(gepConstants.size() == 1);
  auto & gepConstant = gepConstants[0];
  JLM_ASSERT(gepConstant.indices.size() == 1 || gepConstant.indices.size() == 2);

  if (gepConstant.indices.size() == 1)
  {
    if (const auto gepConstantIndex = gepConstant.indices[0]; gepConstantIndex == 0)
      return 0;

    JLM_ASSERT(gepConstant.pointeeType == rvsdg::BitType::Create(8));
    const auto offsetInBytes = gepConstant.getOffsetInBytes();
    const auto elementSize = GetTypeAllocSize(constantDataArray.type());
    JLM_ASSERT(offsetInBytes % elementSize == 0);
    return offsetInBytes / elementSize;
  }

  if (gepConstant.indices.size() == 2)
  {
    JLM_ASSERT(gepConstant.indices[0] == 0);
    return gepConstant.indices.back();
  }

  throw std::logic_error("Unhandled number of GEP constant indices.");
}

void
StoreValueForwarding::forwardLoadWithoutMemoryStates(
    rvsdg::SimpleNode & loadNode,
    const TracedDelta & tracedDelta)
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));
  JLM_ASSERT(LoadOperation::numMemoryStates(loadNode) == 0);
  const auto loadOperation =
      dynamic_cast<const LoadNonVolatileOperation *>(&loadNode.GetOperation());
  const auto & deltaResultOrigin = *tracedDelta.deltaNode->result().origin();

  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(deltaResultOrigin))
  {
    rvsdg::MatchTypeWithDefault(
        node->GetOperation(),
        [&](const IntegerConstantOperation &)
        {
          JLM_ASSERT(tracedDelta.gepConstants.empty());

          auto copiedNode = node->copy(loadNode.region(), {});
          if (*loadOperation->GetLoadedType() != *node->output(0)->Type())
          {
            copiedNode =
                &TruncOperation::createNode(*copiedNode->output(0), loadOperation->GetLoadedType());
          }
          LoadOperation::LoadedValueOutput(loadNode).divert_users(copiedNode->output(0));

          context_->numForwardedLoadsWithoutMemoryState++;
        },
        [&](const ConstantFP &)
        {
          JLM_ASSERT(tracedDelta.gepConstants.empty());
          auto copiedNode = node->copy(loadNode.region(), {});
          LoadOperation::LoadedValueOutput(loadNode).divert_users(copiedNode->output(0));
          context_->numForwardedLoadsWithoutMemoryState++;
        },
        [&](const GetElementPtrOperation &)
        {
          // FIXME: handle operation
        },
        [&](const ConstantStructOperation &)
        {
          // FIXME: handle operation
        },
        [&](const ConstantDataArrayOperation & constantDataArray)
        {
          const auto elementIndex =
              getConstantDataArrayElementIndex(constantDataArray, tracedDelta.gepConstants);

          if (constantDataArray.type() == *loadOperation->GetLoadedType())
          {
            const auto elementNode =
                rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*node->input(elementIndex)->origin());
            auto copiedNode = elementNode->copy(loadNode.region(), {});
            LoadOperation::LoadedValueOutput(loadNode).divert_users(copiedNode->output(0));
          }
          else
          {
            [[maybe_unused]] auto cdaBitType =
                dynamic_cast<const rvsdg::BitType *>(&constantDataArray.type());
            [[maybe_unused]] auto loadBitType =
                dynamic_cast<const rvsdg::BitType *>(loadOperation->GetLoadedType().get());
            JLM_ASSERT(cdaBitType && loadBitType);
            JLM_ASSERT(cdaBitType->nbits() < loadBitType->nbits());
            JLM_ASSERT(loadBitType->nbits() % cdaBitType->nbits() == 0);
            const auto numRequiredConstants = loadBitType->nbits() / cdaBitType->nbits();
            JLM_ASSERT(numRequiredConstants > 1);

            std::vector<rvsdg::BitValueRepresentation> constants;
            constants.reserve(numRequiredConstants);
            for (size_t n = elementIndex; n < elementIndex + numRequiredConstants; n++)
            {
              const auto elementNode =
                  rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*node->input(n)->origin());
              const auto constantOp =
                  util::assertedCast<const IntegerConstantOperation>(&elementNode->GetOperation());
              constants.push_back(constantOp->Representation());
            }
            JLM_ASSERT(constants.size() == numRequiredConstants);
            // FIXME: take care of endianness
            auto & constantNode = IntegerConstantOperation::Create(
                *loadNode.region(),
                rvsdg::BitValueRepresentation::create(constants));
            LoadOperation::LoadedValueOutput(loadNode).divert_users(constantNode.output(0));
          }
          context_->numForwardedLoadsWithoutMemoryState++;
        },
        [&](const ConstantArrayOperation &)
        {
          // FIXME: handle operation
        },
        [&](const ConstantAggregateZeroOperation &)
        {
          const auto loadedType = loadOperation->GetLoadedType();
          if (is<PointerType>(loadedType))
          {
            const auto & nullPtrNode = ConstantPointerNullOperation::createNode(*loadNode.region());
            LoadOperation::LoadedValueOutput(loadNode).divert_users(nullPtrNode.output(0));
          }
          else if (const auto bitType = std::dynamic_pointer_cast<const rvsdg::BitType>(loadedType))
          {
            const auto & zeroNode =
                IntegerConstantOperation::Create(*loadNode.region(), bitType->nbits(), 0);
            LoadOperation::LoadedValueOutput(loadNode).divert_users(zeroNode.output(0));
          }
          else if (
              const auto floatType =
                  std::dynamic_pointer_cast<const llvm::FloatingPointType>(loadedType))
          {
            const auto zero = ConstantFP::getZeroRepresentation(floatType->size());
            const auto & zeroNode =
                ConstantFP::createNode(*loadNode.region(), floatType->size(), zero);
            LoadOperation::LoadedValueOutput(loadNode).divert_users(zeroNode.output(0));
          }
          else if (
              const auto vectorType =
                  std::dynamic_pointer_cast<const llvm::FixedVectorType>(loadedType))
          {
            // FIXME: Handle loading of vectors of zero values
            return;
          }
          else
          {
            throw std::logic_error("Unsupported load type");
          }

          context_->numForwardedLoadsWithoutMemoryState++;
        },
        [&](const ConstantPointerNullOperation &)
        {
          JLM_ASSERT(tracedDelta.gepConstants.empty());
          const auto & nullPtrNode = ConstantPointerNullOperation::createNode(*loadNode.region());
          LoadOperation::LoadedValueOutput(loadNode).divert_users(nullPtrNode.output(0));
          context_->numForwardedLoadsWithoutMemoryState++;
        },
        [&](const IntToPtrOperation &)
        {
          // FIXME: handle operation
        },
        [&]()
        {
          throw std::logic_error("Unsupported operation: " + node->DebugString());
        });
  }
  else if (
      const auto deltaNode = rvsdg::TryGetRegionParentNode<rvsdg::DeltaNode>(deltaResultOrigin))
  {
    auto [ctxInput, _] = deltaNode->MapBinderContextVar(deltaResultOrigin);
    JLM_ASSERT(ctxInput->region()->IsRootRegion());
    auto & routedValue = rvsdg::RouteToRegion(*ctxInput->origin(), *loadNode.region());
    LoadOperation::LoadedValueOutput(loadNode).divert_users(&routedValue);
    context_->numForwardedLoadsWithoutMemoryState++;
  }
  else
  {
    throw std::logic_error("Unsupported output owner");
  }
}

// Performs StoreValueForwarding to the load node represented by the tracingInfo.
void
StoreValueForwarding::forwardValueOrigins(LoadTracingInfo & tracingInfo)
{
  context_->numForwardedLoadsWithMemoryState++;

  auto & loadNode = tracingInfo.loadNode;
  auto & loadedValueOutput = LoadOperation::LoadedValueOutput(loadNode);
  auto & loadRegion = *loadNode.region();

  // There must be a node providing the last value stored before the load
  // Since we are starting from the load node, no loop back-edge has been taken
  const auto lastStoreNode = tracingInfo.lastValueOriginBeforeNode[{ &loadNode, false }];
  JLM_ASSERT(lastStoreNode.isKnown());
  auto & storedValueOutput = getValueOriginOutput(lastStoreNode, loadRegion, tracingInfo, false);

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

// Gets an rvsdg output providing the output referenced by the value origin.
rvsdg::Output &
StoreValueForwarding::getValueOriginOutput(
    ValueOrigin valueOrigin,
    rvsdg::Region & targetRegion,
    LoadTracingInfo & tracingInfo,
    bool loopBackEdgeTaken)
{
  JLM_ASSERT(valueOrigin.isKnown());

  if (valueOrigin.kind == ValueOrigin::Kind::Uninitialized)
  {
    // When forwarding uninitialized memory, create an undef node
    return *UndefValueOperation::Create(targetRegion, tracingInfo.loadedType);
  }

  if (valueOrigin.kind == ValueOrigin::Kind::StoreNode)
  {
    // For store nodes, the stored value is the origin of the node's value input
    auto & storedValue = *StoreOperation::StoredValueInput(*valueOrigin.node).origin();
    JLM_ASSERT(*storedValue.Type() == *tracingInfo.loadedType);
    return routeOutputToRegion(storedValue, targetRegion);
  }

  if (valueOrigin.kind == ValueOrigin::Kind::LoadNode)
  {
    // For load nodes, the load output is the value origin
    auto & loadedValue = LoadOperation::LoadedValueOutput(*valueOrigin.node);
    JLM_ASSERT(*loadedValue.Type() == *tracingInfo.loadedType);
    return routeOutputToRegion(loadedValue, targetRegion);
  }

  // For gamma nodes, create an exit variable by finding the stored value in each of its regions
  if (valueOrigin.kind == ValueOrigin::Kind::GammaNodeOutput)
  {
    auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(valueOrigin.node);
    JLM_ASSERT(gammaNode);

    // We only need to create at most one exit variable per gamma, so memoize it
    auto [it, inserted] = tracingInfo.createdExitVars.emplace(gammaNode, nullptr);
    if (inserted)
    {
      std::vector<rvsdg::Output *> lastStorePerSubregion;
      for (auto & subregion : gammaNode->Subregions())
      {
        auto lastValueOrigin =
            tracingInfo.lastValueOriginInRegion.find({ &subregion, loopBackEdgeTaken });
        JLM_ASSERT(lastValueOrigin != tracingInfo.lastValueOriginInRegion.end());
        auto & valueOriginOutput = getValueOriginOutput(
            lastValueOrigin->second,
            subregion,
            tracingInfo,
            loopBackEdgeTaken);
        lastStorePerSubregion.push_back(&valueOriginOutput);
      }

      auto exitVar = gammaNode->AddExitVar(lastStorePerSubregion);
      it->second = exitVar.output;
    }
    JLM_ASSERT(it->second);
    JLM_ASSERT(*it->second->Type() == *tracingInfo.loadedType);
    return routeOutputToRegion(*it->second, targetRegion);
  }

  // For theta nodes, create a loop variable
  if (valueOrigin.kind == ValueOrigin::Kind::ThetaNodeOutput
      || valueOrigin.kind == ValueOrigin::Kind::ThetaNodePre)
  {
    auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(valueOrigin.node);
    JLM_ASSERT(thetaNode);

    // If the theta output is being requested, and no clobber happens inside the theta,
    // skip making a loop variable and return the last store before the theta instead
    if (valueOrigin.kind == ValueOrigin::Kind::ThetaNodeOutput)
    {
      auto lastStoreInRegion =
          tracingInfo.lastValueOriginInRegion.find({ thetaNode->subregion(), loopBackEdgeTaken });
      JLM_ASSERT(lastStoreInRegion != tracingInfo.lastValueOriginInRegion.end());
      JLM_ASSERT(lastStoreInRegion->second.isKnown());

      // If the last store is the theta pre, the loop body never clobbers
      if (lastStoreInRegion->second.kind == ValueOrigin::Kind::ThetaNodePre)
      {
        auto lastStoreBeforeTheta =
            tracingInfo.lastValueOriginBeforeNode.find({ thetaNode, loopBackEdgeTaken });
        JLM_ASSERT(lastStoreBeforeTheta != tracingInfo.lastValueOriginBeforeNode.end());
        return getValueOriginOutput(
            lastStoreBeforeTheta->second,
            targetRegion,
            tracingInfo,
            loopBackEdgeTaken);
      }
    }

    // If the loop variable has not yet been created in this theta, create it now
    auto loopVarSlot = tracingInfo.createdLoopVars.find(thetaNode);
    if (loopVarSlot == tracingInfo.createdLoopVars.end())
    {
      rvsdg::Output * initialValue = nullptr;

      if (auto lastStoreBeforeTheta =
              tracingInfo.lastValueOriginBeforeNode.find({ thetaNode, loopBackEdgeTaken });
          lastStoreBeforeTheta != tracingInfo.lastValueOriginBeforeNode.end())
      {
        JLM_ASSERT(lastStoreBeforeTheta->second.isKnown());
        auto & outerRegion = *thetaNode->region();
        initialValue = &getValueOriginOutput(
            lastStoreBeforeTheta->second,
            outerRegion,
            tracingInfo,
            loopBackEdgeTaken);
      }
      else
      {
        // Tracing never left reached the loop entry, so the value must be defined inside the loop.
        // The created loop variable can therefore take undef as its input.
        // Since tracing never reached the top of the loop, the loop post was never added to the
        // list of back-edges, and possibly never traced with loopBackEdgeTaken=true.
        // In that case, the created loop variable will use the loopBackEdgeTaken=false version
        // of the traced ValueOrigin when routing loop post origins later.
        initialValue = UndefValueOperation::Create(*thetaNode->region(), tracingInfo.loadedType);
      }

      // Create the loop variable and add it to the map
      JLM_ASSERT(initialValue);
      JLM_ASSERT(*initialValue->Type() == *tracingInfo.loadedType);
      auto loopVar = thetaNode->AddLoopVar(initialValue);
      auto [it, inserted] = tracingInfo.createdLoopVars.emplace(thetaNode, loopVar);
      JLM_ASSERT(inserted);
      loopVarSlot = it;

      // To prevent looping during routing, the created loop variable's post is added to a
      // queue of loop variable posts that are routed properly later.
      tracingInfo.unroutedLoopVarPosts.push(loopVar.post);
    }

    // Return the correct output, depending on the query kind
    switch (valueOrigin.kind)
    {
    case ValueOrigin::Kind::ThetaNodePre:
      return routeOutputToRegion(*loopVarSlot->second.pre, targetRegion);
    case ValueOrigin::Kind::ThetaNodeOutput:
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

    // Since we are starting from a loop variable post, try using loopBackEdgeTaken=true
    auto loopBackEdgeTaken = true;
    auto lastStore =
        tracingInfo.lastValueOriginInRegion.find({ post->region(), loopBackEdgeTaken });
    if (lastStore == tracingInfo.lastValueOriginInRegion.end())
    {
      // If loopBackEdgeTaken=true does not exist, tracing never needed to take the back-edge
      loopBackEdgeTaken = false;
      lastStore = tracingInfo.lastValueOriginInRegion.find({ post->region(), loopBackEdgeTaken });
      JLM_ASSERT(lastStore != tracingInfo.lastValueOriginInRegion.end());
    }
    auto & origin =
        getValueOriginOutput(lastStore->second, *post->region(), tracingInfo, loopBackEdgeTaken);
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
    {
      // The output in the map key may have been deleted, and had its address re-used, so double
      // check
      auto & branchArgument = *it->second;
      if (gammaNode->mapBranchArgumentToInput(branchArgument).origin() == &outerOutput)
      {
        JLM_ASSERT(*branchArgument.Type() == *output.Type());
        return branchArgument;
      }
    }

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
    {
      // The output in the map key may have been deleted, and had its address re-used, so double
      // check
      auto & loopVarPre = *it->second;
      if (thetaNode->MapPreLoopVar(loopVarPre).input->origin() == &outerOutput)
      {
        JLM_ASSERT(*loopVarPre.Type() == *output.Type());
        return loopVarPre;
      }
    }

    // Create an invariant LoopVar for the output and add it to the cache
    auto loopVar = thetaNode->AddLoopVar(&outerOutput);
    context_->routedOutputs[{ &outerOutput, &region }] = loopVar.pre;
    return *loopVar.pre;
  }

  JLM_UNREACHABLE("routeOutputToRegion reached unhandled structural node");
}

static std::unique_ptr<aa::AliasAnalysis>
createAliasAnalysis(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector)
{
  auto localAA = std::make_unique<aa::LocalAliasAnalysis>();

  if (!ENABLE_AGGRESSIVE_LOCALAA)
  {
    // Setting the trace collection size to 1 limits the analysis to only the most trivial tracing
    localAA->setMaxTraceCollectionSize(1);
  }

  if (!ENABLE_PTGAA)
    return localAA;

  aa::Andersen andersen;
  auto ptg = andersen.Analyze(module, statisticsCollector);
  auto ptgAA = std::make_unique<aa::PointsToGraphAliasAnalysis>(std::move(ptg));

  return std::make_unique<aa::ChainedAliasAnalysis>(std::move(localAA), std::move(ptgAA));
}

void
StoreValueForwarding::Run(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  auto aliasAnalysis = createAliasAnalysis(module, statisticsCollector);
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  context_ = std::make_unique<Context>(*aliasAnalysis, *statistics);

  statistics->StartStatistics();

  auto & rvsdg = module.Rvsdg();
  traverseInterProceduralRegion(rvsdg.GetRootRegion());

  statistics->StopStatistics(
      context_->numLoadsWithMemoryState,
      context_->numLoadsWithoutMemoryState,
      context_->numLoadsTracedToDeltaNode,
      context_->numForwardedLoadsWithMemoryState,
      context_->numForwardedLoadsWithoutMemoryState,
      context_->storeAAResponses,
      context_->loadAAResponses);
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done
  context_.reset();
}
}
