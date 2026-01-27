/*
 * Copyright 2026 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

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

  // Any forwarded loads will have been removed
  region.prune(false);
}

void
StoreValueForwarding::processLoadNode(rvsdg::SimpleNode & loadNode)
{
  context_->numTotalLoads++;

  // Only non-volatile loads are candidates for being forwarded to
  JLM_ASSERT(is<LoadNonVolatileOperation>(&loadNode));

  // Extract info about the loaded address and loaded type
  auto & loadedAddress = llvm::traceOutput(*LoadOperation::AddressInput(loadNode).origin());
  auto loadedType = LoadOperation::LoadedValueOutput(loadNode).Type();
  const auto loadedTypeSize = GetTypeStoreSize(*loadedType);

  // Try tracing all memory state inputs to the same store node.
  // Its address must be a "MustAlias" with the loadedAddress, and the type must match loadedType
  rvsdg::SimpleNode * commonStoreNode = nullptr;

  // This function attempts to assign the commonStoreNode variable by following a memory state edge
  // It traces through regions and loads until a store is reached.
  // If tracing stops at something that is not a store, false is returned.
  // If the found store "MayAlias"es the load, forwarding is not safe, and false is returned.
  // If the found store "NoAlias"es the load, tracing continiues through the store.
  // Once a store that "MustAlias"es the load is found:
  //  - If there already is a different commonStoreNode, false is returned.
  //  - If the store has the wrong type to be forwarded to the load, false is returned.
  //  - Otherwise, the found store is the commonStoreNode, and true is returned.
  std::function<bool(rvsdg::Output &)> traceMemoryStateToCommonStore =
      [&](rvsdg::Output & memoryState) -> bool
  {
    rvsdg::Output * foundStoreOutput = traceStateEdgeToStoreNode(memoryState);
    if (!foundStoreOutput)
      return false;

    auto & foundStoreNode = rvsdg::AssertGetOwnerNode<rvsdg::SimpleNode>(*foundStoreOutput);
    JLM_ASSERT(is<StoreOperation>(&foundStoreNode));

    // If we have re-discovered the existsing commonStoreNode, return true early
    if (&foundStoreNode == commonStoreNode)
      return true;

    const auto aliasReponse = queryAliasAnalysis(loadedAddress, loadedTypeSize, foundStoreNode);
    switch (aliasReponse)
    {
    case aa::AliasAnalysis::MayAlias:
      return false; // May alias responses always disqualify from forwarding
    case aa::AliasAnalysis::NoAlias:
    {
      // NoAlias means the store node can be ignored.
      // Tracing continues from the memory state input corresponding to the memory state output.
      auto & memoryStateInput = StoreOperation::MapMemoryStateOutputToInput(*foundStoreOutput);
      return traceMemoryStateToCommonStore(*memoryStateInput.origin());
    }
    case aa::AliasAnalysis::MustAlias:
      break; // Must alias means we have found a store
    }

    // If we already have a different commonStoreNode, forwarding is not possible
    if (commonStoreNode)
      return false;

    // Only allow forwarding values with identical types
    auto storedType = StoreOperation::StoredValueInput(foundStoreNode).Type();
    if (*storedType != *loadedType)
      return false;

    // We have found a candidate for the common store node
    commonStoreNode = &foundStoreNode;
    return true;
  };

  // Trace each memory state input to find a single predecessor store common to them all
  for (auto & memoryStateInput : LoadNonVolatileOperation::MemoryStateInputs(loadNode))
  {
    // If any memory state is unable to agree on the common predecessor store, we give up
    if (!traceMemoryStateToCommonStore(*memoryStateInput.origin()))
      return;
  }

  // If the load has no state inputs, commonStoreNode will still be nullptr
  if (!commonStoreNode)
    return;

  forwardStoredValue(*commonStoreNode, loadNode);
}

rvsdg::Output *
StoreValueForwarding::traceStateEdgeToStoreNode(rvsdg::Output & state)
{
  // Allow the tracer to go through loads, as we only care about stores
  llvm::OutputTracer tracer;
  tracer.setTraceThroughStructuralNodes(true);
  tracer.setTraceThroughLoadedStates(true);
  auto & tracedOutput = tracer.trace(state);

  // Check if we traced to a store operation
  if (auto [storeNode, storeOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<StoreOperation>(tracedOutput);
      storeNode && storeOp)
  {
    return &tracedOutput;
  }

  // If we reached something else (gamma output, function call, etc.), return nullptr
  return nullptr;
}

aa::AliasAnalysis::AliasQueryResponse
StoreValueForwarding::queryAliasAnalysis(
    rvsdg::Output & loadAddress,
    size_t loadedTypeSize,
    rvsdg::SimpleNode & storeNode)
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
  return localAA.Query(loadAddress, loadedTypeSize, storeAddress, storedSize);
}

void
StoreValueForwarding::forwardStoredValue(
    rvsdg::SimpleNode & storeNode,
    rvsdg::SimpleNode & loadNode)
{
  context_->numLoadsForwarded++;

  auto & storedValueOutput = *StoreOperation::StoredValueInput(storeNode).origin();
  auto & loadedValueOutput = LoadOperation::LoadedValueOutput(loadNode);
  auto & loadRegion = *loadNode.region();

  // Route the stored value to the load's region
  // TODO: Memoize routed values to prevent tons of duplicated edges
  auto & routedStoredValue = rvsdg::RouteToRegion(storedValueOutput, loadRegion);

  loadedValueOutput.divert_users(&routedStoredValue);

  // Make the load dead by routing all memory state users around it
  for (auto & memoryStateOutput : LoadNonVolatileOperation::MemoryStateOutputs(loadNode))
  {
    auto & memoryStateInput =
        LoadNonVolatileOperation::MapMemoryStateOutputToInput(memoryStateOutput);
    memoryStateOutput.divert_users(memoryStateInput.origin());
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
