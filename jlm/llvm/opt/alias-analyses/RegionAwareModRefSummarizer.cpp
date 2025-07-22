/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/TarjanScc.hpp>

namespace jlm::llvm::aa
{

/** \brief Region-aware mod/ref summarizer statistics
 *
 * The statistics collected when running the region-aware mod/ref summarizer.
 *
 * @see RegionAwareModRefSummarizer
 */
class RegionAwareModRefSummarizer::Statistics final : public util::Statistics
{
  const char * NumRvsdgRegionsLabel_ = "#RvsdgRegions";
  const char * NumCallGraphSccs_ = "#CallGraphSccs";

  const char * CallGraphTimerLabel_ = "CallGraphTimer";
  const char * AnnotationTimerLabel_ = "AnnotationTimer";
  const char * PropagateTimerLabel_ = "PropagateTimer";

public:
  ~Statistics() override = default;

  explicit Statistics(
      const util::StatisticsCollector & statisticsCollector,
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph)
      : util::Statistics(Id::RegionAwareModRefSummarizer, rvsdgModule.SourceFilePath().value()),
        StatisticsCollector_(statisticsCollector)
  {
    if (!IsDemanded())
      return;

    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&rvsdgModule.Rvsdg().GetRootRegion()));
    AddMeasurement(
        NumRvsdgRegionsLabel_,
        rvsdg::Region::NumRegions(rvsdgModule.Rvsdg().GetRootRegion()));
    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.NumMemoryNodes());
  }

  void
  StartCallGraphStatistics()
  {
    if (!IsDemanded())
      return;

    AddTimer(CallGraphTimerLabel_).start();
  }

  void
  StopCallGraphStatistics(size_t numSccs)
  {
    if (!IsDemanded())
      return;

    GetTimer(CallGraphTimerLabel_).stop();
    AddMeasurement(NumCallGraphSccs_, numSccs);
  }

  void
  StartAnnotationStatistics()
  {
    if (!IsDemanded())
      return;

    AddTimer(AnnotationTimerLabel_).start();
  }

  void
  StopAnnotationStatistics()
  {
    if (!IsDemanded())
      return;

    GetTimer(AnnotationTimerLabel_).stop();
  }

  void
  StartPropagationStatistics()
  {
    if (!IsDemanded())
      return;

    AddTimer(PropagateTimerLabel_).start();
  }

  void
  StopPropagationStatistics()
  {
    if (!IsDemanded())
      return;

    GetTimer(PropagateTimerLabel_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(
      const util::StatisticsCollector & statisticsCollector,
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(statisticsCollector, rvsdgModule, pointsToGraph);
  }

private:
  /**
   * Checks if the pass statistics are demanded.
   *
   * @return True if the pass statistic is demanded, otherwise false.
   */
  [[nodiscard]] bool
  IsDemanded() const noexcept
  {
    return StatisticsCollector_.GetSettings().IsDemanded(GetId());
  }

  const util::StatisticsCollector & StatisticsCollector_;
};

/**
 * Class containing information about the memory locations that may be read from or written to by a
 * function call
 */
class CallSummary final
{
public:
  CallSummary(const rvsdg::SimpleNode & callNode, size_t sccIndex)
      : CallNode_(&callNode),
        CallGraphSccIndex_(sccIndex),
        PossiblyRecursive_(false)
  {
    JLM_ASSERT(is<CallOperation>(&callNode));
  }

  CallSummary(const CallSummary &) = delete;

  CallSummary(CallSummary &&) = delete;

  CallSummary &
  operator=(const CallSummary &) = delete;

  CallSummary &
  operator=(CallSummary &&) = delete;

  [[nodiscard]] const rvsdg::SimpleNode &
  GetCallNode() const noexcept
  {
    return *CallNode_;
  }

  void
  AddMemoryNodes(const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    MemoryNodes_.UnionWith(memoryNodes);
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetMemoryNodes() const noexcept
  {
    return MemoryNodes_;
  }

  void
  SetPossiblyRecursive() noexcept
  {
    PossiblyRecursive_ = true;
  }

  bool
  IsPossiblyRecursive() const noexcept
  {
    return PossiblyRecursive_;
  }

  /**
   * @return the SCC index the function containing this call belongs to in the call graph
   */
  size_t
  GetCallGraphSccIndex() const noexcept
  {
    return CallGraphSccIndex_;
  }

  /**
   * Creates a new CallSummary representing the given call node.
   * @param callNode the call node represented by the summary
   * @param sccIndex the index of the SCC the function containing the call belongs to, in the module
   * call graph
   * @return the newly created RegionSummary
   */
  static std::unique_ptr<CallSummary>
  Create(const rvsdg::SimpleNode & callNode, size_t sccIndex)
  {
    return std::make_unique<CallSummary>(callNode, sccIndex);
  }

private:
  // The call node represented by this summary
  const rvsdg::SimpleNode * CallNode_;

  // The set of memory locations that may be read from or written to by this call
  util::HashSet<const PointsToGraph::MemoryNode *> MemoryNodes_;

  // Which SCC in the call graph does the function containing this call belong to
  size_t CallGraphSccIndex_;
  // If it is possible for this call to be recursive, this flag will be set to true
  bool PossiblyRecursive_;
};

/**
 * Class containing a summary of memory locations possibly read from or written to inside a given
 * region
 */
class RegionSummary final
{
public:
  RegionSummary(const rvsdg::Region & region, size_t sccIndex)
      : Region_(&region),
        CallGraphSccIndex_(sccIndex),
        ContainsPossiblyRecursiveCall_(false)
  {}

  RegionSummary(const RegionSummary &) = delete;

  RegionSummary(RegionSummary &&) = delete;

  RegionSummary &
  operator=(const RegionSummary &) = delete;

  RegionSummary &
  operator=(RegionSummary &&) = delete;

  [[nodiscard]] const rvsdg::Region &
  GetRegion() const noexcept
  {
    return *Region_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetMemoryNodes() const
  {
    return MemoryNodes_;
  }

  void
  AddMemoryNodes(const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    MemoryNodes_.UnionWith(memoryNodes);
  }

  bool
  IsContainingPossiblyRecursiveCall() const noexcept
  {
    return ContainsPossiblyRecursiveCall_;
  }

  /**
   * @return the SCC index the function containing this region belongs to in the call graph
   */
  size_t
  GetCallGraphSccIndex() const noexcept
  {
    return CallGraphSccIndex_;
  }

  /**
   * Propagate information about memory locations being read from / written to by a call,
   * to the region containing the call.
   * @param callSummary the call summary
   * @param regionSummary the region summary representing the region containing the call
   */
  static void
  PropagateFromCall(RegionSummary & regionSummary, const CallSummary & callSummary)
  {
    regionSummary.AddMemoryNodes(callSummary.GetMemoryNodes());
    regionSummary.ContainsPossiblyRecursiveCall_ |= callSummary.IsPossiblyRecursive();
  }

  /**
   * Propagate information about memory locations being read from / written to in a region, to
   * another region summary.
   * @param dstSummary the target region summary
   * @param srcSummary the source region summary
   */
  static void
  PropagateToParentRegion(RegionSummary & dstSummary, const RegionSummary & srcSummary)
  {
    dstSummary.AddMemoryNodes(srcSummary.GetMemoryNodes());
    dstSummary.ContainsPossiblyRecursiveCall_ |= srcSummary.ContainsPossiblyRecursiveCall_;
  }

  /**
   * Creates a new RegionSummary representing the given region.
   * @param region the region represented by the summary
   * @param sccIndex the index of the SCC the region belongs to in the module call graph
   * @return the newly created RegionSummary
   */
  static std::unique_ptr<RegionSummary>
  Create(const rvsdg::Region & region, size_t sccIndex)
  {
    return std::make_unique<RegionSummary>(region, sccIndex);
  }

private:
  // The region represented by this summary
  const rvsdg::Region * Region_;
  // The set of memory locations that may be utilized in this region, any sub-region, or in calls
  // made in this region.
  util::HashSet<const PointsToGraph::MemoryNode *> MemoryNodes_;

  // Which SCC in the call graph does the function containing this region belong to
  size_t CallGraphSccIndex_;

  // If this region contains a call that may enter a new instance of this same region
  bool ContainsPossiblyRecursiveCall_;
};

/** \brief Mod/Ref summary of region-aware mod/ref summarizer
 *
 */
class RegionAwareModRefSummary final : public ModRefSummary
{
  using RegionSummaryMap =
      std::unordered_map<const rvsdg::Region *, std::unique_ptr<RegionSummary>>;
  using CallSummaryMap =
      std::unordered_map<const rvsdg::SimpleNode *, std::unique_ptr<CallSummary>>;

  using RegionSummaryIterator =
      util::MapValuePtrIterator<RegionSummary, RegionSummaryMap::const_iterator>;
  using RegionSummaryConstIterator =
      util::MapValuePtrIterator<const RegionSummary, RegionSummaryMap::const_iterator>;

  using CallSummaryIterator =
      util::MapValuePtrIterator<CallSummary, CallSummaryMap::const_iterator>;
  using CallSummaryConstIterator =
      util::MapValuePtrIterator<const CallSummary, CallSummaryMap::const_iterator>;

  using RegionSummaryRange = util::IteratorRange<RegionSummaryIterator>;
  using RegionSummaryConstRange = util::IteratorRange<RegionSummaryConstIterator>;

  using CallSummaryRange = util::IteratorRange<CallSummaryIterator>;
  using CallSummaryConstRange = util::IteratorRange<CallSummaryConstIterator>;

public:
  explicit RegionAwareModRefSummary(const PointsToGraph & pointsToGraph)
      : PointsToGraph_(pointsToGraph)
  {}

  RegionAwareModRefSummary(const RegionAwareModRefSummary &) = delete;

  RegionAwareModRefSummary(RegionAwareModRefSummary &&) = delete;

  RegionAwareModRefSummary &
  operator=(const RegionAwareModRefSummary &) = delete;

  RegionAwareModRefSummary &
  operator=(RegionAwareModRefSummary &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionEntryNodes(const rvsdg::Region & region) const override
  {
    const auto & regionSummary = GetRegionSummary(region);
    return regionSummary.GetMemoryNodes();
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionExitNodes(const rvsdg::Region & region) const override
  {
    const auto & regionSummary = GetRegionSummary(region);
    return regionSummary.GetMemoryNodes();
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallEntryNodes(const rvsdg::SimpleNode & callNode) const override
  {
    const auto & callSummary = GetCallSummary(callNode);
    return callSummary.GetMemoryNodes();
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallExitNodes(const rvsdg::SimpleNode & callNode) const override
  {
    return GetCallEntryNodes(callNode);
  }

  [[nodiscard]] util::HashSet<const PointsToGraph::MemoryNode *>
  GetOutputNodes(const rvsdg::Output & output) const override
  {
    JLM_ASSERT(is<PointerType>(output.Type()));

    util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
    const auto registerNode = &PointsToGraph_.GetRegisterNode(output);
    for (auto & memoryNode : registerNode->Targets())
      memoryNodes.Insert(&memoryNode);

    return memoryNodes;
  }

  [[nodiscard]] RegionSummaryRange
  GetRegionSummaries()
  {
    return { RegionSummaryIterator(RegionSummaries_.begin()),
             RegionSummaryIterator(RegionSummaries_.end()) };
  }

  [[nodiscard]] RegionSummaryConstRange
  GetRegionSummaries() const
  {
    return { RegionSummaryConstIterator(RegionSummaries_.begin()),
             RegionSummaryConstIterator(RegionSummaries_.end()) };
  }

  [[nodiscard]] RegionSummary *
  TryGetRegionSummary(const rvsdg::Region & region) const
  {
    const auto it = RegionSummaries_.find(&region);
    if (it == RegionSummaries_.end())
      return nullptr;
    return it->second.get();
  }

  [[nodiscard]] bool
  ContainsRegionSummary(const rvsdg::Region & region) const
  {
    return TryGetRegionSummary(region) != nullptr;
  }

  [[nodiscard]] RegionSummary &
  GetRegionSummary(const rvsdg::Region & region) const
  {
    const auto regionSummary = TryGetRegionSummary(region);
    JLM_ASSERT(regionSummary != nullptr);
    return *regionSummary;
  }

  RegionSummary &
  AddRegionSummary(std::unique_ptr<RegionSummary> regionSummary)
  {
    JLM_ASSERT(!ContainsRegionSummary(regionSummary->GetRegion()));

    auto region = &regionSummary->GetRegion();
    auto regionSummaryPointer = regionSummary.get();
    RegionSummaries_[region] = std::move(regionSummary);
    return *regionSummaryPointer;
  }

  [[nodiscard]] CallSummaryRange
  GetCallSummaries()
  {
    return { CallSummaryIterator(CallSummaries_.begin()),
             CallSummaryIterator(CallSummaries_.end()) };
  }

  [[nodiscard]] CallSummaryConstRange
  GetCallSummaries() const
  {
    return { CallSummaryConstIterator(CallSummaries_.begin()),
             CallSummaryConstIterator(CallSummaries_.end()) };
  }

  [[nodiscard]] CallSummary *
  TryGetCallSummary(const rvsdg::SimpleNode & call) const
  {
    const auto it = CallSummaries_.find(&call);
    if (it == CallSummaries_.end())
      return nullptr;
    return it->second.get();
  }

  [[nodiscard]] bool
  ContainsCallSummary(const rvsdg::SimpleNode & call) const
  {
    JLM_ASSERT(is<CallOperation>(&call));
    return TryGetCallSummary(call) != nullptr;
  }

  [[nodiscard]] CallSummary &
  GetCallSummary(const rvsdg::SimpleNode & call) const
  {
    const auto callSummary = TryGetCallSummary(call);
    JLM_ASSERT(callSummary != nullptr);
    return *callSummary;
  }

  CallSummary &
  AddCallSummary(std::unique_ptr<CallSummary> callSummary)
  {
    JLM_ASSERT(!ContainsCallSummary(callSummary->GetCallNode()));

    auto callNode = &callSummary->GetCallNode();
    auto callSummaryPointer = callSummary.get();
    CallSummaries_[callNode] = std::move(callSummary);
    return *callSummaryPointer;
  }

  [[nodiscard]] static std::unique_ptr<RegionAwareModRefSummary>
  Create(const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<RegionAwareModRefSummary>(pointsToGraph);
  }

private:
  const PointsToGraph & PointsToGraph_;
  RegionSummaryMap RegionSummaries_;
  CallSummaryMap CallSummaries_;
};

RegionAwareModRefSummarizer::~RegionAwareModRefSummarizer() noexcept = default;

RegionAwareModRefSummarizer::RegionAwareModRefSummarizer() = default;

std::unique_ptr<ModRefSummary>
RegionAwareModRefSummarizer::SummarizeModRefs(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  ModRefSummary_ = RegionAwareModRefSummary::Create(pointsToGraph);
  Context_ = Context{};
  auto statistics = Statistics::Create(statisticsCollector, rvsdgModule, pointsToGraph);

  statistics->StartCallGraphStatistics();
  CreateCallGraph(rvsdgModule);
  statistics->StopCallGraphStatistics(Context_.SccFunctions.size());

  // Create summaries per SCC to quickly handle function calls and recursion
  Context_.SccSummaries.resize(Context_.SccFunctions.size());

  statistics->StartAnnotationStatistics();
  // Go through SCCs in reverse topological order and annotate all functions
  for (size_t sccIndex = 0; sccIndex < Context_.SccFunctions.size(); sccIndex++)
  {
    for (auto function : Context_.SccFunctions[sccIndex].Items())
    {
      AnnotateFunction(*function, sccIndex);
    }

    // The SCC containing all external functions possibly utilizes any memory location that has
    // escaped
    if (sccIndex == Context_.ExternalNodeSccIndex)
    {
      auto escapedMemoryLocation = ModRefSummary_->GetPointsToGraph().GetEscapedMemoryNodes();
      Context_.SccSummaries[Context_.ExternalNodeSccIndex].UnionWithAndClear(escapedMemoryLocation);
    }
  }
  statistics->StopAnnotationStatistics();

  // All functions have been annotated, but recursion within SCCs has not been handled yet.
  // This is fixed by revisiting all recursive RegionSummaries and CallSummaries and adding the SCC
  // summary.
  statistics->StartPropagationStatistics();
  PropagateRecursiveMemoryLocations();
  statistics->StopPropagationStatistics();

  // Print debug output
  // std::cerr << "Call Graph SCCs:" << std::endl << CallGraphSCCsToString(*this) << std::endl;
  // std::cerr << "RegionTree:" << std::endl << ToRegionTree(rvsdgModule.Rvsdg(), *ModRefSummary_)
  // << std::endl;

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
  return std::move(ModRefSummary_);
}

std::unique_ptr<ModRefSummary>
RegionAwareModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  RegionAwareModRefSummarizer summarizer;
  return summarizer.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<ModRefSummary>
RegionAwareModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph)
{
  util::StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

void
RegionAwareModRefSummarizer::CreateCallGraph(const rvsdg::RvsdgModule & rvsdgModule)
{
  // The list of lambdas becomes the list of nodes in the call graph
  auto lambdaNodes = CollectLambdaNodes(rvsdgModule);

  // Mapping from LambdaNode* to its index in lambdaNodes
  std::unordered_map<const rvsdg::LambdaNode *, size_t> callGraphNodeIndex;
  callGraphNodeIndex.reserve(lambdaNodes.size());
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    callGraphNodeIndex.insert({ lambdaNodes[i], i });
  }

  // Add a dummy node representing all external functions, with no associated LambdaNode
  const auto externalNodeIndex = lambdaNodes.size();
  const auto numCallGraphNodes = externalNodeIndex + 1;

  // Outgoing edges for each node in the call graph
  std::vector<util::HashSet<size_t>> callGraphSuccessors(numCallGraphNodes);

  const auto & pointsToGraph = ModRefSummary_->GetPointsToGraph();

  // Add outgoing edges from the given caller to any function the call may target
  const auto HandleCall = [&](rvsdg::Node & callNode, size_t callerIndex) -> void
  {
    JLM_ASSERT(is<CallOperation>(&callNode));
    const auto targetPtr = callNode.input(0)->origin();
    const auto & targetPtrNode = pointsToGraph.GetRegisterNode(*targetPtr);

    // Go through all locations the called function pointer may target
    for (auto & callee : targetPtrNode.Targets())
    {
      if (auto lambdaCallee = dynamic_cast<const PointsToGraph::LambdaNode *>(&callee))
      {
        const auto & lambdaNode = lambdaCallee->GetLambdaNode();

        // Look up which call graph node represents the target lambda
        JLM_ASSERT(callGraphNodeIndex.find(&lambdaNode) != callGraphNodeIndex.end());
        const auto calleeCallGraphNode = callGraphNodeIndex[&lambdaNode];

        // Add the edge caller -> callee to the call graph
        callGraphSuccessors[callerIndex].Insert(calleeCallGraphNode);
      }
      else if (
          PointsToGraph::Node::Is<PointsToGraph::ExternalMemoryNode>(callee)
          || PointsToGraph::Node::Is<PointsToGraph::ImportNode>(callee))
      {
        // Add the edge caller -> node representing external functions
        callGraphSuccessors[callerIndex].Insert(externalNodeIndex);
      }
    }
  };

  // Recursive function finding all call operations, adding edges to the call graph
  const std::function<void(rvsdg::Region &, size_t)> HandleCalls = [&](rvsdg::Region & region,
                                                                       size_t callerIndex) -> void
  {
    for (auto & node : region.Nodes())
    {
      if (is<CallOperation>(&node))
      {
        HandleCall(node, callerIndex);
      }
      else if (auto structural = dynamic_cast<rvsdg::StructuralNode *>(&node))
      {
        for (size_t i = 0; i < structural->nsubregions(); i++)
        {
          HandleCalls(*structural->subregion(i), callerIndex);
        }
      }
    }
  };

  // For all functions, visit all their calls and add outgoing edges in the call graph
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    HandleCalls(*lambdaNodes[i]->subregion(), i);

    // If the function has escaped, add an edge from the node representing all external functions
    const auto & lambdaMemoryNode = pointsToGraph.GetLambdaNode(*lambdaNodes[i]);
    if (pointsToGraph.GetEscapedMemoryNodes().Contains(&lambdaMemoryNode))
    {
      callGraphSuccessors[externalNodeIndex].Insert(i);
    }
  }

  const auto GetSuccessor = [&](size_t nodeIndex)
  {
    return callGraphSuccessors[nodeIndex].Items();
  };

  // Find SCCs in the call graph
  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSCCs = util::FindStronglyConnectedComponents<size_t>(
      numCallGraphNodes,
      GetSuccessor,
      sccIndex,
      reverseTopologicalOrder);

  // sccIndex are distributed in a reverse topological order, so the sccIndex is used
  // when creating the list of SCCs and the functions they contain
  Context_.SccFunctions.resize(numSCCs);
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    Context_.SccFunctions[sccIndex[i]].Insert(lambdaNodes[i]);
    Context_.FunctionToSccIndex[lambdaNodes[i]] = sccIndex[i];
  }

  // Also note which SCC contains all external functions
  Context_.ExternalNodeSccIndex = sccIndex[externalNodeIndex];
}

void
RegionAwareModRefSummarizer::AnnotateFunction(const rvsdg::LambdaNode & lambda, size_t sccIndex)
{
  auto & summary = AnnotateRegion(*lambda.subregion(), sccIndex);

  // Inform the SCC about the memory locations being affected by functions inside it
  Context_.SccSummaries[sccIndex].UnionWith(summary.GetMemoryNodes());
}

RegionSummary &
RegionAwareModRefSummarizer::AnnotateRegion(rvsdg::Region & region, size_t sccIndex)
{
  auto & summary = ModRefSummary_->AddRegionSummary(RegionSummary::Create(region, sccIndex));

  for (auto & node : region.Nodes())
  {
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      AnnotateStructuralNode(*structuralNode, summary);
    }
    else if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
    {
      AnnotateSimpleNode(*simpleNode, summary);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }

  return summary;
}

void
RegionAwareModRefSummarizer::AnnotateStructuralNode(
    const rvsdg::StructuralNode & structuralNode,
    RegionSummary & regionSummary)
{
  // A subregion always belongs to the same call graph SCC as its parent region
  const auto sccIndex = regionSummary.GetCallGraphSccIndex();

  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    auto & subregionSummary = AnnotateRegion(*structuralNode.subregion(n), sccIndex);
    RegionSummary::PropagateToParentRegion(regionSummary, subregionSummary);
  }
}

void
RegionAwareModRefSummarizer::AnnotateSimpleNode(
    const rvsdg::SimpleNode & simpleNode,
    RegionSummary & regionSummary)
{
  if (is<LoadOperation>(&simpleNode))
  {
    AnnotateLoad(simpleNode, regionSummary);
  }
  else if (is<StoreOperation>(&simpleNode))
  {
    AnnotateStore(simpleNode, regionSummary);
  }
  else if (is<AllocaOperation>(&simpleNode))
  {
    AnnotateAlloca(simpleNode, regionSummary);
  }
  else if (is<MallocOperation>(&simpleNode))
  {
    AnnotateMalloc(simpleNode, regionSummary);
  }
  else if (is<FreeOperation>(&simpleNode))
  {
    AnnotateFree(simpleNode, regionSummary);
  }
  else if (is<CallOperation>(&simpleNode))
  {
    AnnotateCall(simpleNode, regionSummary);
  }
  else if (is<MemCpyOperation>(&simpleNode))
  {
    AnnotateMemcpy(simpleNode, regionSummary);
  }
}

void
RegionAwareModRefSummarizer::AnnotateLoad(
    const rvsdg::SimpleNode & loadNode,
    RegionSummary & regionSummary)
{
  const auto origin = LoadOperation::AddressInput(loadNode).origin();
  const auto memoryNodes = ModRefSummary_->GetOutputNodes(*origin);
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareModRefSummarizer::AnnotateStore(
    const rvsdg::SimpleNode & storeNode,
    RegionSummary & regionSummary)
{
  const auto origin = StoreOperation::AddressInput(storeNode).origin();
  const auto memoryNodes = ModRefSummary_->GetOutputNodes(*origin);
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareModRefSummarizer::AnnotateAlloca(
    const rvsdg::SimpleNode & allocaNode,
    RegionSummary & regionSummary)
{
  JLM_ASSERT(is<AllocaOperation>(&allocaNode));

  auto & memoryNode = ModRefSummary_->GetPointsToGraph().GetAllocaNode(allocaNode);
  regionSummary.AddMemoryNodes({ &memoryNode });
}

void
RegionAwareModRefSummarizer::AnnotateMalloc(
    const rvsdg::SimpleNode & mallocNode,
    RegionSummary & regionSummary)
{
  JLM_ASSERT(is<MallocOperation>(&mallocNode));

  auto & memoryNode = ModRefSummary_->GetPointsToGraph().GetMallocNode(mallocNode);
  regionSummary.AddMemoryNodes({ &memoryNode });
}

void
RegionAwareModRefSummarizer::AnnotateFree(
    const rvsdg::SimpleNode & freeNode,
    RegionSummary & regionSummary)
{
  JLM_ASSERT(is<FreeOperation>(&freeNode));

  auto memoryNodes = ModRefSummary_->GetOutputNodes(*freeNode.input(0)->origin());
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareModRefSummarizer::AnnotateMemcpy(
    const rvsdg::SimpleNode & memcpyNode,
    RegionSummary & regionSummary)
{
  JLM_ASSERT(is<MemCpyOperation>(&memcpyNode));

  auto dstNodes = ModRefSummary_->GetOutputNodes(*memcpyNode.input(0)->origin());
  regionSummary.AddMemoryNodes(dstNodes);

  auto srcNodes = ModRefSummary_->GetOutputNodes(*memcpyNode.input(1)->origin());
  regionSummary.AddMemoryNodes(srcNodes);
}

void
RegionAwareModRefSummarizer::AnnotateCall(
    const rvsdg::SimpleNode & callNode,
    RegionSummary & regionSummary)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  // A call has the same sccIndex as the region it lives in
  const auto sccIndex = regionSummary.GetCallGraphSccIndex();
  auto & callSummary = ModRefSummary_->AddCallSummary(CallSummary::Create(callNode, sccIndex));

  // Go over all possible targets of the call and add them to the call summary
  const auto targetPtr = callNode.input(0)->origin();
  const auto & targetPtrNode = ModRefSummary_->GetPointsToGraph().GetRegisterNode(*targetPtr);

  // Go through all locations the called function pointer may target
  for (auto & callee : targetPtrNode.Targets())
  {
    size_t targetSccIndex = 0;
    if (auto lambdaCallee = dynamic_cast<const PointsToGraph::LambdaNode *>(&callee))
    {
      const auto & lambdaNode = lambdaCallee->GetLambdaNode();

      // Look up which SCC the callee belongs to
      JLM_ASSERT(
          Context_.FunctionToSccIndex.find(&lambdaNode) != Context_.FunctionToSccIndex.end());
      targetSccIndex = Context_.FunctionToSccIndex[&lambdaNode];
    }
    else if (
        PointsToGraph::Node::Is<PointsToGraph::ExternalMemoryNode>(callee)
        || PointsToGraph::Node::Is<PointsToGraph::ImportNode>(callee))
    {
      targetSccIndex = Context_.ExternalNodeSccIndex;
    }
    else
    {
      // The target is not a callable type, ignore it
      continue;
    }

    if (targetSccIndex < sccIndex)
    {
      callSummary.AddMemoryNodes(Context_.SccSummaries[targetSccIndex]);
    }
    else if (targetSccIndex == sccIndex)
    {
      callSummary.SetPossiblyRecursive();
    }
    else
    {
      JLM_UNREACHABLE("Calls can never target functions in later SCCs");
    }
  }

  // Inform the region about everything the call may affect, and if the call is recursive
  RegionSummary::PropagateFromCall(regionSummary, callSummary);
}

void
RegionAwareModRefSummarizer::PropagateRecursiveMemoryLocations()
{
  // Go over all summaries that may contain recursion and add all memory locations utilized
  // anywhere in the SCC to the summary.
  for (auto & regionSummary : ModRefSummary_->GetRegionSummaries())
  {
    if (!regionSummary.IsContainingPossiblyRecursiveCall())
      continue;

    const auto sccIndex = regionSummary.GetCallGraphSccIndex();
    regionSummary.AddMemoryNodes(Context_.SccSummaries[sccIndex]);
  }

  for (auto & callSummary : ModRefSummary_->GetCallSummaries())
  {
    if (!callSummary.IsPossiblyRecursive())
      continue;

    const auto sccIndex = callSummary.GetCallGraphSccIndex();
    callSummary.AddMemoryNodes(Context_.SccSummaries[sccIndex]);
  }
}

std::vector<const rvsdg::LambdaNode *>
RegionAwareModRefSummarizer::CollectLambdaNodes(const rvsdg::RvsdgModule & rvsdgModule)
{
  std::vector<const rvsdg::LambdaNode *> result;

  // Recursively traverses all structural nodes, but does not enter into lambdas
  const std::function<void(rvsdg::Region &)> CollectLambdasInRegion =
      [&](rvsdg::Region & region) -> void
  {
    for (auto & node : region.Nodes())
    {
      if (auto lambda = dynamic_cast<rvsdg::LambdaNode *>(&node))
      {
        result.push_back(lambda);
      }
      else if (auto structural = dynamic_cast<rvsdg::StructuralNode *>(&node))
      {
        for (size_t i = 0; i < structural->nsubregions(); i++)
        {
          CollectLambdasInRegion(*structural->subregion(i));
        }
      }
    }
  };

  CollectLambdasInRegion(rvsdgModule.Rvsdg().GetRootRegion());

  return result;
}

std::string
RegionAwareModRefSummarizer::CallGraphSCCsToString(const RegionAwareModRefSummarizer & summarizer)
{
  std::ostringstream ss;
  for (size_t i = 0; i < summarizer.Context_.SccFunctions.size(); i++)
  {
    if (i != 0)
      ss << " <- ";
    ss << "[" << std::endl;
    if (i == summarizer.Context_.ExternalNodeSccIndex)
    {
      ss << "  " << "<external>" << std::endl;
    }
    for (auto function : summarizer.Context_.SccFunctions[i].Items())
    {
      ss << "  " << function->DebugString() << std::endl;
    }
    ss << "]";
  }
  return ss.str();
}

std::string
RegionAwareModRefSummarizer::ToRegionTree(
    const rvsdg::Graph & rvsdg,
    const RegionAwareModRefSummary & modRefSummary)
{
  auto toString = [](const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    std::string s = "{";
    for (auto & memoryNode : memoryNodes.Items())
    {
      s += util::strfmt(memoryNode, ", ");
    }
    s += "}";
    return s;
  };

  auto indent = [](size_t depth)
  {
    return std::string(depth, '-');
  };

  std::function<std::string(const rvsdg::Region *, size_t)> toRegionTree =
      [&](const rvsdg::Region * region, size_t depth)
  {
    std::string subtree;
    if (region->node())
    {
      subtree += util::strfmt(indent(depth), region, "\n");
    }
    else
    {
      subtree = "ROOT\n";
    }

    depth += 1;
    if (modRefSummary.ContainsRegionSummary(*region))
    {
      auto & regionSummary = modRefSummary.GetRegionSummary(*region);
      auto & memoryNodes = regionSummary.GetMemoryNodes();
      subtree += util::strfmt(indent(depth), "MemoryNodes: ", toString(memoryNodes), "\n");
    }

    for (const auto & node : region->Nodes())
    {
      if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
      {
        subtree += util::strfmt(indent(depth), structuralNode->DebugString(), "\n");
        for (size_t n = 0; n < structuralNode->nsubregions(); n++)
        {
          subtree += toRegionTree(structuralNode->subregion(n), depth + 1);
        }
      }
    }

    return subtree;
  };

  return toRegionTree(&rvsdg.GetRootRegion(), 0);
}

}
