/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/trace.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareModRefSummarizer.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/rvsdg/FunctionType.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/TarjanScc.hpp>
#include <jlm/util/Worklist.hpp>

#include <queue>

namespace jlm::llvm::aa
{

/**
 * allocas that are not defined in f(), and not defined in a predecessor of f() in the call graph,
 * cannot be live inside f(). They are added to the DeadAllocaBlocklist.
 */
static const bool ENABLE_DEAD_ALLOCA_BLOCKLIST = !std::getenv("JLM_DISABLE_DEAD_ALLOCA_BLOCKLIST");

/**
 * In a region with an alloca definition, the memory node representing the alloca does not need to
 * be routed into the region if the alloca is shown to be non-reentrant.
 * Such allocas are added to the NonReentrantBlocklist.
 */
static const bool ENABLE_NON_REENTRANT_ALLOCA_BLOCKLIST =
    !std::getenv("JLM_DISABLE_NON_REENTRANT_ALLOCA_BLOCKLIST");

/**
 * Operations like loads and stores have a size.
 * If the size is larger than the size of a memory represented by a memory node X,
 * X can be excluded from the Mod/Ref summary of the operation.
 */
static const bool ENABLE_OPERATION_SIZE_BLOCKING =
    !std::getenv("JLM_DISABLE_OPERATION_SIZE_BLOCKING");

/**
 * Constant memory, such as functions, constant globals and constant import, can never change.
 * We therefore never need to route their memory states through anything.
 */
static const bool ENABLE_CONSTANT_MEMORY_BLOCKING =
    !std::getenv("JLM_DISABLE_CONSTANT_MEMORY_BLOCKING");

/** \brief Region-aware mod/ref summarizer statistics
 *
 * The statistics collected when running the region-aware mod/ref summarizer.
 *
 * @see RegionAwareModRefSummarizer
 */
class RegionAwareModRefSummarizer::Statistics final : public util::Statistics
{
  static constexpr auto NumRvsdgRegionsLabel_ = "#RvsdgRegions";
  static constexpr auto NumSimpleAllocas_ = "#SimpleAllocas";
  static constexpr auto NumNonReentrantAllocas_ = "#NonReentrantAllocas";
  static constexpr auto NumCallGraphSccs_ = "#CallGraphSccs";
  static constexpr auto NumFunctionsInActiveSetjmp_ = "#FunctionsInActiveSetjmp";
  static constexpr auto NumCallGraphSccsCanCallExternal_ = "#CallGraphSccsCanCallExternal";

  static constexpr auto CallGraphTimer_ = "CallGraphTimer";
  static constexpr auto SccsThatCanCallExternalTimer_ = "SccsThatCanCallExternalTimer";
  static constexpr auto AllocasDeadInSccsTimer_ = "AllocasDeadInSccsTimer";
  static constexpr auto SimpleAllocasSetTimer_ = "SimpleAllocasSetTimer";
  static constexpr auto NonReentrantAllocaSetsTimer_ = "NonReentrantAllocaSetsTimer";
  static constexpr auto CreateExternalModRefSetTimer_ = "CreateExternalModRefSetTimer";
  static constexpr auto AnnotationTimer_ = "AnnotationTimer";
  static constexpr auto SolvingTimer_ = "SolvingTimer";

public:
  ~Statistics() override = default;

  explicit Statistics(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph)
      : util::Statistics(Id::RegionAwareModRefSummarizer, rvsdgModule.SourceFilePath().value())
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&rvsdgModule.Rvsdg().GetRootRegion()));
    AddMeasurement(
        NumRvsdgRegionsLabel_,
        rvsdg::Region::NumRegions(rvsdgModule.Rvsdg().GetRootRegion()));
    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.numMemoryNodes());
  }

  void
  startCallGraphStatistics()
  {
    AddTimer(CallGraphTimer_).start();
  }

  void
  stopCallGraphStatistics(size_t numSccs, size_t numFunctionsInActiveSetjmp)
  {
    GetTimer(CallGraphTimer_).stop();
    AddMeasurement(NumCallGraphSccs_, numSccs);
    AddMeasurement(NumFunctionsInActiveSetjmp_, numFunctionsInActiveSetjmp);
  }

  void
  startFindSccsThatCanCallExternalStatistics()
  {
    AddTimer(SccsThatCanCallExternalTimer_).start();
  }

  void
  stopFindSccsThatCanCallExternalStatistics(size_t numSccsCanCallExternal)
  {
    GetTimer(SccsThatCanCallExternalTimer_).stop();
    AddMeasurement(NumCallGraphSccsCanCallExternal_, numSccsCanCallExternal);
  }

  void
  StartAllocasDeadInSccStatistics()
  {
    AddTimer(AllocasDeadInSccsTimer_).start();
  }

  void
  StopAllocasDeadInSccStatistics()
  {
    GetTimer(AllocasDeadInSccsTimer_).stop();
  }

  void
  StartCreateSimpleAllocasSetStatistics()
  {
    AddTimer(SimpleAllocasSetTimer_).start();
  }

  void
  StopCreateSimpleAllocasSetStatistics(uint64_t numSimpleAllocas)
  {
    GetTimer(SimpleAllocasSetTimer_).stop();
    AddMeasurement(NumSimpleAllocas_, numSimpleAllocas);
  }

  void
  StartCreateNonReentrantAllocaSetsStatistics()
  {
    AddTimer(NonReentrantAllocaSetsTimer_).start();
  }

  void
  StopCreateNonReentrantAllocaSetsStatistics(size_t numNonReentrantAllocas)
  {
    AddMeasurement(NumNonReentrantAllocas_, numNonReentrantAllocas);
    GetTimer(NonReentrantAllocaSetsTimer_).stop();
  }

  void
  StartCreateExternalModRefSet()
  {
    AddTimer(CreateExternalModRefSetTimer_).start();
  }

  void
  StopCreateExternalModRefSet()
  {
    GetTimer(CreateExternalModRefSetTimer_).stop();
  }

  void
  StartAnnotationStatistics()
  {
    AddTimer(AnnotationTimer_).start();
  }

  void
  StopAnnotationStatistics()
  {
    GetTimer(AnnotationTimer_).stop();
  }

  void
  StartSolvingStatistics()
  {
    AddTimer(SolvingTimer_).start();
  }

  void
  StopSolvingStatistics()
  {
    GetTimer(SolvingTimer_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const rvsdg::RvsdgModule & rvsdgModule, const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Statistics>(rvsdgModule, pointsToGraph);
  }
};

/**
 * Class representing the set of MemoryNodes that may be modified or referenced by some operation,
 * or within some region.
 */
class ModRefSet final
{
public:
  ModRefSet() = default;

  [[nodiscard]] util::HashSet<PointsToGraph::NodeIndex> &
  GetMemoryNodes()
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetMemoryNodes() const
  {
    return MemoryNodes_;
  }

private:
  util::HashSet<PointsToGraph::NodeIndex> MemoryNodes_;
};

/** \brief Mod/Ref summary of region-aware mod/ref summarizer
 */
class RegionAwareModRefSummary final : public ModRefSummary
{
public:
  explicit RegionAwareModRefSummary(const PointsToGraph & pointsToGraph)
      : PointsToGraph_(pointsToGraph)
  {}

  RegionAwareModRefSummary(const RegionAwareModRefSummary &) = delete;
  RegionAwareModRefSummary &
  operator=(const RegionAwareModRefSummary &) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] size_t
  NumModRefSets() const noexcept
  {
    return ModRefSets_.size();
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetModRefSet(ModRefSetIndex index) const
  {
    JLM_ASSERT(index < ModRefSets_.size());
    return ModRefSets_[index].GetMemoryNodes();
  }

  bool
  AddToModRefSet(ModRefSetIndex index, PointsToGraph::NodeIndex ptgNode)
  {
    JLM_ASSERT(index < ModRefSets_.size());
    return ModRefSets_[index].GetMemoryNodes().insert(ptgNode);
  }

  bool
  PropagateModRefSet(ModRefSetIndex from, ModRefSetIndex to)
  {
    JLM_ASSERT(from < ModRefSets_.size());
    JLM_ASSERT(to < ModRefSets_.size());
    return ModRefSets_[to].GetMemoryNodes().UnionWith(ModRefSets_[from].GetMemoryNodes());
  }

  /**
   * Creates a new ModRefSet that is not mapped to any node
   * @return the index of the new ModRefSet
   */
  [[nodiscard]] ModRefSetIndex
  CreateModRefSet()
  {
    ModRefSets_.emplace_back();
    return ModRefSets_.size() - 1;
  }

  [[nodiscard]] bool
  HasSetForNode(const rvsdg::Node & node) const
  {
    return NodeMap_.find(&node) != NodeMap_.end();
  }

  [[nodiscard]] ModRefSetIndex
  GetSetForNode(const rvsdg::Node & node) const
  {
    const auto it = NodeMap_.find(&node);
    JLM_ASSERT(it != NodeMap_.end());
    return it->second;
  }

  [[nodiscard]] ModRefSetIndex
  GetOrCreateSetForNode(const rvsdg::Node & node)
  {
    if (const auto it = NodeMap_.find(&node); it != NodeMap_.end())
      return it->second;

    return NodeMap_[&node] = CreateModRefSet();
  }

  void
  MapNodeToSet(const rvsdg::Node & node, ModRefSetIndex index)
  {
    JLM_ASSERT(!HasSetForNode(node));
    NodeMap_[&node] = index;
  }

  const util::HashSet<PointsToGraph::NodeIndex> &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const override
  {
    return ModRefSets_[GetSetForNode(node)].GetMemoryNodes();
  }

  const util::HashSet<PointsToGraph::NodeIndex> &
  GetGammaEntryModRef(const rvsdg::GammaNode & gamma) const override
  {
    return ModRefSets_[GetSetForNode(gamma)].GetMemoryNodes();
  }

  const util::HashSet<PointsToGraph::NodeIndex> &
  GetGammaExitModRef(const rvsdg::GammaNode & gamma) const override
  {
    return GetGammaEntryModRef(gamma);
  }

  const util::HashSet<PointsToGraph::NodeIndex> &
  GetThetaModRef(const rvsdg::ThetaNode & theta) const override
  {
    return ModRefSets_[GetSetForNode(theta)].GetMemoryNodes();
  }

  const util::HashSet<PointsToGraph::NodeIndex> &
  GetLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const override
  {
    return ModRefSets_[GetSetForNode(lambda)].GetMemoryNodes();
  }

  const util::HashSet<PointsToGraph::NodeIndex> &
  GetLambdaExitModRef(const rvsdg::LambdaNode & lambda) const override
  {
    return GetLambdaEntryModRef(lambda);
  }

  [[nodiscard]] static std::unique_ptr<RegionAwareModRefSummary>
  Create(const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<RegionAwareModRefSummary>(pointsToGraph);
  }

private:
  const PointsToGraph & PointsToGraph_;

  /**
   * All sets of ModRef information in the summary
   */
  std::vector<ModRefSet> ModRefSets_;

  /**
   * Map from nodes that have memory side effects, to their ModRefSet.
   * Includes nodes like loads, stores, memcpy, free and calls.
   * Also includes structural nodes like gamma, theta and lambda.
   */
  std::unordered_map<const rvsdg::Node *, ModRefSetIndex> NodeMap_;
};

/**
 * Struct holding temporary data used during the creation of a single mod/ref summary
 */
struct RegionAwareModRefSummarizer::Context
{
  explicit Context(const PointsToGraph & ptg)
      : pointsToGraph(ptg)
  {}

  /**
   * The points to graph used to create the Mod/Ref summary.
   */
  const PointsToGraph & pointsToGraph;

  /**
   * The set of functions belonging to each SCC in the call graph.
   * The SCCs are ordered in reverse topological order, so
   * if function a() calls b(), and they are not in the same SCC,
   * the SCC containing a() comes after the SCC containing b().
   *
   * External functions are not included in these sets, see \ref ExternalNodeSccIndex.
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<const rvsdg::LambdaNode *>> SccFunctions;

  /**
   * The index of the SCC in the call graph that represent containing all external functions
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  size_t ExternalNodeSccIndex = 0;

  /**
   * For each SCC in the call graph, the set of SCCs it targets using calls.
   * Since SCCs are ordered in reverse topological order, an SCC never targets higher indices.
   * If there is any possibility of recursion within an SCC, it also targets itself.
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<size_t>> SccCallTargets;

  /**
   * A mapping from functions to the index of the SCC they belong to in the call graph
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  std::unordered_map<const rvsdg::LambdaNode *, size_t> FunctionToSccIndex;

  /**
   * The set of functions that can be on the top of the stack within an active setjmp.
   * In these functions, any call to an external function may trigger a longjmp,
   * clearing the stack back to the setjmp point.
   *
   * Call stacks that go via external modules do not count.
   * The setjmp call, and all intermediate calls, must occur in this module. Take for example:
   *
   *   /-[external] <-
   *  v               \
   * f() --> g() --> h() --> k()
   *         \---> setjmp()
   *
   * Here, the functions g(), h() and k() should be marked as being in an active setjmp,
   * but f() should not, as g() can only call f() via [external].
   *
   * Assigned in \ref createCallGraph(). Remains constant after.
   */
  util::HashSet<const rvsdg::LambdaNode *> FunctionsInActiveSetjmp;

  /**
   * This array is true iff it is possible for a function in the given SCC
   * to call an externally defined function, either directly or indirectly.
   * The array is indexed by SCC index.
   *
   * Assigned in \ref findSccsThatCanCallExternal(). Remains constant after.
   */
  std::vector<bool> SccCanCallExternal;

  /**
   * For each SCC, only allocas defined within the SCC, or within a predecessor of the SCC,
   * can possibly be live. All other allocas are considered dead in the SCC.
   *
   * Assigned in \ref FindAllocasDeadInSccs(). Remains constant after.
   */
  std::vector<util::HashSet<PointsToGraph::NodeIndex>> AllocasDeadInScc;

  /**
   * The set of all Simple Allocas in the module.
   *
   * Assigned in \ref CreateSimpleAllocaSet(). Remains constant after.
   */
  util::HashSet<PointsToGraph::NodeIndex> SimpleAllocas;

  /**
   * For each region, this field contains the set of allocas defined in the region,
   * that have been shown to be non-reentrant.
   *
   * Assigned in \ref CreateNonReentrantAllocaSets(). Remains constant after.
   */
  std::unordered_map<const rvsdg::Region *, util::HashSet<PointsToGraph::NodeIndex>>
      NonReentrantAllocas;

  /**
   * A ModRefSet containing all MemoryNodes that can be read or written to from external functions.
   *
   * Assigned in \ref CreateExternalModRefSet(). Remains constant after.
   */
  ModRefSetIndex ExternalModRefIndex = 0;

  /**
   * Simple edges in the ModRefSet constraint graph.
   * A simple edge a -> b indicates that the ModRefSet b should contain everything in a.
   * ModRefSetSimpleEdges[a] contains b, as well as any other simple edge successors.
   */
  std::vector<util::HashSet<ModRefSetIndex>> ModRefSetSimpleConstraints;

  /**
   * Blocklists in the ModRefSet constraint graphs.
   * During solving, a MemoryNode x that is about to be propagated to a ModRefSet a
   * will be skipped if x is in the Blocklist associated with a.
   * The pointer to the blocklist must remain valid until solving is finished.
   */
  std::unordered_map<ModRefSetIndex, const util::HashSet<PointsToGraph::NodeIndex> *>
      ModRefSetBlocklists;
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
  Context_ = std::make_unique<Context>(pointsToGraph);
  auto statistics = Statistics::Create(rvsdgModule, pointsToGraph);

  statistics->startCallGraphStatistics();
  createCallGraph(rvsdgModule);
  statistics->stopCallGraphStatistics(
      Context_->SccFunctions.size(),
      Context_->FunctionsInActiveSetjmp.Size());

  statistics->startFindSccsThatCanCallExternalStatistics();
  const auto numSccsCanCallExternal = findSccsThatCanCallExternal();
  statistics->stopFindSccsThatCanCallExternalStatistics(numSccsCanCallExternal);

  statistics->StartAllocasDeadInSccStatistics();
  FindAllocasDeadInSccs();
  statistics->StopAllocasDeadInSccStatistics();

  statistics->StartCreateSimpleAllocasSetStatistics();
  Context_->SimpleAllocas = CreateSimpleAllocaSet(pointsToGraph);
  statistics->StopCreateSimpleAllocasSetStatistics(Context_->SimpleAllocas.Size());

  statistics->StartCreateNonReentrantAllocaSetsStatistics();
  auto numNonReentrantAllocas = CreateNonReentrantAllocaSets();
  statistics->StopCreateNonReentrantAllocaSetsStatistics(numNonReentrantAllocas);

  statistics->StartCreateExternalModRefSet();
  CreateExternalModRefSet();
  statistics->StopCreateExternalModRefSet();

  statistics->StartAnnotationStatistics();
  // Go through and recursively annotate all functions, regions and nodes
  for (const auto & scc : Context_->SccFunctions)
  {
    for (const auto lambda : scc.Items())
    {
      AnnotateFunction(*lambda);
    }
  }
  statistics->StopAnnotationStatistics();

  statistics->StartSolvingStatistics();
  SolveModRefSetConstraintGraph();
  statistics->StopSolvingStatistics();

  // Print debug output
  // std::cerr << PointsToGraph::dumpDot(pointsToGraph) << std::endl;
  // std::cerr << "numSimpleAllocas: " << Context_->SimpleAllocas.Size() << std::endl;
  // std::cerr << "numNonReentrantAllocas: " << numNonReentrantAllocas << std::endl;
  // std::cerr << "Call Graph SCCs:" << std::endl << CallGraphSCCsToString(*this) << std::endl;
  // std::cerr << "RegionTree:" << std::endl
  //           << ToRegionTree(rvsdgModule.Rvsdg(), *ModRefSummary_) << std::endl;

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
  Context_.reset();
  return std::move(ModRefSummary_);
}

/**
 * Collects all lambda nodes defined in the given module, in an unspecified order.
 * @param rvsdgModule the module
 * @return a list of all lambda nodes in the module
 */
static std::vector<const rvsdg::LambdaNode *>
CollectLambdaNodes(const rvsdg::RvsdgModule & rvsdgModule)
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

/**
 * Helper function for checking if the given call is a call to the special setjmp function.
 * @param callNode the node in question
 * @return true if the call is a direct call to setjmp, false otherwise
 */
static bool
isSetjmpCall(const rvsdg::SimpleNode & callNode)
{
  const auto classification = llvm::CallOperation::ClassifyCall(callNode);
  if (!classification->IsExternalCall())
    return false;

  const auto & regionArgument = classification->GetImport();
  if (const auto graphImport = dynamic_cast<const llvm::GraphImport *>(&regionArgument))
  {
    // In C and C++, setjmp is a macro to some underlying function. Clang uses _setjmp
    return graphImport->Name() == "_setjmp";
  }

  return false;
}

void
RegionAwareModRefSummarizer::createCallGraph(const rvsdg::RvsdgModule & rvsdgModule)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

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

  // Outgoing edges for each node in the call graph, indexed by position in lambdaNodes
  std::vector<util::HashSet<size_t>> callGraphSuccessors(numCallGraphNodes);

  // Track the nodes in the call graph that contain direct calls to setjmp
  util::HashSet<size_t> nodesCallingSetjmp;

  // Add outgoing edges from the given caller to any function the call may target
  const auto handleCall = [&](const rvsdg::SimpleNode & callNode, size_t callerIndex) -> void
  {
    JLM_ASSERT(is<CallOperation>(&callNode));
    if (isSetjmpCall(callNode))
    {
      // This function is calling setjmp, so only add it to the set of functions on setjmp stacks
      nodesCallingSetjmp.insert(callerIndex);
      return;
    }

    const auto target = callNode.input(0)->origin();
    const auto targetPtgNode = pointsToGraph.getNodeForRegister(*target);

    // Go through all locations the called function pointer may target
    for (const auto calleePtgNode : pointsToGraph.getExplicitTargets(targetPtgNode).Items())
    {
      const auto kind = pointsToGraph.getNodeKind(calleePtgNode);
      if (kind == PointsToGraph::NodeKind::LambdaNode)
      {
        const auto & lambdaNode = pointsToGraph.getLambdaForNode(calleePtgNode);

        // Look up which call graph node represents the target lambda
        JLM_ASSERT(callGraphNodeIndex.find(&lambdaNode) != callGraphNodeIndex.end());
        const auto calleeCallGraphNode = callGraphNodeIndex[&lambdaNode];

        // Add the edge caller -> callee to the call graph
        callGraphSuccessors[callerIndex].insert(calleeCallGraphNode);
      }
      else if (kind == PointsToGraph::NodeKind::ImportNode)
      {
        // Add the edge caller -> node representing external functions
        callGraphSuccessors[callerIndex].insert(externalNodeIndex);
      }
    }

    if (pointsToGraph.isTargetingAllExternallyAvailable(targetPtgNode))
    {
      // If the call target pointer is flagged, add an edge to external functions
      callGraphSuccessors[callerIndex].insert(externalNodeIndex);
    }
  };

  // Recursive function finding all call operations, adding edges to the call graph
  const std::function<void(const rvsdg::Region &, size_t)> handleCalls =
      [&](const rvsdg::Region & region, size_t callerIndex) -> void
  {
    for (auto & node : region.Nodes())
    {
      if (const auto [callNode, callOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<CallOperation>(node);
          callOp)
      {
        handleCall(*callNode, callerIndex);
      }

      rvsdg::MatchType(
          node,
          [&](const rvsdg::StructuralNode & structural)
          {
            for (auto & subregion : structural.Subregions())
            {
              handleCalls(subregion, callerIndex);
            }
          });
    }
  };

  // For all functions, visit all their calls and add outgoing edges in the call graph
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    handleCalls(*lambdaNodes[i]->subregion(), i);

    // If the function has escaped, add an edge from the node representing all external functions
    if (pointsToGraph.isExternallyAvailable(pointsToGraph.getNodeForLambda(*lambdaNodes[i])))
    {
      callGraphSuccessors[externalNodeIndex].insert(i);
    }
  }

  // Finally, add the fact that the external node may call itself
  callGraphSuccessors[externalNodeIndex].insert(externalNodeIndex);

  // Go through the call graph and mark all functions that can be called while a setjmp is active.
  // Calls via external functions do not count, so we skip going through that node.
  std::queue<size_t> onSetjmpStackQueue;
  for (const auto function : nodesCallingSetjmp.Items())
    onSetjmpStackQueue.push(function);
  while (!onSetjmpStackQueue.empty())
  {
    const auto functionNodeIndex = onSetjmpStackQueue.front();
    onSetjmpStackQueue.pop();

    // Mark the LambdaNode* itself as possibly being on a setjmp stack
    Context_->FunctionsInActiveSetjmp.insert(lambdaNodes[functionNodeIndex]);

    // Go through all successors and mark them as on a setjmp stack if not already marked
    for (const auto calleeNodeIndex : callGraphSuccessors[functionNodeIndex].Items())
    {
      if (calleeNodeIndex == externalNodeIndex)
        continue;
      if (nodesCallingSetjmp.insert(calleeNodeIndex))
        onSetjmpStackQueue.push(calleeNodeIndex);
    }
  }

  // Used by the implementation of Tarjan's SCC algorithm
  const auto getSuccessors = [&](size_t nodeIndex)
  {
    return callGraphSuccessors[nodeIndex].Items();
  };

  // Find SCCs in the call graph
  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSCCs = util::FindStronglyConnectedComponents<size_t>(
      numCallGraphNodes,
      getSuccessors,
      sccIndex,
      reverseTopologicalOrder);

  // sccIndex are distributed in a reverse topological order, so the sccIndex is used
  // when creating the list of SCCs and the functions they contain
  Context_->SccFunctions.resize(numSCCs);
  for (size_t i = 0; i < lambdaNodes.size(); i++)
  {
    Context_->SccFunctions[sccIndex[i]].insert(lambdaNodes[i]);
    Context_->FunctionToSccIndex[lambdaNodes[i]] = sccIndex[i];
  }

  // Add edges between the SCCs for all calls
  Context_->SccCallTargets.resize(numSCCs);
  for (size_t i = 0; i < numCallGraphNodes; i++)
  {
    for (auto target : callGraphSuccessors[i].Items())
    {
      Context_->SccCallTargets[sccIndex[i]].insert(sccIndex[target]);
    }
  }

  // Also note which SCC contains all external functions
  Context_->ExternalNodeSccIndex = sccIndex[externalNodeIndex];
}

size_t
RegionAwareModRefSummarizer::findSccsThatCanCallExternal()
{
  const auto numSccs = Context_->SccCallTargets.size();

  // Initially, only the SCC containing external can call external
  Context_->SccCanCallExternal.resize(numSccs);
  Context_->SccCanCallExternal[Context_->ExternalNodeSccIndex] = true;
  size_t numSccsCanCallExternal = 1;

  // Traverse the SCCs in reverse topological order,
  // and check if any of the successors can reach external
  for (size_t sccIndex = 0; sccIndex < numSccs; sccIndex++)
  {
    if (Context_->SccCanCallExternal[sccIndex])
      continue;

    for (auto targetScc : Context_->SccCallTargets[sccIndex].Items())
    {
      // The target should already have been processed at this point
      JLM_ASSERT(targetScc <= sccIndex);
      if (Context_->SccCanCallExternal[targetScc])
      {
        numSccsCanCallExternal++;
        Context_->SccCanCallExternal[sccIndex] = true;
        break;
      }
    }
  }

  return numSccsCanCallExternal;
}

void
RegionAwareModRefSummarizer::FindAllocasDeadInSccs()
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // First find which allocas may be live in each SCC
  std::vector<util::HashSet<PointsToGraph::NodeIndex>> liveAllocas(Context_->SccFunctions.size());

  util::HashSet<PointsToGraph::NodeIndex> allAllocas;

  // Add all Allocas to the SCC of the function they are defined in
  for (auto allocaPtgNode : Context_->pointsToGraph.allocaNodes())
  {
    allAllocas.insert(allocaPtgNode);
    const auto & allocaNode = pointsToGraph.getAllocaForNode(allocaPtgNode);
    const auto & lambdaNode = rvsdg::getSurroundingLambdaNode(allocaNode);
    JLM_ASSERT(Context_->FunctionToSccIndex.count(&lambdaNode));
    const auto sccIndex = Context_->FunctionToSccIndex[&lambdaNode];
    liveAllocas[sccIndex].insert(allocaPtgNode);
  }

  // Propagate live allocas to targets of function calls.
  // I.e., if a() -> b(), then any alloca that is live in a() may also be live in b()
  // Start at the topologically earliest SCC, which has the largest SCC index
  // The topologically latest SCC can't have any targets
  for (size_t sccIndex = Context_->SccFunctions.size() - 1; sccIndex > 0; sccIndex--)
  {
    for (auto targetScc : Context_->SccCallTargets[sccIndex].Items())
    {
      JLM_ASSERT(targetScc <= sccIndex);
      if (targetScc != sccIndex)
        liveAllocas[targetScc].UnionWith(liveAllocas[sccIndex]);
    }
  }

  // Any alloca that is not live in the SCC gets added to the set of allocas that are dead
  Context_->AllocasDeadInScc.resize(Context_->SccFunctions.size());
  for (size_t sccIndex = 0; sccIndex < Context_->SccFunctions.size(); sccIndex++)
  {
    Context_->AllocasDeadInScc[sccIndex].UnionWith(allAllocas);
    Context_->AllocasDeadInScc[sccIndex].DifferenceWith(liveAllocas[sccIndex]);
  }
}

util::HashSet<PointsToGraph::NodeIndex>
RegionAwareModRefSummarizer::CreateSimpleAllocaSet(const PointsToGraph & pointsToGraph)
{
  // The set of allocas that are simple. Starts off as an over-approximation
  util::HashSet<PointsToGraph::NodeIndex> simpleAllocas;
  // A queue used to visit all PtG memory nodes that are not simple allocas
  std::queue<PointsToGraph::NodeIndex> notSimple;

  for (PointsToGraph::NodeIndex ptgNode = 0; ptgNode < pointsToGraph.numNodes(); ptgNode++)
  {
    // Only memory nodes are relevant
    if (!pointsToGraph.isMemoryNode(ptgNode))
      continue;

    // Allocas that are not externally available start of as presumed simple
    if (pointsToGraph.getNodeKind(ptgNode) == PointsToGraph::NodeKind::AllocaNode
        && !pointsToGraph.isExternallyAvailable(ptgNode))
      simpleAllocas.insert(ptgNode);
    else
      notSimple.push(ptgNode);
  }

  // Process the queue to visit all memory nodes that may disqualify allocas from being simple
  while (!notSimple.empty())
  {
    const auto ptgNode = notSimple.front();
    notSimple.pop();

    // Any node targeted by the not-simple memory node can themselves not be simple
    for (const auto targetPtgNode : pointsToGraph.getExplicitTargets(ptgNode).Items())
    {
      // If the target is currently in the simple allocas candiate set, move it to the queue
      if (simpleAllocas.Remove(targetPtgNode))
        notSimple.push(targetPtgNode);
    }
  }

  return simpleAllocas;
}

util::HashSet<PointsToGraph::NodeIndex>
RegionAwareModRefSummarizer::GetSimpleAllocasReachableFromRegionArguments(
    const rvsdg::Region & region)
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // Use a queue and a set to traverse the PointsToGraph
  util::HashSet<PointsToGraph::NodeIndex> reachableSimpleAllocas;
  std::queue<PointsToGraph::NodeIndex> nodes;
  for (auto argument : region.Arguments())
  {
    if (!IsPointerCompatible(*argument))
      continue;
    const auto ptgNode = pointsToGraph.getNodeForRegister(*argument);
    nodes.push(ptgNode);
  }

  // Traverse along PointsToGraph edges to find all reachable simple allocas
  while (!nodes.empty())
  {
    const auto ptgNode = nodes.front();
    nodes.pop();

    for (const auto targetPtgNode : pointsToGraph.getExplicitTargets(ptgNode).Items())
    {
      // We only are about following simple allocas, as simple allocas are only reachable from them.
      if (!Context_->SimpleAllocas.Contains(targetPtgNode))
        continue;

      if (reachableSimpleAllocas.insert(targetPtgNode))
        nodes.push(targetPtgNode);
    }
  }

  return reachableSimpleAllocas;
}

bool
RegionAwareModRefSummarizer::IsRecursionPossible(const rvsdg::LambdaNode & lambda) const
{
  const auto scc = Context_->FunctionToSccIndex[&lambda];
  return Context_->SccCallTargets[scc].Contains(scc);
}

size_t
RegionAwareModRefSummarizer::CreateNonReentrantAllocaSets()
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  // Caching the sets of simple allocas reachable from region arguments
  std::unordered_map<const rvsdg::Region *, util::HashSet<PointsToGraph::NodeIndex>>
      reachableSimpleAllocas;

  // Returns the set of simple allocas reachable from the region's arguments
  const auto getReachableSimpleAllocas =
      [&](const rvsdg::Region & region) -> const util::HashSet<PointsToGraph::NodeIndex> &
  {
    if (const auto it = reachableSimpleAllocas.find(&region); it != reachableSimpleAllocas.end())
    {
      return it->second;
    }
    return reachableSimpleAllocas[&region] = GetSimpleAllocasReachableFromRegionArguments(region);
  };

  // Checks if the simple alloca represented by the given points-to graph node is non-reentrant
  const auto isNonReentrant = [&](PointsToGraph::NodeIndex simpleAllocaPtgNode) -> bool
  {
    auto & allocaNode = pointsToGraph.getAllocaForNode(simpleAllocaPtgNode);
    const auto & region = *allocaNode.region();

    // If the alloca's function is never involved in any recursion,
    // the alloca is trivially non-reentrant.
    const auto & lambda = getSurroundingLambdaNode(allocaNode);
    if (!IsRecursionPossible(lambda))
      return true;

    // In lambdas where recursion is possible, simple allocas that are reachable from
    // region arguments via edges in the points-to graph must be considered reentrant.
    if (getReachableSimpleAllocas(region).Contains(simpleAllocaPtgNode))
      return false;

    // Otherwise the simple alloca is non-reentrant
    return true;
  };

  size_t numNonReentrantAllocas = 0;

  // Only simple allocas are candidates for being non-reentrant
  for (auto simpleAllocaPtgNode : Context_->SimpleAllocas.Items())
  {
    if (isNonReentrant(simpleAllocaPtgNode))
    {
      const auto region = pointsToGraph.getAllocaForNode(simpleAllocaPtgNode).region();
      // Creates a set for the region if it does not already have one, and add the alloca
      Context_->NonReentrantAllocas[region].insert(simpleAllocaPtgNode);
      numNonReentrantAllocas++;
    }
  }

  return numNonReentrantAllocas;
}

void
RegionAwareModRefSummarizer::CreateExternalModRefSet()
{
  const auto & pointsToGraph = Context_->pointsToGraph;

  Context_->ExternalModRefIndex = ModRefSummary_->CreateModRefSet();

  // Go through all types of memory node and add them to the external ModRefSet if escaping
  for (PointsToGraph::NodeIndex ptgNode = 0; ptgNode < pointsToGraph.numNodes(); ptgNode++)
  {
    // Must be a memory node
    if (!pointsToGraph.isMemoryNode(ptgNode))
      continue;

    // Must be externally available
    if (!pointsToGraph.isExternallyAvailable(ptgNode))
      continue;

    // Must be non-constant
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(ptgNode))
      continue;

    ModRefSummary_->AddToModRefSet(Context_->ExternalModRefIndex, ptgNode);
  }
}

void
RegionAwareModRefSummarizer::AddModRefSimpleConstraint(ModRefSetIndex from, ModRefSetIndex to)
{
  // Ensure the constraint vector is large enough
  Context_->ModRefSetSimpleConstraints.resize(ModRefSummary_->NumModRefSets());
  Context_->ModRefSetSimpleConstraints[from].insert(to);
}

void
RegionAwareModRefSummarizer::AddModRefSetBlocklist(
    ModRefSetIndex index,
    const util::HashSet<PointsToGraph::NodeIndex> & blocklist)
{
  JLM_ASSERT(Context_->ModRefSetBlocklists.find(index) == Context_->ModRefSetBlocklists.end());
  Context_->ModRefSetBlocklists[index] = &blocklist;
}

void
RegionAwareModRefSummarizer::AnnotateFunction(const rvsdg::LambdaNode & lambda)
{
  const auto & region = *lambda.subregion();
  const auto regionModRefSet = AnnotateRegion(region, lambda);
  const auto lambdaModRefSet = ModRefSummary_->GetOrCreateSetForNode(lambda);
  AddModRefSimpleConstraint(regionModRefSet, lambdaModRefSet);
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateRegion(
    const rvsdg::Region & region,
    const rvsdg::LambdaNode & lambda)
{
  const auto regionModRefSet = ModRefSummary_->CreateModRefSet();

  for (auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          const auto nodeModRefSet = AnnotateStructuralNode(structuralNode, lambda);
          AddModRefSimpleConstraint(nodeModRefSet, regionModRefSet);
        },
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          if (const auto nodeModRefSet = AnnotateSimpleNode(simpleNode, lambda))
            AddModRefSimpleConstraint(*nodeModRefSet, regionModRefSet);
        });
  }

  // Check if this region has any non-reentrant allocas. If so, block them
  if (const auto it = Context_->NonReentrantAllocas.find(&region);
      it != Context_->NonReentrantAllocas.end() && ENABLE_NON_REENTRANT_ALLOCA_BLOCKLIST)
  {
    JLM_ASSERT(ModRefSummary_->GetModRefSet(regionModRefSet).IsEmpty());
    AddModRefSetBlocklist(regionModRefSet, it->second);
  }

  return regionModRefSet;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateStructuralNode(
    const rvsdg::StructuralNode & structuralNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRefSet = ModRefSummary_->GetOrCreateSetForNode(structuralNode);

  for (auto & subregion : structuralNode.Subregions())
  {
    const auto subregionModRefSef = AnnotateRegion(subregion, lambda);
    AddModRefSimpleConstraint(subregionModRefSef, nodeModRefSet);
  }

  return nodeModRefSet;
}

std::optional<ModRefSetIndex>
RegionAwareModRefSummarizer::AnnotateSimpleNode(
    const rvsdg::SimpleNode & simpleNode,
    const rvsdg::LambdaNode & lambda)
{
  if (is<LoadOperation>(&simpleNode))
    return AnnotateLoad(simpleNode, lambda);

  if (is<StoreOperation>(&simpleNode))
    return AnnotateStore(simpleNode, lambda);

  if (is<AllocaOperation>(&simpleNode))
    return AnnotateAlloca(simpleNode);

  if (is<MallocOperation>(&simpleNode))
    return AnnotateMalloc(simpleNode);

  if (is<FreeOperation>(&simpleNode))
    return AnnotateFree(simpleNode, lambda);

  if (is<MemCpyOperation>(&simpleNode))
    return AnnotateMemcpy(simpleNode, lambda);

  if (is<CallOperation>(&simpleNode))
    return AnnotateCall(simpleNode, lambda);

  if (is<MemoryStateOperation>(&simpleNode))
  {
    // MemoryStateOperations are only used to route memory states, and can be ignored
    return std::nullopt;
  }

  // Any remaining type of node should not involve any memory states
  JLM_ASSERT(!hasMemoryState(simpleNode));

  return std::nullopt;
}

void
RegionAwareModRefSummarizer::AddPointerOriginTargets(
    ModRefSetIndex modRefSetIndex,
    const rvsdg::Output & origin,
    std::optional<size_t> minTargetSize,
    const rvsdg::LambdaNode & lambda)
{
  const auto & pointsToGraph = Context_->pointsToGraph;
  const auto & allocasDead = Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]];

  // TODO: Re-use ModRefSets for all uses of the registerNode in this function
  const auto registerPtgNode = pointsToGraph.getNodeForRegister(origin);

  const auto tryAddToModRefSet = [&](PointsToGraph::NodeIndex targetPtgNode)
  {
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && pointsToGraph.isNodeConstant(targetPtgNode))
      return;
    if (ENABLE_OPERATION_SIZE_BLOCKING && minTargetSize)
    {
      const auto targetSize = pointsToGraph.tryGetNodeSize(targetPtgNode);
      if (targetSize && *targetSize < minTargetSize)
        return;
    }
    if (ENABLE_DEAD_ALLOCA_BLOCKLIST && allocasDead.Contains(targetPtgNode))
      return;
    ModRefSummary_->AddToModRefSet(modRefSetIndex, targetPtgNode);
  };

  // If the pointer is targeting everything external, add it all to the Mod/Ref set
  if (pointsToGraph.isTargetingAllExternallyAvailable(registerPtgNode))
  {
    for (const auto targetPtgNode : pointsToGraph.getExternallyAvailableNodes())
    {
      tryAddToModRefSet(targetPtgNode);
    }
  }

  for (const auto targetPtgNode : pointsToGraph.getExplicitTargets(registerPtgNode).Items())
  {
    tryAddToModRefSet(targetPtgNode);
  }
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateLoad(
    const rvsdg::SimpleNode & loadNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(loadNode);
  const auto origin = LoadOperation::AddressInput(loadNode).origin();
  const auto loadOperation = util::assertedCast<const LoadOperation>(&loadNode.GetOperation());
  const auto loadSize = GetTypeStoreSize(*loadOperation->GetLoadedType());

  AddPointerOriginTargets(nodeModRef, *origin, loadSize, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateStore(
    const rvsdg::SimpleNode & storeNode,
    const rvsdg::LambdaNode & lambda)
{
  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(storeNode);
  const auto origin = StoreOperation::AddressInput(storeNode).origin();
  const auto storeOperation = util::assertedCast<const StoreOperation>(&storeNode.GetOperation());
  const auto storeSize = GetTypeStoreSize(storeOperation->GetStoredType());

  AddPointerOriginTargets(nodeModRef, *origin, storeSize, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateAlloca(const rvsdg::SimpleNode & allocaNode)
{
  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(allocaNode);
  const auto allocaMemoryNode = Context_->pointsToGraph.getNodeForAlloca(allocaNode);
  ModRefSummary_->AddToModRefSet(nodeModRef, allocaMemoryNode);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMalloc(const rvsdg::SimpleNode & mallocNode)
{
  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(mallocNode);
  const auto mallocMemoryNode = Context_->pointsToGraph.getNodeForMalloc(mallocNode);
  ModRefSummary_->AddToModRefSet(nodeModRef, mallocMemoryNode);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateFree(
    const rvsdg::SimpleNode & freeNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<FreeOperation>(&freeNode));

  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(freeNode);
  const auto origin = FreeOperation::addressInput(freeNode).origin();

  // TODO: Only free MallocMemoryNodes
  AddPointerOriginTargets(nodeModRef, *origin, std::nullopt, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMemcpy(
    const rvsdg::SimpleNode & memcpyNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<MemCpyOperation>(&memcpyNode));

  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(memcpyNode);
  const auto dstOrigin = MemCpyOperation::destinationInput(memcpyNode).origin();
  const auto srcOrigin = MemCpyOperation::sourceInput(memcpyNode).origin();
  const auto countOrigin = MemCpyOperation::countInput(memcpyNode).origin();
  const auto count = tryGetConstantSignedInteger(*countOrigin);
  AddPointerOriginTargets(nodeModRef, *dstOrigin, count, lambda);
  AddPointerOriginTargets(nodeModRef, *srcOrigin, count, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateCall(
    const rvsdg::SimpleNode & callNode,
    const rvsdg::LambdaNode & lambda)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  const auto & pointsToGraph = Context_->pointsToGraph;

  // This ModRefSet represents everything the call may affect
  const auto callModRef = ModRefSummary_->GetOrCreateSetForNode(callNode);

  // Go over all possible targets of the call and add them to the call summary
  const auto targetPtr = callNode.input(0)->origin();
  const auto targetPtgNode = Context_->pointsToGraph.getNodeForRegister(*targetPtr);

  // Is it possible for this call to, either directly or indirectly, call into external functions
  bool canCallExternalFunction = false;

  // Go through all locations the called function pointer may target
  for (const auto calleePtgNode : pointsToGraph.getExplicitTargets(targetPtgNode).Items())
  {
    const auto kind = pointsToGraph.getNodeKind(calleePtgNode);
    if (kind == PointsToGraph::NodeKind::LambdaNode)
    {
      const auto & calleeLambda = pointsToGraph.getLambdaForNode(calleePtgNode);
      const auto targetModRefSet = ModRefSummary_->GetOrCreateSetForNode(calleeLambda);
      AddModRefSimpleConstraint(targetModRefSet, callModRef);

      const auto targetScc = Context_->FunctionToSccIndex[&calleeLambda];
      canCallExternalFunction |= Context_->SccCanCallExternal[targetScc];
    }
    else if (kind == PointsToGraph::NodeKind::ImportNode)
    {
      AddModRefSimpleConstraint(Context_->ExternalModRefIndex, callModRef);
      canCallExternalFunction = true;
    }
  }
  if (pointsToGraph.isTargetingAllExternallyAvailable(targetPtgNode))
  {
    AddModRefSimpleConstraint(Context_->ExternalModRefIndex, callModRef);
    canCallExternalFunction = true;
  }

  // Allocas that are live within the call, might no longer be live from the call site
  if (ENABLE_DEAD_ALLOCA_BLOCKLIST)
  {
    JLM_ASSERT(ModRefSummary_->GetModRefSet(callModRef).IsEmpty());
    AddModRefSetBlocklist(
        callModRef,
        Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]]);
  }

  // If we are currently within an active setjmp stack, and the call we are making might end up
  // calling external functions (such as a longjmp), the function call might return to a
  // completely different place on the call stack.
  // We must therefore conservatively assume that this call must be sequentialized with stores.
  const bool possiblyActiveSetjmp = Context_->FunctionsInActiveSetjmp.Contains(&lambda);
  if (canCallExternalFunction && possiblyActiveSetjmp)
  {
    // Get the Mod/Ref set for the calling function, propagate it to the Mod/Ref of the call
    const auto lambdaModRefSet = ModRefSummary_->GetOrCreateSetForNode(lambda);
    AddModRefSimpleConstraint(lambdaModRefSet, callModRef);
  }

  return callModRef;
}

void
RegionAwareModRefSummarizer::SolveModRefSetConstraintGraph()
{
  Context_->ModRefSetSimpleConstraints.resize(ModRefSummary_->NumModRefSets());
  util::TwoPhaseLrfWorklist<ModRefSetIndex> worklist;

  // Start by pushing everything to the worklist
  for (ModRefSetIndex i = 0; i < ModRefSummary_->NumModRefSets(); i++)
    worklist.PushWorkItem(i);

  while (worklist.HasMoreWorkItems())
  {
    const auto workItem = worklist.PopWorkItem();

    // Handle all simple constraints workItem -> target
    for (auto target : Context_->ModRefSetSimpleConstraints[workItem].Items())
    {
      bool changed = false;
      if (auto blocklist = Context_->ModRefSetBlocklists.find(target);
          blocklist != Context_->ModRefSetBlocklists.end())
      {
        // The target has a blocklist, avoid propagating blocked MemoryNodes
        for (auto memoryNode : ModRefSummary_->GetModRefSet(workItem).Items())
        {
          if (blocklist->second->Contains(memoryNode))
            continue;
          changed |= ModRefSummary_->AddToModRefSet(target, memoryNode);
        }
      }
      else
      {
        // The target has no blocklist, propagate everything
        changed |= ModRefSummary_->PropagateModRefSet(workItem, target);
      }

      if (changed)
        worklist.PushWorkItem(target);
    }
  }

  JLM_ASSERT(VerifyBlocklists());
}

bool
RegionAwareModRefSummarizer::VerifyBlocklists() const
{
  // For all ModRefSets where a blocklist has been defined,
  // check that none of its MemoryNodes are on the blocklist
  for (auto [index, blocklist] : Context_->ModRefSetBlocklists)
  {
    for (auto memoryNode : ModRefSummary_->GetModRefSet(index).Items())
    {
      if (blocklist->Contains(memoryNode))
        return false;
    }
  }
  return true;
}

std::string
RegionAwareModRefSummarizer::CallGraphSCCsToString(const RegionAwareModRefSummarizer & summarizer)
{
  std::ostringstream ss;
  for (size_t i = 0; i < summarizer.Context_->SccFunctions.size(); i++)
  {
    if (i != 0)
      ss << " <- ";
    ss << "[" << std::endl;
    if (i == summarizer.Context_->ExternalNodeSccIndex)
    {
      ss << "  " << "<external>" << std::endl;
    }
    for (auto function : summarizer.Context_->SccFunctions[i].Items())
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
  const auto & pointsToGraph = modRefSummary.GetPointsToGraph();

  std::ostringstream ss;

  auto toString = [&](const util::HashSet<PointsToGraph::NodeIndex> & memoryNodes)
  {
    ss << "MemoryNodes: {";
    for (auto & memoryNode : memoryNodes.Items())
    {
      ss << pointsToGraph.getNodeDebugString(memoryNode);
      ss << ", ";
    }
    ss << "}" << std::endl;
  };

  auto indent = [&](size_t depth, char c = '-')
  {
    for (size_t i = 0; i < depth; i++)
      ss << c;
  };

  std::function<void(const rvsdg::Node &, size_t)> toRegionTree =
      [&](const rvsdg::Node & node, size_t depth)
  {
    if (!modRefSummary.HasSetForNode(node))
      return;

    auto modRefIndex = modRefSummary.GetSetForNode(node);
    auto & memoryNodes = modRefSummary.GetModRefSet(modRefIndex);
    ss << " ";
    toString(memoryNodes);

    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (auto & region : structuralNode->Subregions())
      {
        indent(depth + 1, '-');
        ss << "region" << std::endl;
        for (auto & n : region.Nodes())
          toRegionTree(n, depth + 2);
      }
    }
  };

  for (auto & node : rvsdg.GetRootRegion().Nodes())
    toRegionTree(node, 0);

  return ss.str();
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

}
