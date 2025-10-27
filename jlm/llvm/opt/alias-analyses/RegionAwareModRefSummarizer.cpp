/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
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
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/TarjanScc.hpp>
#include <jlm/util/Worklist.hpp>

#include <queue>

namespace jlm::llvm::aa
{

/**
 * allocas that are not defined in f(), and not defined in a predecessors of f() in the call graph,
 * can not be live inside f(). They are added to the DeadAllocaBlocklist.
 */
static const bool ENABLE_DEAD_ALLOCA_BLOCKLIST = !std::getenv("JLM_DISABLE_DEAD_ALLOCA_BLOCKLIST");

/**
 * In a region with an alloca definition, the MemoryNode representing the alloca does not need to
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

  static constexpr auto CallGraphTimer_ = "CallGraphTimer";
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
    AddMeasurement(Label::NumPointsToGraphMemoryNodes, pointsToGraph.NumMemoryNodes());
  }

  void
  StartCallGraphStatistics()
  {
    AddTimer(CallGraphTimer_).start();
  }

  void
  StopCallGraphStatistics(size_t numSccs)
  {
    GetTimer(CallGraphTimer_).stop();
    AddMeasurement(NumCallGraphSccs_, numSccs);
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

  [[nodiscard]] util::HashSet<const PointsToGraph::MemoryNode *> &
  GetMemoryNodes()
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetMemoryNodes() const
  {
    return MemoryNodes_;
  }

private:
  util::HashSet<const PointsToGraph::MemoryNode *> MemoryNodes_;
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

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetModRefSet(ModRefSetIndex index) const
  {
    JLM_ASSERT(index < ModRefSets_.size());
    return ModRefSets_[index].GetMemoryNodes();
  }

  bool
  AddToModRefSet(ModRefSetIndex index, const PointsToGraph::MemoryNode & node)
  {
    JLM_ASSERT(index < ModRefSets_.size());
    return ModRefSets_[index].GetMemoryNodes().insert(&node);
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

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const override
  {
    return ModRefSets_[GetSetForNode(node)].GetMemoryNodes();
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaEntryModRef(const rvsdg::GammaNode & gamma) const override
  {
    return ModRefSets_[GetSetForNode(gamma)].GetMemoryNodes();
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaExitModRef(const rvsdg::GammaNode & gamma) const override
  {
    return ModRefSets_[GetSetForNode(gamma)].GetMemoryNodes();
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetThetaModRef(const rvsdg::ThetaNode & theta) const override
  {
    return ModRefSets_[GetSetForNode(theta)].GetMemoryNodes();
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaEntryModRef(const rvsdg::LambdaNode & lambda) const override
  {
    return ModRefSets_[GetSetForNode(lambda)].GetMemoryNodes();
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaExitModRef(const rvsdg::LambdaNode & lambda) const override
  {
    return ModRefSets_[GetSetForNode(lambda)].GetMemoryNodes();
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
  /**
   * The set of functions belonging to each SCC in the call graph.
   * The SCCs are ordered in reverse topological order, so
   * if function a() calls b(), and they are not in the same SCC,
   * the SCC containing a() comes after the SCC containing b().
   *
   * External functions are not included in these sets, see \ref ExternalNodeSccIndex.
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<const rvsdg::LambdaNode *>> SccFunctions;

  /**
   * The index of the SCC in the call graph that represent containing all external functions
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  size_t ExternalNodeSccIndex = 0;

  /**
   * For each SCC in the call graph, the set of SCCs it targets using calls.
   * Since SCCs are ordered in reverse topological order, an SCC never targets higher indices.
   * If there is any possibility of recursion within an SCC, it also targets itself.
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  std::vector<util::HashSet<size_t>> SccCallTargets;

  /**
   * A mapping from functions to the index of the SCC they belong to in the call graph
   *
   * Assigned in \ref CreateCallGraph(). Remains constant after.
   */
  std::unordered_map<const rvsdg::LambdaNode *, size_t> FunctionToSccIndex;

  /**
   * For each SCC, only allocas defined within the SCC, or within a predecessor of the SCC,
   * can possibly be live. All other allocas are considered dead in the SCC.
   *
   * Assigned in \ref FindAllocasDeadInSccs(). Remains constant after.
   */
  std::vector<util::HashSet<const PointsToGraph::MemoryNode *>> AllocasDeadInScc;

  /**
   * The set of all Simple Allocas in the module.
   *
   * Assigned in \ref CreateSimpleAllocaSet(). Remains constant after.
   */
  util::HashSet<const PointsToGraph::MemoryNode *> SimpleAllocas;

  /**
   * For each region, this field contains the set of allocas defined in the region,
   * that have been show to be Non-Reentrant.
   *
   * Assigned in \ref CreateNonReentrantAllocaSets(). Remains constant after.
   */
  std::unordered_map<const rvsdg::Region *, util::HashSet<const PointsToGraph::MemoryNode *>>
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
  std::unordered_map<ModRefSetIndex, const util::HashSet<const PointsToGraph::MemoryNode *> *>
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
  Context_ = std::make_unique<Context>();
  auto statistics = Statistics::Create(rvsdgModule, pointsToGraph);

  statistics->StartCallGraphStatistics();
  CreateCallGraph(rvsdgModule);
  statistics->StopCallGraphStatistics(Context_->SccFunctions.size());

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
  // std::cerr << PointsToGraph::ToDot(pointsToGraph) << std::endl;
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
        callGraphSuccessors[callerIndex].insert(calleeCallGraphNode);
      }
      else if (
          PointsToGraph::Node::Is<PointsToGraph::ExternalMemoryNode>(callee)
          || PointsToGraph::Node::Is<PointsToGraph::ImportNode>(callee))
      {
        // Add the edge caller -> node representing external functions
        callGraphSuccessors[callerIndex].insert(externalNodeIndex);
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
      callGraphSuccessors[externalNodeIndex].insert(i);
    }
  }

  // Finally add the fact that the external node may call itself
  callGraphSuccessors[externalNodeIndex].insert(externalNodeIndex);

  // Used by the implementation of Tarjan's SCC algorithm
  const auto GetSuccessors = [&](size_t nodeIndex)
  {
    return callGraphSuccessors[nodeIndex].Items();
  };

  // Find SCCs in the call graph
  std::vector<size_t> sccIndex;
  std::vector<size_t> reverseTopologicalOrder;
  auto numSCCs = util::FindStronglyConnectedComponents<size_t>(
      numCallGraphNodes,
      GetSuccessors,
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

[[nodiscard]] static const rvsdg::LambdaNode &
GetSurroundingLambdaNode(const rvsdg::Node & node)
{
  auto it = &node;
  while (it)
  {
    if (auto lambda = dynamic_cast<const rvsdg::LambdaNode *>(it))
      return *lambda;
    it = it->region()->node();
  }
  JLM_UNREACHABLE("node was not in a lambda");
}

void
RegionAwareModRefSummarizer::FindAllocasDeadInSccs()
{
  // First find which allocas may be live in each SCC
  std::vector<util::HashSet<const PointsToGraph::MemoryNode *>> liveAllocas(
      Context_->SccFunctions.size());

  util::HashSet<const PointsToGraph::MemoryNode *> allAllocas;

  // Add all Allocas to the SCC of the function they are defined in
  for (auto & allocaNode : ModRefSummary_->GetPointsToGraph().AllocaNodes())
  {
    allAllocas.insert(&allocaNode);
    const auto & lambdaNode = GetSurroundingLambdaNode(allocaNode.GetAllocaNode());
    JLM_ASSERT(Context_->FunctionToSccIndex.count(&lambdaNode));
    const auto sccIndex = Context_->FunctionToSccIndex[&lambdaNode];
    liveAllocas[sccIndex].insert(&allocaNode);
  }

  // Propagate live allocas to targets of function calls.
  // I.e., if a() -> b(), then any alloca that is live in a() may also be live in b()
  // Start at the topologically earliest SCC, which has the largest SCC index
  // The topologically latest SCC can't have any targets
  for (size_t sccIndex = Context_->SccFunctions.size() - 1; sccIndex > 0; sccIndex--)
  {
    for (auto targetScc : Context_->SccCallTargets[sccIndex].Items())
    {
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

util::HashSet<const PointsToGraph::MemoryNode *>
RegionAwareModRefSummarizer::CreateSimpleAllocaSet(const PointsToGraph & pointsToGraph)
{
  // The set of allocas that are simple. Starts off as an over-approximation
  util::HashSet<const PointsToGraph::MemoryNode *> simpleAllocas;
  // A queue used to visit all allocas that have been found to not be simple
  std::queue<const PointsToGraph::MemoryNode *> notSimple;

  const auto OnlyAllocaSources = [](const PointsToGraph::MemoryNode & node)
  {
    // Allocas that have escaped the module are never simple
    if (node.IsModuleEscaping())
      return false;

    for (const auto & source : node.Sources())
    {
      if (!PointsToGraph::Node::Is<PointsToGraph::AllocaNode>(source)
          && !PointsToGraph::Node::Is<PointsToGraph::RegisterNode>(source))
        return false;
    }

    return true;
  };

  for (const auto & allocaNode : pointsToGraph.AllocaNodes())
  {
    if (OnlyAllocaSources(allocaNode))
      simpleAllocas.insert(&allocaNode);
    else
      notSimple.push(&allocaNode);
  }

  // Now all Allocas are either in the simpleAllocas candidate set,
  // or in the notSimple queue. Process the queue until empty
  while (!notSimple.empty())
  {
    const auto & allocaNode = *notSimple.front();
    notSimple.pop();

    // Any node targeted by the allocaNode can not be simple
    for (const auto & target : allocaNode.Targets())
    {
      // If the target is currently in the simple allocas candiate set, move it to the queue
      if (simpleAllocas.Remove(&target))
        notSimple.push(&target);
    }
  }

  return simpleAllocas;
}

/**
 * Checks if it is possible to reach the given \p allocaMemoryNode from an argument of \p region,
 * via zero or more AllocaMemoryNodes in the PointsToGraph.
 * Other types of MemoryNodes are ignored, as they would disqualify the alloca from being simple.
 * @param allocaMemoryNode the AllocaMemoryNode representing a simple alloca
 * @param region the region whose arguments are checked
 * @return true if it was possible to reach the alloca from the region parameters
 */
static bool
IsSimpleAllocaReachableFromRegionArguments(
    const PointsToGraph::AllocaNode & allocaMemoryNode,
    const rvsdg::Region & region)
{
  const auto & pointsToGraph = allocaMemoryNode.Graph();

  // Use a queue and a set to traverse the PointsToGraph
  util::HashSet<const PointsToGraph::Node *> seen;
  std::queue<const PointsToGraph::Node *> nodes;
  for (auto argument : region.Arguments())
  {
    if (!IsPointerCompatible(*argument))
      continue;
    auto & ptgNode = pointsToGraph.GetRegisterNode(*argument);
    nodes.push(&ptgNode);
    seen.insert(&ptgNode);
  }

  // Traverse along PointsToGraph edges to find all reachable allocas
  while (!nodes.empty())
  {
    auto & ptgNode = *nodes.front();
    nodes.pop();

    for (auto & target : ptgNode.Targets())
    {
      if (&target == &allocaMemoryNode)
        return true;

      // We only are about following allocas, as simple allocas are only reachable from them.
      if (!PointsToGraph::Node::Is<PointsToGraph::AllocaNode>(target))
        continue;

      if (seen.insert(&target))
        nodes.push(&target);
    }
  }

  return false;
}

bool
RegionAwareModRefSummarizer::IsRecursionPossible(const rvsdg::LambdaNode & lambda)
{
  const auto scc = Context_->FunctionToSccIndex[&lambda];
  return Context_->SccCallTargets[scc].Contains(scc);
}

size_t
RegionAwareModRefSummarizer::CreateNonReentrantAllocaSets()
{
  size_t numNonReentrantAllocas = 0;

  // Only simple allocas are candidates for being Non-Reentrant
  for (auto memoryNode : Context_->SimpleAllocas.Items())
  {
    auto & allocaMemoryNode = *util::assertedCast<const PointsToGraph::AllocaNode>(memoryNode);
    auto & allocaNode = allocaMemoryNode.GetAllocaNode();
    const auto & region = *allocaNode.region();

    // If the alloca's function is never involved in any recursion
    // the alloca is definitely non-reentrant.
    const auto & lambda = GetSurroundingLambdaNode(allocaNode);

    if (IsRecursionPossible(lambda)
        && IsSimpleAllocaReachableFromRegionArguments(allocaMemoryNode, region))
      continue;

    // Creates a set for the region if it does not already have one, and add the alloca
    Context_->NonReentrantAllocas[&region].insert(&allocaMemoryNode);
    numNonReentrantAllocas++;
  }

  return numNonReentrantAllocas;
}

void
RegionAwareModRefSummarizer::CreateExternalModRefSet()
{
  Context_->ExternalModRefIndex = ModRefSummary_->CreateModRefSet();

  // Go through all types of memory node and add them to the external ModRefSet if escaping
  for (auto & alloca : ModRefSummary_->GetPointsToGraph().AllocaNodes())
  {
    if (!alloca.IsModuleEscaping())
      continue;
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && alloca.isConstant())
      continue;
    ModRefSummary_->AddToModRefSet(Context_->ExternalModRefIndex, alloca);
  }
  for (auto & malloc : ModRefSummary_->GetPointsToGraph().MallocNodes())
  {
    if (!malloc.IsModuleEscaping())
      continue;
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && malloc.isConstant())
      continue;
    ModRefSummary_->AddToModRefSet(Context_->ExternalModRefIndex, malloc);
  }
  for (auto & delta : ModRefSummary_->GetPointsToGraph().DeltaNodes())
  {
    if (!delta.IsModuleEscaping())
      continue;
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && delta.isConstant())
      continue;
    ModRefSummary_->AddToModRefSet(Context_->ExternalModRefIndex, delta);
  }
  for (auto & lambda : ModRefSummary_->GetPointsToGraph().LambdaNodes())
  {
    if (!lambda.IsModuleEscaping())
      continue;

    // Add a call from external to the function
    auto lambdaModRefIndex = ModRefSummary_->GetOrCreateSetForNode(lambda.GetLambdaNode());
    AddModRefSimpleConstraint(lambdaModRefIndex, Context_->ExternalModRefIndex);

    if (ENABLE_CONSTANT_MEMORY_BLOCKING && lambda.isConstant())
      continue;
    ModRefSummary_->AddToModRefSet(Context_->ExternalModRefIndex, lambda);
  }
  for (auto & import : ModRefSummary_->GetPointsToGraph().ImportNodes())
  {
    if (!import.IsModuleEscaping())
      continue;
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && import.isConstant())
      continue;
    ModRefSummary_->AddToModRefSet(Context_->ExternalModRefIndex, import);
  }

  ModRefSummary_->AddToModRefSet(
      Context_->ExternalModRefIndex,
      ModRefSummary_->GetPointsToGraph().GetExternalMemoryNode());
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
    const util::HashSet<const PointsToGraph::MemoryNode *> & blocklist)
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
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      const auto nodeModRefSet = AnnotateStructuralNode(*structuralNode, lambda);
      AddModRefSimpleConstraint(nodeModRefSet, regionModRefSet);
    }
    else if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
    {
      const auto nodeModRefSet = AnnotateSimpleNode(*simpleNode, lambda);
      if (nodeModRefSet)
        AddModRefSimpleConstraint(*nodeModRefSet, regionModRefSet);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }

  // Check if this region has any Non-Reentrant Allocas. If so, block them
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

  return std::nullopt;
}

void
RegionAwareModRefSummarizer::AddPointerOriginTargets(
    ModRefSetIndex modRefSetIndex,
    const rvsdg::Output & origin,
    std::optional<size_t> minTargetSize,
    const rvsdg::LambdaNode & lambda)
{
  // TODO Re-use ModRefSets for all uses of the registerNode in this function
  const auto & registerNode = ModRefSummary_->GetPointsToGraph().GetRegisterNode(origin);

  const auto & allocasDead = Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]];
  for (const auto & target : registerNode.Targets())
  {
    if (ENABLE_CONSTANT_MEMORY_BLOCKING && target.isConstant())
      continue;
    if (ENABLE_OPERATION_SIZE_BLOCKING && minTargetSize)
    {
      const auto targetSize = target.tryGetSize();
      if (targetSize && *targetSize < minTargetSize)
        continue;
    }
    if (ENABLE_DEAD_ALLOCA_BLOCKLIST && allocasDead.Contains(&target))
      continue;

    ModRefSummary_->AddToModRefSet(modRefSetIndex, target);
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
  const auto loadSize = GetTypeSize(*loadOperation->GetLoadedType());

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
  const auto storeSize = GetTypeSize(storeOperation->GetStoredType());

  AddPointerOriginTargets(nodeModRef, *origin, storeSize, lambda);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateAlloca(const rvsdg::SimpleNode & allocaNode)
{
  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(allocaNode);
  const auto & allocaMemoryNode = ModRefSummary_->GetPointsToGraph().GetAllocaNode(allocaNode);
  ModRefSummary_->AddToModRefSet(nodeModRef, allocaMemoryNode);
  return nodeModRef;
}

ModRefSetIndex
RegionAwareModRefSummarizer::AnnotateMalloc(const rvsdg::SimpleNode & mallocNode)
{
  const auto nodeModRef = ModRefSummary_->GetOrCreateSetForNode(mallocNode);
  const auto & mallocMemoryNode = ModRefSummary_->GetPointsToGraph().GetMallocNode(mallocNode);
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
  const auto origin = freeNode.input(0)->origin();
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

  // This ModRefSet represents everything the call may affect
  const auto callModRef = ModRefSummary_->GetOrCreateSetForNode(callNode);

  // Go over all possible targets of the call and add them to the call summary
  const auto targetPtr = callNode.input(0)->origin();
  const auto & targetPtrNode = ModRefSummary_->GetPointsToGraph().GetRegisterNode(*targetPtr);

  // Go through all locations the called function pointer may target
  for (auto & callee : targetPtrNode.Targets())
  {
    if (auto lambdaCallee = dynamic_cast<const PointsToGraph::LambdaNode *>(&callee))
    {
      const auto & lambdaNode = lambdaCallee->GetLambdaNode();
      const auto targetModRefSet = ModRefSummary_->GetOrCreateSetForNode(lambdaNode);
      AddModRefSimpleConstraint(targetModRefSet, callModRef);
    }
    else if (
        PointsToGraph::Node::Is<PointsToGraph::ExternalMemoryNode>(callee)
        || PointsToGraph::Node::Is<PointsToGraph::ImportNode>(callee))
    {
      AddModRefSimpleConstraint(Context_->ExternalModRefIndex, callModRef);
    }
  }

  // Allocas that are live within the call, might no longer be live from the call site
  if (ENABLE_DEAD_ALLOCA_BLOCKLIST)
  {
    JLM_ASSERT(ModRefSummary_->GetModRefSet(callModRef).IsEmpty());
    AddModRefSetBlocklist(
        callModRef,
        Context_->AllocasDeadInScc[Context_->FunctionToSccIndex[&lambda]]);
  }

  return callModRef;
}

void
RegionAwareModRefSummarizer::SolveModRefSetConstraintGraph()
{
  Context_->ModRefSetSimpleConstraints.resize(ModRefSummary_->NumModRefSets());
  util::FifoWorklist<ModRefSetIndex> worklist;

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
          changed |= ModRefSummary_->AddToModRefSet(target, *memoryNode);
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
  std::ostringstream ss;

  auto toString = [&](const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    ss << "MemoryNodes: {";
    for (auto & memoryNode : memoryNodes.Items())
    {
      ss << memoryNode->DebugString();
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
