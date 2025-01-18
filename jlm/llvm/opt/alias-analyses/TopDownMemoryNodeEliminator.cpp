/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/opt/alias-analyses/MemoryNodeProvider.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownMemoryNodeEliminator.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::llvm::aa
{

/** \brief Collect statistics about TopDownMemoryNodeEliminator pass
 *
 */
class TopDownMemoryNodeEliminator::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(Statistics::Id::TopDownMemoryNodeEliminator, sourceFile)
  {}

  void
  Start(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(Label::Timer).start();
  }

  void
  Stop() noexcept
  {
    GetTimer(Label::Timer).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const jlm::util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

/** \brief Memory node provisioning of TopDownMemoryNodeEliminator
 *
 */
class TopDownMemoryNodeEliminator::Provisioning final : public MemoryNodeProvisioning
{
  using RegionMap =
      std::unordered_map<const rvsdg::Region *, util::HashSet<const PointsToGraph::MemoryNode *>>;
  using CallMap =
      std::unordered_map<const CallNode *, util::HashSet<const PointsToGraph::MemoryNode *>>;

public:
  explicit Provisioning(const PointsToGraph & pointsToGraph)
      : PointsToGraph_(pointsToGraph)
  {}

  Provisioning(const Provisioning &) = delete;

  Provisioning(Provisioning &&) = delete;

  Provisioning &
  operator=(const Provisioning &) = delete;

  Provisioning &
  operator=(Provisioning &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionEntryNodes(const rvsdg::Region & region) const override
  {
    JLM_ASSERT(HasRegionEntryMemoryNodesSet(region));
    return RegionEntryMemoryNodes_.find(&region)->second;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionExitNodes(const rvsdg::Region & region) const override
  {
    JLM_ASSERT(HasRegionExitMemoryNodesSet(region));
    return RegionExitMemoryNodes_.find(&region)->second;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallEntryNodes(const CallNode & callNode) const override
  {
    auto callTypeClassifier = CallNode::ClassifyCall(callNode);

    if (callTypeClassifier->IsNonRecursiveDirectCall()
        || callTypeClassifier->IsRecursiveDirectCall())
    {
      auto & lambdaNode =
          rvsdg::AssertGetOwnerNode<lambda::node>(callTypeClassifier->GetLambdaOutput());
      return GetLambdaEntryNodes(lambdaNode);
    }
    else if (callTypeClassifier->IsExternalCall())
    {
      return GetExternalCallNodesSet(callNode);
    }
    else if (callTypeClassifier->IsIndirectCall())
    {
      return GetIndirectCallNodesSet(callNode);
    }

    JLM_UNREACHABLE("Unhandled call type!");
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetCallExitNodes(const CallNode & callNode) const override
  {
    auto callTypeClassifier = CallNode::ClassifyCall(callNode);

    if (callTypeClassifier->IsNonRecursiveDirectCall()
        || callTypeClassifier->IsRecursiveDirectCall())
    {
      auto & lambdaNode =
          rvsdg::AssertGetOwnerNode<lambda::node>(callTypeClassifier->GetLambdaOutput());
      return GetLambdaExitNodes(lambdaNode);
    }
    else if (callTypeClassifier->IsExternalCall())
    {
      return GetExternalCallNodesSet(callNode);
    }
    else if (callTypeClassifier->IsIndirectCall())
    {
      return GetIndirectCallNodesSet(callNode);
    }

    JLM_UNREACHABLE("Unhandled call type!");
  }

  [[nodiscard]] util::HashSet<const PointsToGraph::MemoryNode *>
  GetOutputNodes(const rvsdg::output & output) const override
  {
    JLM_ASSERT(is<PointerType>(output.type()));
    auto & registerNode = PointsToGraph_.GetRegisterNode(output);

    util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
    for (auto & memoryNode : registerNode.Targets())
      memoryNodes.Insert(&memoryNode);

    return memoryNodes;
  }

  void
  AddRegionEntryNodes(
      const rvsdg::Region & region,
      const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    auto & set = GetOrCreateRegionEntryMemoryNodesSet(region);
    set.UnionWith(memoryNodes);
  }

  void
  AddRegionExitNodes(
      const rvsdg::Region & region,
      const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    auto & set = GetOrCreateRegionExitMemoryNodesSet(region);
    set.UnionWith(memoryNodes);
  }

  void
  AddExternalCallNodes(
      const CallNode & externalCall,
      const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    auto & set = GetOrCreateExternalCallNodesSet(externalCall);
    set.UnionWith(memoryNodes);
  }

  void
  AddIndirectCallNodes(
      const CallNode & indirectCall,
      const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    JLM_ASSERT(CallNode::ClassifyCall(indirectCall)->IsIndirectCall());
    auto & set = GetOrCreateIndirectCallNodesSet(indirectCall);
    set.UnionWith(memoryNodes);
  }

  static std::unique_ptr<Provisioning>
  Create(const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<Provisioning>(pointsToGraph);
  }

private:
  bool
  HasExternalCallNodesSet(const CallNode & externalCall) const noexcept
  {
    return ExternalCallNodes_.find(&externalCall) != ExternalCallNodes_.end();
  }

  bool
  HasIndirectCallNodesSet(const CallNode & indirectCall) const noexcept
  {
    return IndirectCallNodes_.find(&indirectCall) != IndirectCallNodes_.end();
  }

  bool
  HasRegionEntryMemoryNodesSet(const rvsdg::Region & region) const noexcept
  {
    return RegionEntryMemoryNodes_.find(&region) != RegionEntryMemoryNodes_.end();
  }

  bool
  HasRegionExitMemoryNodesSet(const rvsdg::Region & region) const noexcept
  {
    return RegionExitMemoryNodes_.find(&region) != RegionExitMemoryNodes_.end();
  }

  util::HashSet<const PointsToGraph::MemoryNode *> &
  GetOrCreateRegionEntryMemoryNodesSet(const rvsdg::Region & region)
  {
    if (!HasRegionEntryMemoryNodesSet(region))
    {
      RegionEntryMemoryNodes_[&region] = {};
    }

    return RegionEntryMemoryNodes_[&region];
  }

  util::HashSet<const PointsToGraph::MemoryNode *> &
  GetOrCreateRegionExitMemoryNodesSet(const rvsdg::Region & region)
  {
    if (!HasRegionExitMemoryNodesSet(region))
    {
      RegionExitMemoryNodes_[&region] = {};
    }

    return RegionExitMemoryNodes_[&region];
  }

  util::HashSet<const PointsToGraph::MemoryNode *> &
  GetOrCreateExternalCallNodesSet(const CallNode & externalCall)
  {
    if (!HasExternalCallNodesSet(externalCall))
    {
      ExternalCallNodes_[&externalCall] = {};
    }

    return ExternalCallNodes_[&externalCall];
  }

  util::HashSet<const PointsToGraph::MemoryNode *> &
  GetOrCreateIndirectCallNodesSet(const CallNode & indirectCall)
  {
    if (!HasIndirectCallNodesSet(indirectCall))
    {
      IndirectCallNodes_[&indirectCall] = {};
    }

    return IndirectCallNodes_[&indirectCall];
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetExternalCallNodesSet(const CallNode & externalCall) const
  {
    JLM_ASSERT(HasExternalCallNodesSet(externalCall));
    return (*ExternalCallNodes_.find(&externalCall)).second;
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetIndirectCallNodesSet(const CallNode & indirectCall) const
  {
    JLM_ASSERT(HasIndirectCallNodesSet(indirectCall));
    return (*IndirectCallNodes_.find(&indirectCall)).second;
  }

  const PointsToGraph & PointsToGraph_;

  RegionMap RegionEntryMemoryNodes_;
  RegionMap RegionExitMemoryNodes_;

  CallMap ExternalCallNodes_;
  CallMap IndirectCallNodes_;
};

/** \brief Context for TopDownMemoryNodeEliminator
 *
 * This class keeps track of all the required state throughout the transformation.
 *
 */
class TopDownMemoryNodeEliminator::Context final
{
public:
  explicit Context(const MemoryNodeProvisioning & seedProvisioning)
      : SeedProvisioning_(seedProvisioning),
        Provisioning_(Provisioning::Create(seedProvisioning.GetPointsToGraph()))
  {}

  Context(const Context &) = delete;

  Context(Context &&) noexcept = delete;

  Context &
  operator=(const Context &) = delete;

  Context &
  operator=(Context &&) noexcept = delete;

  [[nodiscard]] const MemoryNodeProvisioning &
  GetSeedProvisioning() const noexcept
  {
    return SeedProvisioning_;
  }

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept
  {
    return GetSeedProvisioning().GetPointsToGraph();
  }

  [[nodiscard]] Provisioning &
  GetProvisioning() noexcept
  {
    return *Provisioning_;
  }

  [[nodiscard]] std::unique_ptr<Provisioning>
  ReleaseProvisioning() noexcept
  {
    return std::move(Provisioning_);
  }

  /**
   *
   * @param region The region of interest.
   * @return Return the points-to graph memory nodes that are considered live in \p region.
   */
  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLiveNodes(const rvsdg::Region & region) noexcept
  {
    return GetOrCreateLiveNodesSet(region);
  }

  /**
   * Adds points-to graph memory nodes to the live set of \p region.
   *
   * @param region The region for which to add the memory nodes.
   * @param memoryNodes The memory nodes to add.
   */
  void
  AddLiveNodes(
      const rvsdg::Region & region,
      const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    auto & liveNodes = GetOrCreateLiveNodesSet(region);
    liveNodes.UnionWith(memoryNodes);
  }

  /**
   * Determines whether \p lambdaNode has annotated live nodes.
   *
   * @param lambdaNode The lambda node for which to check.
   * @return True if \p lambdaNode has annotated live nodes, otherwise false.
   */
  bool
  HasAnnotatedLiveNodes(const lambda::node & lambdaNode) const noexcept
  {
    return LiveNodesAnnotatedLambdaNodes_.Contains(&lambdaNode);
  }

  /**
   * Marks \p lambdaNode as having annotated live nodes.
   *
   * @param lambdaNode The lambda node which is marked.
   */
  void
  AddLiveNodesAnnotatedLambda(const lambda::node & lambdaNode)
  {
    LiveNodesAnnotatedLambdaNodes_.Insert(&lambdaNode);
  }

  static std::unique_ptr<Context>
  Create(const MemoryNodeProvisioning & seedProvisioning)
  {
    return std::make_unique<Context>(seedProvisioning);
  }

private:
  bool
  HasLiveNodesSet(const rvsdg::Region & region) const noexcept
  {
    return LiveNodes_.find(&region) != LiveNodes_.end();
  }

  util::HashSet<const PointsToGraph::MemoryNode *> &
  GetOrCreateLiveNodesSet(const rvsdg::Region & region)
  {
    if (!HasLiveNodesSet(region))
    {
      LiveNodes_[&region] = {};
    }

    return LiveNodes_[&region];
  }

  const MemoryNodeProvisioning & SeedProvisioning_;
  std::unique_ptr<Provisioning> Provisioning_;

  // Keeps track of the memory nodes that are live within a region.
  std::unordered_map<const rvsdg::Region *, util::HashSet<const PointsToGraph::MemoryNode *>>
      LiveNodes_;

  // Keeps track of all lambda nodes where we annotated live nodes BEFORE traversing the lambda
  // subregion.
  util::HashSet<const lambda::node *> LiveNodesAnnotatedLambdaNodes_;
};

TopDownMemoryNodeEliminator::~TopDownMemoryNodeEliminator() noexcept = default;

TopDownMemoryNodeEliminator::TopDownMemoryNodeEliminator() = default;

std::unique_ptr<MemoryNodeProvisioning>
TopDownMemoryNodeEliminator::EliminateMemoryNodes(
    const rvsdg::RvsdgModule & rvsdgModule,
    const MemoryNodeProvisioning & seedProvisioning,
    util::StatisticsCollector & statisticsCollector)
{
  Context_ = Context::Create(seedProvisioning);
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->Start(rvsdgModule.Rvsdg());
  EliminateTopDown(rvsdgModule);
  statistics->Stop();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  auto provisioning = Context_->ReleaseProvisioning();
  Context_.reset();

  JLM_ASSERT(CheckInvariants(rvsdgModule, seedProvisioning, *provisioning));

  return provisioning;
}

std::unique_ptr<MemoryNodeProvisioning>
TopDownMemoryNodeEliminator::CreateAndEliminate(
    const rvsdg::RvsdgModule & rvsdgModule,
    const MemoryNodeProvisioning & seedProvisioning,
    util::StatisticsCollector & statisticsCollector)
{
  TopDownMemoryNodeEliminator provider;
  return provider.EliminateMemoryNodes(rvsdgModule, seedProvisioning, statisticsCollector);
}

std::unique_ptr<MemoryNodeProvisioning>
TopDownMemoryNodeEliminator::CreateAndEliminate(
    const rvsdg::RvsdgModule & rvsdgModule,
    const MemoryNodeProvisioning & seedProvisioning)
{
  util::StatisticsCollector statisticsCollector;
  return CreateAndEliminate(rvsdgModule, seedProvisioning, statisticsCollector);
}

void
TopDownMemoryNodeEliminator::EliminateTopDown(const rvsdg::RvsdgModule & rvsdgModule)
{
  // Initialize the memory nodes that are alive at beginning of every tail-lambda
  InitializeLiveNodesOfTailLambdas(rvsdgModule);

  // Start the processing of the RVSDG module
  EliminateTopDownRootRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

void
TopDownMemoryNodeEliminator::EliminateTopDownRootRegion(rvsdg::Region & region)
{
  JLM_ASSERT(region.IsRootRegion() || rvsdg::is<phi::operation>(region.node()));

  // Process the lambda, phi, and delta nodes bottom-up.
  // This ensures that we visit all the call nodes before we visit the respective lambda nodes.
  // The tail-lambdas (lambda nodes without calls in the RVSDG module) have already been visited and
  // initialized by InitializeLiveNodesOfTailLambdas().
  rvsdg::bottomup_traverser traverser(&region);
  for (auto & node : traverser)
  {
    if (auto lambdaNode = dynamic_cast<const lambda::node *>(node))
    {
      EliminateTopDownLambda(*lambdaNode);
    }
    else if (auto phiNode = dynamic_cast<const phi::node *>(node))
    {
      EliminateTopDownPhi(*phiNode);
    }
    else if (dynamic_cast<const delta::node *>(node))
    {
      // Nothing needs to be done.
    }
    else if (
        is<FunctionToPointerOperation>(node->GetOperation())
        || is<PointerToFunctionOperation>(node->GetOperation()))
    {
      // Nothing needs to be done.
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }
}

void
TopDownMemoryNodeEliminator::EliminateTopDownRegion(rvsdg::Region & region)
{
  auto isLambdaSubregion = rvsdg::is<lambda::operation>(region.node());
  auto isThetaSubregion = rvsdg::is<rvsdg::ThetaOperation>(region.node());
  auto isGammaSubregion = rvsdg::is<rvsdg::GammaOperation>(region.node());
  JLM_ASSERT(isLambdaSubregion || isThetaSubregion || isGammaSubregion);

  // Process the intra-procedural nodes top-down.
  // This ensures that we add the live memory nodes to the live sets when the respective RVSDG nodes
  // appear in the visitation.
  rvsdg::topdown_traverser traverser(&region);
  for (auto & node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(node))
    {
      EliminateTopDownSimpleNode(*simpleNode);
    }
    else if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(node))
    {
      EliminateTopDownStructuralNode(*structuralNode);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }
}

void
TopDownMemoryNodeEliminator::EliminateTopDownStructuralNode(
    const rvsdg::StructuralNode & structuralNode)
{
  if (auto gammaNode = dynamic_cast<const rvsdg::GammaNode *>(&structuralNode))
  {
    EliminateTopDownGamma(*gammaNode);
  }
  else if (auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(&structuralNode))
  {
    EliminateTopDownTheta(*thetaNode);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled structural node type!");
  }
}

void
TopDownMemoryNodeEliminator::EliminateTopDownLambda(const lambda::node & lambdaNode)
{
  EliminateTopDownLambdaEntry(lambdaNode);
  EliminateTopDownRegion(*lambdaNode.subregion());
  EliminateTopDownLambdaExit(lambdaNode);
}

void
TopDownMemoryNodeEliminator::EliminateTopDownLambdaEntry(const lambda::node & lambdaNode)
{
  auto & lambdaSubregion = *lambdaNode.subregion();
  auto & provisioning = Context_->GetProvisioning();
  auto & seedProvisioning = Context_->GetSeedProvisioning();

  if (Context_->HasAnnotatedLiveNodes(lambdaNode))
  {
    // Live nodes were annotated. This means that either:
    // 1. This lambda node has direct calls that were already handled due to bottom-up visitation.
    // 2. This lambda is a tail-lambda and live nodes were annotated by
    // InitializeLiveNodesOfTailLambdas()
    auto & liveNodes = Context_->GetLiveNodes(lambdaSubregion);
    provisioning.AddRegionEntryNodes(lambdaSubregion, liveNodes);
  }
  else
  {
    // Live nodes were not annotated. This means that:
    // 1. This lambda has no direct calls (but potentially only indirect calls)
    // 2. This lambda is dead and is not used at all
    //
    // Thus, we have no idea what memory nodes are live at its entry. Thus, we need to be
    // conservative and simply say that all memory nodes from the seed provisioning are live.
    auto & seedLambdaEntryNodes = seedProvisioning.GetLambdaEntryNodes(lambdaNode);
    Context_->AddLiveNodes(lambdaSubregion, seedLambdaEntryNodes);
    provisioning.AddRegionEntryNodes(lambdaSubregion, seedLambdaEntryNodes);
  }
}

void
TopDownMemoryNodeEliminator::EliminateTopDownLambdaExit(const lambda::node & lambdaNode)
{
  auto & lambdaSubregion = *lambdaNode.subregion();
  auto & provisioning = Context_->GetProvisioning();
  auto & seedProvisioning = Context_->GetSeedProvisioning();

  if (Context_->HasAnnotatedLiveNodes(lambdaNode))
  {
    // Live nodes were annotated. This means that either:
    // 1. This lambda node has direct calls that were already handled due to bottom-up visitation.
    // 2. This lambda is a tail-lambda and live nodes were annotated by
    // InitializeLiveNodesOfTailLambdas()
    auto & entryNodes = provisioning.GetLambdaEntryNodes(lambdaNode);
    provisioning.AddRegionExitNodes(lambdaSubregion, entryNodes);
  }
  else
  {
    // Live nodes were not annotated. This means that:
    // 1. This lambda has no direct calls (but potentially only indirect calls)
    // 2. This lambda is dead and is not used at all
    //
    // Thus, we have no idea what memory nodes are live at its entry. Thus, we need to be
    // conservative and simply say that all memory nodes from the seed provisioning are live.
    auto & seedLambdaExitNodes = seedProvisioning.GetLambdaExitNodes(lambdaNode);
    provisioning.AddRegionExitNodes(lambdaSubregion, seedLambdaExitNodes);
  }
}

void
TopDownMemoryNodeEliminator::EliminateTopDownPhi(const phi::node & phiNode)
{
  auto unifyLiveNodes = [&](const rvsdg::Region & phiSubregion)
  {
    std::vector<const lambda::node *> lambdaNodes;
    util::HashSet<const PointsToGraph::MemoryNode *> liveNodes;
    for (auto & node : phiSubregion.Nodes())
    {
      if (auto lambdaNode = dynamic_cast<const lambda::node *>(&node))
      {
        lambdaNodes.emplace_back(lambdaNode);

        auto & lambdaSubregion = *lambdaNode->subregion();
        auto & lambdaLiveNodes = Context_->GetLiveNodes(lambdaSubregion);
        liveNodes.UnionWith(lambdaLiveNodes);
      }
      else if (is<delta::operation>(&node))
      {
        // Nothing needs to be done.
      }
      else
      {
        JLM_UNREACHABLE("Unhandled node Type!");
      }
    }

    for (auto & lambdaNode : lambdaNodes)
    {
      auto & lambdaSubregion = *lambdaNode->subregion();
      Context_->AddLiveNodes(lambdaSubregion, liveNodes);
      Context_->AddLiveNodesAnnotatedLambda(*lambdaNode);
    }
  };

  auto & phiSubregion = *phiNode.subregion();

  // Compute initial live node solution for all lambda nodes in the phi
  EliminateTopDownRootRegion(phiSubregion);
  // Unify the live node sets from all lambda nodes in the phi
  unifyLiveNodes(phiSubregion);
  // Ensure that the unified live node sets are propagated to every lambda
  EliminateTopDownRootRegion(phiSubregion);
}

void
TopDownMemoryNodeEliminator::EliminateTopDownGamma(const rvsdg::GammaNode & gammaNode)
{
  auto addSubregionLiveAndEntryNodes =
      [](const rvsdg::GammaNode & gammaNode, TopDownMemoryNodeEliminator::Context & context)
  {
    auto & gammaRegion = *gammaNode.region();
    auto & seedProvisioning = context.GetSeedProvisioning();
    auto & provisioning = context.GetProvisioning();
    auto & gammaRegionLiveNodes = context.GetLiveNodes(gammaRegion);

    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);

      auto subregionEntryNodes = seedProvisioning.GetRegionEntryNodes(subregion);
      subregionEntryNodes.IntersectWith(gammaRegionLiveNodes);

      context.AddLiveNodes(subregion, subregionEntryNodes);
      provisioning.AddRegionEntryNodes(subregion, subregionEntryNodes);
    }
  };

  auto eliminateTopDownForSubregions = [&](const rvsdg::GammaNode & gammaNode)
  {
    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);
      EliminateTopDownRegion(subregion);
    }
  };

  auto addSubregionExitNodes =
      [](const rvsdg::GammaNode & gammaNode, TopDownMemoryNodeEliminator::Context & context)
  {
    auto & provisioning = context.GetProvisioning();

    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);
      auto & liveNodes = context.GetLiveNodes(subregion);
      provisioning.AddRegionExitNodes(subregion, liveNodes);
    }
  };

  auto updateGammaRegionLiveNodes =
      [](const rvsdg::GammaNode & gammaNode, TopDownMemoryNodeEliminator::Context & context)
  {
    auto & gammaRegion = *gammaNode.region();
    auto & provisioning = context.GetProvisioning();

    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);
      auto & subregionExitNodes = provisioning.GetRegionExitNodes(subregion);
      context.AddLiveNodes(gammaRegion, subregionExitNodes);
    }
  };

  addSubregionLiveAndEntryNodes(gammaNode, *Context_);
  eliminateTopDownForSubregions(gammaNode);
  addSubregionExitNodes(gammaNode, *Context_);
  updateGammaRegionLiveNodes(gammaNode, *Context_);
}

void
TopDownMemoryNodeEliminator::EliminateTopDownTheta(const rvsdg::ThetaNode & thetaNode)
{
  auto & thetaRegion = *thetaNode.region();
  auto & thetaSubregion = *thetaNode.subregion();
  auto & seedProvisioning = Context_->GetSeedProvisioning();
  auto & provisioning = Context_->GetProvisioning();
  auto & thetaRegionLiveNodes = Context_->GetLiveNodes(thetaRegion);

  auto subregionEntryNodes = seedProvisioning.GetRegionEntryNodes(thetaSubregion);
  subregionEntryNodes.IntersectWith(thetaRegionLiveNodes);

  Context_->AddLiveNodes(thetaSubregion, subregionEntryNodes);
  provisioning.AddRegionEntryNodes(thetaSubregion, subregionEntryNodes);

  EliminateTopDownRegion(thetaSubregion);

  auto & thetaSubregionRegionLiveNodes = Context_->GetLiveNodes(thetaSubregion);
  auto subregionExitNodes = seedProvisioning.GetRegionExitNodes(thetaSubregion);
  subregionExitNodes.IntersectWith(thetaSubregionRegionLiveNodes);

  // Theta entry and exit needs to be equivalent
  provisioning.AddRegionEntryNodes(thetaSubregion, subregionExitNodes);
  provisioning.AddRegionExitNodes(thetaSubregion, subregionExitNodes);

  Context_->AddLiveNodes(thetaRegion, subregionExitNodes);
}

void
TopDownMemoryNodeEliminator::EliminateTopDownSimpleNode(const rvsdg::SimpleNode & simpleNode)
{
  if (is<alloca_op>(&simpleNode))
  {
    EliminateTopDownAlloca(simpleNode);
  }
  else if (auto callNode = dynamic_cast<const CallNode *>(&simpleNode))
  {
    EliminateTopDownCall(*callNode);
  }
}

void
TopDownMemoryNodeEliminator::EliminateTopDownAlloca(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<alloca_op>(&node));

  // We found an alloca node. Add the respective points-to graph memory node to the live nodes.
  auto & allocaNode = Context_->GetPointsToGraph().GetAllocaNode(node);
  Context_->AddLiveNodes(*node.region(), { &allocaNode });
}

void
TopDownMemoryNodeEliminator::EliminateTopDownCall(const CallNode & callNode)
{
  auto callTypeClassifier = CallNode::ClassifyCall(callNode);

  switch (callTypeClassifier->GetCallType())
  {
  case CallTypeClassifier::CallType::NonRecursiveDirectCall:
    EliminateTopDownNonRecursiveDirectCall(callNode, *callTypeClassifier);
    break;
  case CallTypeClassifier::CallType::RecursiveDirectCall:
    EliminateTopDownRecursiveDirectCall(callNode, *callTypeClassifier);
    break;
  case CallTypeClassifier::CallType::ExternalCall:
    EliminateTopDownExternalCall(callNode, *callTypeClassifier);
    break;
  case CallTypeClassifier::CallType::IndirectCall:
    EliminateTopDownIndirectCall(callNode, *callTypeClassifier);
    break;
  default:
    JLM_UNREACHABLE("Unhandled call type classifier!");
  }
}

void
TopDownMemoryNodeEliminator::EliminateTopDownNonRecursiveDirectCall(
    const CallNode & callNode,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsNonRecursiveDirectCall());

  auto & liveNodes = Context_->GetLiveNodes(*callNode.region());
  auto & lambdaNode = rvsdg::AssertGetOwnerNode<lambda::node>(callTypeClassifier.GetLambdaOutput());

  Context_->AddLiveNodes(*lambdaNode.subregion(), liveNodes);
  Context_->AddLiveNodesAnnotatedLambda(lambdaNode);
}

void
TopDownMemoryNodeEliminator::EliminateTopDownRecursiveDirectCall(
    const CallNode & callNode,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsRecursiveDirectCall());

  auto & liveNodes = Context_->GetLiveNodes(*callNode.region());
  auto & lambdaNode = rvsdg::AssertGetOwnerNode<lambda::node>(callTypeClassifier.GetLambdaOutput());

  Context_->AddLiveNodes(*lambdaNode.subregion(), liveNodes);
  Context_->AddLiveNodesAnnotatedLambda(lambdaNode);
}

void
TopDownMemoryNodeEliminator::EliminateTopDownExternalCall(
    const CallNode & callNode,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsExternalCall());

  auto & liveNodes = Context_->GetLiveNodes(*callNode.region());

  auto & seedCallEntryNodes = Context_->GetSeedProvisioning().GetCallEntryNodes(callNode);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallEntryNodes));

  auto & seedCallExitNodes = Context_->GetSeedProvisioning().GetCallExitNodes(callNode);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallExitNodes));

  Context_->GetProvisioning().AddExternalCallNodes(callNode, liveNodes);
}

void
TopDownMemoryNodeEliminator::EliminateTopDownIndirectCall(
    const CallNode & indirectCall,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsIndirectCall());

  auto & liveNodes = Context_->GetLiveNodes(*indirectCall.region());

  auto & seedCallEntryNodes = Context_->GetSeedProvisioning().GetCallEntryNodes(indirectCall);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallEntryNodes));

  auto & seedCallExitNodes = Context_->GetSeedProvisioning().GetCallExitNodes(indirectCall);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallExitNodes));

  Context_->GetProvisioning().AddIndirectCallNodes(indirectCall, liveNodes);
}

void
TopDownMemoryNodeEliminator::InitializeLiveNodesOfTailLambdas(
    const rvsdg::RvsdgModule & rvsdgModule)
{
  auto nodes = rvsdg::Graph::ExtractTailNodes(rvsdgModule.Rvsdg());
  for (auto & node : nodes)
  {
    if (auto lambdaNode = dynamic_cast<const lambda::node *>(node))
    {
      InitializeLiveNodesOfTailLambda(*lambdaNode);
    }
    else if (auto phiNode = dynamic_cast<const phi::node *>(node))
    {
      auto lambdaNodes = phi::node::ExtractLambdaNodes(*phiNode);
      for (auto & phiLambdaNode : lambdaNodes)
      {
        InitializeLiveNodesOfTailLambda(*phiLambdaNode);
      }
    }
    else if (dynamic_cast<const delta::node *>(node))
    {
      // Nothing needs to be done for delta nodes.
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }
}

void
TopDownMemoryNodeEliminator::InitializeLiveNodesOfTailLambda(const lambda::node & tailLambdaNode)
{
  auto IsUnescapedAllocaNode = [&](const PointsToGraph::MemoryNode * memoryNode)
  {
    auto & escapedMemoryNodes = Context_->GetPointsToGraph().GetEscapedMemoryNodes();

    return PointsToGraph::Node::Is<PointsToGraph::AllocaNode>(*memoryNode)
        && !escapedMemoryNodes.Contains(memoryNode);
  };

  auto & lambdaSubregion = *tailLambdaNode.subregion();
  auto & seedProvisioning = Context_->GetSeedProvisioning();

  auto memoryNodes = seedProvisioning.GetLambdaEntryNodes(tailLambdaNode);
  memoryNodes.RemoveWhere(IsUnescapedAllocaNode);

  Context_->AddLiveNodes(lambdaSubregion, memoryNodes);
  Context_->AddLiveNodesAnnotatedLambda(tailLambdaNode);
}

bool
TopDownMemoryNodeEliminator::CheckInvariants(
    const rvsdg::RvsdgModule & rvsdgModule,
    const MemoryNodeProvisioning & seedProvisioning,
    const Provisioning & provisioning)
{
  std::function<void(
      const rvsdg::Region &,
      std::vector<const rvsdg::Region *> &,
      std::vector<const CallNode *> &)>
      collectRegionsAndCalls = [&](const rvsdg::Region & rootRegion,
                                   std::vector<const rvsdg::Region *> & regions,
                                   std::vector<const CallNode *> & callNodes)
  {
    for (auto & node : rootRegion.Nodes())
    {
      if (auto lambdaNode = dynamic_cast<const lambda::node *>(&node))
      {
        auto lambdaSubregion = lambdaNode->subregion();
        regions.push_back(lambdaSubregion);
        collectRegionsAndCalls(*lambdaSubregion, regions, callNodes);
      }
      else if (auto phiNode = dynamic_cast<const phi::node *>(&node))
      {
        auto subregion = phiNode->subregion();
        collectRegionsAndCalls(*subregion, regions, callNodes);
      }
      else if (auto gammaNode = dynamic_cast<const rvsdg::GammaNode *>(&node))
      {
        for (size_t n = 0; n < gammaNode->nsubregions(); n++)
        {
          auto subregion = gammaNode->subregion(n);
          regions.push_back(subregion);
          collectRegionsAndCalls(*subregion, regions, callNodes);
        }
      }
      else if (auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(&node))
      {
        auto subregion = thetaNode->subregion();
        regions.push_back(subregion);
        collectRegionsAndCalls(*subregion, regions, callNodes);
      }
      else if (auto callNode = dynamic_cast<const CallNode *>(&node))
      {
        callNodes.push_back(callNode);
      }
    }
  };

  std::vector<const CallNode *> callNodes;
  std::vector<const rvsdg::Region *> regions;
  collectRegionsAndCalls(rvsdgModule.Rvsdg().GetRootRegion(), regions, callNodes);

  for (auto region : regions)
  {
    auto & regionEntry = provisioning.GetRegionEntryNodes(*region);
    auto & seedRegionEntry = seedProvisioning.GetRegionEntryNodes(*region);
    if (!regionEntry.IsSubsetOf(seedRegionEntry))
    {
      return false;
    }

    auto & regionExit = provisioning.GetRegionExitNodes(*region);
    auto & seedRegionExit = provisioning.GetRegionExitNodes(*region);
    if (!regionExit.IsSubsetOf(seedRegionExit))
    {
      return false;
    }
  }

  for (auto callNode : callNodes)
  {
    auto & callEntry = provisioning.GetCallEntryNodes(*callNode);
    auto & seedCallEntry = provisioning.GetCallEntryNodes(*callNode);
    if (!callEntry.IsSubsetOf(seedCallEntry))
    {
      return false;
    }

    auto & callExit = provisioning.GetCallExitNodes(*callNode);
    auto & seedCallExit = provisioning.GetCallExitNodes(*callNode);
    if (!callExit.IsSubsetOf(seedCallExit))
    {
      return false;
    }
  }

  return true;
}

}
