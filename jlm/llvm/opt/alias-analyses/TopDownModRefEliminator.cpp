/*
 * Copyright 2023 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/TopDownModRefEliminator.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::llvm::aa
{

/** \brief Collect statistics about \ref TopDownModRefEliminator pass
 *
 */
class TopDownModRefEliminator::Statistics final : public util::Statistics
{
public:
  ~Statistics() override = default;

  explicit Statistics(const util::FilePath & sourceFile)
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
  Create(const jlm::util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

TopDownModRefEliminator::~TopDownModRefEliminator() noexcept = default;

TopDownModRefEliminator::TopDownModRefEliminator() = default;

std::unique_ptr<ModRefSummary>
TopDownModRefEliminator::EliminateModRefs(
    const rvsdg::RvsdgModule & rvsdgModule,
    const aa::ModRefSummary & seedModRefSummary,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics = Statistics::Create(rvsdgModule.SourceFilePath().value());

  statistics->Start(rvsdgModule.Rvsdg());
  EliminateTopDown(rvsdgModule);
  statistics->Stop();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  auto modRefSummary = Context_->ReleaseModRefSummary();
  Context_.reset();

  JLM_ASSERT(CheckInvariants(rvsdgModule, seedModRefSummary, *modRefSummary));

  return seedModRefSummary;
}

std::unique_ptr<ModRefSummary>
TopDownModRefEliminator::CreateAndEliminate(
    const rvsdg::RvsdgModule & rvsdgModule,
    const aa::ModRefSummary & modRefSummary,
    util::StatisticsCollector & statisticsCollector)
{
  TopDownModRefEliminator summarizer;
  return summarizer.EliminateModRefs(rvsdgModule, modRefSummary, statisticsCollector);
}

std::unique_ptr<ModRefSummary>
TopDownModRefEliminator::CreateAndEliminate(
    const rvsdg::RvsdgModule & rvsdgModule,
    const aa::ModRefSummary & seedModRefSummary)
{
  util::StatisticsCollector statisticsCollector;
  return CreateAndEliminate(rvsdgModule, seedModRefSummary, statisticsCollector);
}

void
TopDownModRefEliminator::EliminateTopDown(const rvsdg::RvsdgModule & rvsdgModule)
{
  // Initialize the memory nodes that are alive at beginning of every tail-lambda
  InitializeLiveNodesOfTailLambdas(rvsdgModule);

  // Start the processing of the RVSDG module
  EliminateTopDownRootRegion(rvsdgModule.Rvsdg().GetRootRegion());
}

void
TopDownModRefEliminator::EliminateTopDownRootRegion(rvsdg::Region & region)
{
  JLM_ASSERT(region.IsRootRegion() || dynamic_cast<const rvsdg::PhiNode *>(region.node()));

  // Process the lambda, phi, and delta nodes bottom-up.
  // This ensures that we visit all the call nodes before we visit the respective lambda nodes.
  // The tail-lambdas (lambda nodes without calls in the RVSDG module) have already been visited and
  // initialized by InitializeLiveNodesOfTailLambdas().
  rvsdg::BottomUpTraverser traverser(&region);
  for (auto & node : traverser)
  {
    rvsdg::MatchTypeWithDefault(
        *node,
        [this](const rvsdg::LambdaNode & node)
        {
          EliminateTopDownLambda(node);
        },
        [this](const rvsdg::PhiNode & node)
        {
          EliminateTopDownPhi(node);
        },
        [](const rvsdg::DeltaNode & node)
        {
          // Nothing needs to be done.
        },
        [](const rvsdg::SimpleNode & node)
        {
          rvsdg::MatchTypeWithDefault(
              node.GetOperation(),
              [](const FunctionToPointerOperation &)
              {
              },
              [](const PointerToFunctionOperation &)
              {
              },
              []()
              {
                JLM_UNREACHABLE("Unhandled node type!");
              });
        },
        []()
        {
          JLM_UNREACHABLE("Unhandled node type!");
        });
  }
}

void
TopDownModRefEliminator::EliminateTopDownRegion(rvsdg::Region & region)
{
  auto isLambdaSubregion = dynamic_cast<const rvsdg::LambdaNode *>(region.node());
  auto isThetaSubregion = dynamic_cast<const rvsdg::ThetaNode *>(region.node());
  auto isGammaSubregion = dynamic_cast<const rvsdg::GammaNode *>(region.node());
  JLM_ASSERT(isLambdaSubregion || isThetaSubregion || isGammaSubregion);

  // Process the intra-procedural nodes top-down.
  // This ensures that we add the live memory nodes to the live sets when the respective RVSDG nodes
  // appear in the visitation.
  rvsdg::TopDownTraverser traverser(&region);
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
TopDownModRefEliminator::EliminateTopDownStructuralNode(
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
TopDownModRefEliminator::EliminateTopDownLambda(const rvsdg::LambdaNode & lambdaNode)
{
  EliminateTopDownLambdaEntry(lambdaNode);
  EliminateTopDownRegion(*lambdaNode.subregion());
  EliminateTopDownLambdaExit(lambdaNode);
}

void
TopDownModRefEliminator::EliminateTopDownLambdaEntry(const rvsdg::LambdaNode & lambdaNode)
{
  auto & lambdaSubregion = *lambdaNode.subregion();
  auto & modRefSummary = Context_->GetModRefSummary();
  auto & seedModRefSummary = Context_->GetSeedModRefSummary();

  if (Context_->HasAnnotatedLiveNodes(lambdaNode))
  {
    // Live nodes were annotated. This means that either:
    // 1. This lambda node has direct calls that were already handled due to bottom-up visitation.
    // 2. This lambda is a tail-lambda and live nodes were annotated by
    // InitializeLiveNodesOfTailLambdas()
    auto & liveNodes = Context_->GetLiveNodes(lambdaSubregion);
    modRefSummary.AddRegionEntryNodes(lambdaSubregion, liveNodes);
  }
  else
  {
    // Live nodes were not annotated. This means that:
    // 1. This lambda has no direct calls (but potentially only indirect calls)
    // 2. This lambda is dead and is not used at all
    //
    // Thus, we have no idea what memory nodes are live at its entry. Thus, we need to be
    // conservative and simply say that all memory nodes from the seed mod/ref summary are live.
    auto & seedLambdaEntryNodes = seedModRefSummary.GetLambdaEntryNodes(lambdaNode);
    Context_->AddLiveNodes(lambdaSubregion, seedLambdaEntryNodes);
    modRefSummary.AddRegionEntryNodes(lambdaSubregion, seedLambdaEntryNodes);
  }
}

void
TopDownModRefEliminator::EliminateTopDownLambdaExit(const rvsdg::LambdaNode & lambdaNode)
{
  auto & lambdaSubregion = *lambdaNode.subregion();
  auto & modRefSummary = Context_->GetModRefSummary();
  auto & seedModRefSummary = Context_->GetSeedModRefSummary();

  if (Context_->HasAnnotatedLiveNodes(lambdaNode))
  {
    // Live nodes were annotated. This means that either:
    // 1. This lambda node has direct calls that were already handled due to bottom-up visitation.
    // 2. This lambda is a tail-lambda and live nodes were annotated by
    // InitializeLiveNodesOfTailLambdas()
    auto & entryNodes = modRefSummary.GetLambdaEntryNodes(lambdaNode);
    modRefSummary.AddRegionExitNodes(lambdaSubregion, entryNodes);
  }
  else
  {
    // Live nodes were not annotated. This means that:
    // 1. This lambda has no direct calls (but potentially only indirect calls)
    // 2. This lambda is dead and is not used at all
    //
    // Thus, we have no idea what memory nodes are live at its entry. Thus, we need to be
    // conservative and simply say that all memory nodes from the seed mod/ref summary are live.
    auto & seedLambdaExitNodes = seedModRefSummary.GetLambdaExitNodes(lambdaNode);
    modRefSummary.AddRegionExitNodes(lambdaSubregion, seedLambdaExitNodes);
  }
}

void
TopDownModRefEliminator::EliminateTopDownPhi(const rvsdg::PhiNode & phiNode)
{
  auto unifyLiveNodes = [&](const rvsdg::Region & phiSubregion)
  {
    std::vector<const rvsdg::LambdaNode *> lambdaNodes;
    util::HashSet<const PointsToGraph::MemoryNode *> liveNodes;
    for (auto & node : phiSubregion.Nodes())
    {
      if (auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(&node))
      {
        lambdaNodes.emplace_back(lambdaNode);

        auto & lambdaSubregion = *lambdaNode->subregion();
        auto & lambdaLiveNodes = Context_->GetLiveNodes(lambdaSubregion);
        liveNodes.UnionWith(lambdaLiveNodes);
      }
      else if (is<DeltaOperation>(&node))
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
TopDownModRefEliminator::EliminateTopDownGamma(const rvsdg::GammaNode & gammaNode)
{
  auto addSubregionLiveAndEntryNodes =
      [](const rvsdg::GammaNode & gammaNode, TopDownModRefEliminator::Context & context)
  {
    auto & gammaRegion = *gammaNode.region();
    auto & seedModRefSummary = context.GetSeedModRefSummary();
    auto & modRefSummary = context.GetModRefSummary();
    auto & gammaRegionLiveNodes = context.GetLiveNodes(gammaRegion);

    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);

      auto subregionEntryNodes = seedModRefSummary.GetRegionEntryNodes(subregion);
      subregionEntryNodes.IntersectWith(gammaRegionLiveNodes);

      context.AddLiveNodes(subregion, subregionEntryNodes);
      modRefSummary.AddRegionEntryNodes(subregion, subregionEntryNodes);
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
      [](const rvsdg::GammaNode & gammaNode, TopDownModRefEliminator::Context & context)
  {
    auto & modRefSummary = context.GetModRefSummary();

    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);
      auto & liveNodes = context.GetLiveNodes(subregion);
      modRefSummary.AddRegionExitNodes(subregion, liveNodes);
    }
  };

  auto updateGammaRegionLiveNodes =
      [](const rvsdg::GammaNode & gammaNode, TopDownModRefEliminator::Context & context)
  {
    auto & gammaRegion = *gammaNode.region();
    auto & modRefSummary = context.GetModRefSummary();

    for (size_t n = 0; n < gammaNode.nsubregions(); n++)
    {
      auto & subregion = *gammaNode.subregion(n);
      auto & subregionExitNodes = modRefSummary.GetRegionExitNodes(subregion);
      context.AddLiveNodes(gammaRegion, subregionExitNodes);
    }
  };

  addSubregionLiveAndEntryNodes(gammaNode, *Context_);
  eliminateTopDownForSubregions(gammaNode);
  addSubregionExitNodes(gammaNode, *Context_);
  updateGammaRegionLiveNodes(gammaNode, *Context_);
}

void
TopDownModRefEliminator::EliminateTopDownTheta(const rvsdg::ThetaNode & thetaNode)
{
  auto & thetaRegion = *thetaNode.region();
  auto & thetaSubregion = *thetaNode.subregion();
  auto & seedModRefSummary = Context_->GetSeedModRefSummary();
  auto & modRefSummary = Context_->GetModRefSummary();
  auto & thetaRegionLiveNodes = Context_->GetLiveNodes(thetaRegion);

  auto subregionEntryNodes = seedModRefSummary.GetRegionEntryNodes(thetaSubregion);
  subregionEntryNodes.IntersectWith(thetaRegionLiveNodes);

  Context_->AddLiveNodes(thetaSubregion, subregionEntryNodes);
  modRefSummary.AddRegionEntryNodes(thetaSubregion, subregionEntryNodes);

  EliminateTopDownRegion(thetaSubregion);

  auto & thetaSubregionRegionLiveNodes = Context_->GetLiveNodes(thetaSubregion);
  auto subregionExitNodes = seedModRefSummary.GetRegionExitNodes(thetaSubregion);
  subregionExitNodes.IntersectWith(thetaSubregionRegionLiveNodes);

  // Theta entry and exit needs to be equivalent
  modRefSummary.AddRegionEntryNodes(thetaSubregion, subregionExitNodes);
  modRefSummary.AddRegionExitNodes(thetaSubregion, subregionExitNodes);

  Context_->AddLiveNodes(thetaRegion, subregionExitNodes);
}

void
TopDownModRefEliminator::EliminateTopDownSimpleNode(const rvsdg::SimpleNode & simpleNode)
{
  if (is<AllocaOperation>(&simpleNode))
  {
    EliminateTopDownAlloca(simpleNode);
  }
  else if (is<CallOperation>(&simpleNode))
  {
    EliminateTopDownCall(simpleNode);
  }
}

void
TopDownModRefEliminator::EliminateTopDownAlloca(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<AllocaOperation>(&node));

  // We found an alloca node. Add the respective points-to graph memory node to the live nodes.
  auto & allocaNode = Context_->GetPointsToGraph().GetAllocaNode(node);
  Context_->AddLiveNodes(*node.region(), { &allocaNode });
}

void
TopDownModRefEliminator::EliminateTopDownCall(const rvsdg::SimpleNode & callNode)
{
  auto callTypeClassifier = CallOperation::ClassifyCall(callNode);

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
TopDownModRefEliminator::EliminateTopDownNonRecursiveDirectCall(
    const rvsdg::SimpleNode & callNode,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsNonRecursiveDirectCall());

  auto & liveNodes = Context_->GetLiveNodes(*callNode.region());
  auto & lambdaNode =
      rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(callTypeClassifier.GetLambdaOutput());

  Context_->AddLiveNodes(*lambdaNode.subregion(), liveNodes);
  Context_->AddLiveNodesAnnotatedLambda(lambdaNode);
}

void
TopDownModRefEliminator::EliminateTopDownRecursiveDirectCall(
    const rvsdg::SimpleNode & callNode,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsRecursiveDirectCall());

  auto & liveNodes = Context_->GetLiveNodes(*callNode.region());
  auto & lambdaNode =
      rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(callTypeClassifier.GetLambdaOutput());

  Context_->AddLiveNodes(*lambdaNode.subregion(), liveNodes);
  Context_->AddLiveNodesAnnotatedLambda(lambdaNode);
}

void
TopDownModRefEliminator::EliminateTopDownExternalCall(
    const rvsdg::SimpleNode & callNode,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsExternalCall());

  auto & liveNodes = Context_->GetLiveNodes(*callNode.region());

  auto & seedCallEntryNodes = Context_->GetSeedModRefSummary().GetCallEntryNodes(callNode);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallEntryNodes));

  auto & seedCallExitNodes = Context_->GetSeedModRefSummary().GetCallExitNodes(callNode);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallExitNodes));

  Context_->GetModRefSummary().AddExternalCallNodes(callNode, liveNodes);
}

void
TopDownModRefEliminator::EliminateTopDownIndirectCall(
    const rvsdg::SimpleNode & indirectCall,
    const CallTypeClassifier & callTypeClassifier)
{
  JLM_ASSERT(callTypeClassifier.IsIndirectCall());

  auto & liveNodes = Context_->GetLiveNodes(*indirectCall.region());

  auto & seedCallEntryNodes = Context_->GetSeedModRefSummary().GetCallEntryNodes(indirectCall);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallEntryNodes));

  auto & seedCallExitNodes = Context_->GetSeedModRefSummary().GetCallExitNodes(indirectCall);
  JLM_ASSERT(liveNodes.IsSubsetOf(seedCallExitNodes));

  Context_->GetModRefSummary().AddIndirectCallNodes(indirectCall, liveNodes);
}

void
TopDownModRefEliminator::InitializeLiveNodesOfTailLambdas(const rvsdg::RvsdgModule & rvsdgModule)
{
  auto nodes = rvsdg::Graph::ExtractTailNodes(rvsdgModule.Rvsdg());
  for (auto & node : nodes)
  {
    MatchTypeOrFail(
        *node,
        [this](const rvsdg::LambdaNode & lambdaNode)
        {
          InitializeLiveNodesOfTailLambda(lambdaNode);
        },
        [this](const rvsdg::PhiNode & phiNode)
        {
          auto lambdaNodes = rvsdg::PhiNode::ExtractLambdaNodes(phiNode);
          for (auto & phiLambdaNode : lambdaNodes)
          {
            InitializeLiveNodesOfTailLambda(*phiLambdaNode);
          }
        },
        [](const rvsdg::DeltaNode &)
        {
          // Nothing needs to be done for delta nodes.
        });
  }
}

void
TopDownModRefEliminator::InitializeLiveNodesOfTailLambda(const rvsdg::LambdaNode & tailLambdaNode)
{
  auto IsUnescapedAllocaNode = [&](const PointsToGraph::MemoryNode * memoryNode)
  {
    auto & escapedMemoryNodes = Context_->GetPointsToGraph().GetEscapedMemoryNodes();

    return PointsToGraph::Node::Is<PointsToGraph::AllocaNode>(*memoryNode)
        && !escapedMemoryNodes.Contains(memoryNode);
  };

  auto & lambdaSubregion = *tailLambdaNode.subregion();
  auto & seedModRefSummary = Context_->GetSeedModRefSummary();

  auto memoryNodes = seedModRefSummary.GetLambdaEntryNodes(tailLambdaNode);
  memoryNodes.RemoveWhere(IsUnescapedAllocaNode);

  Context_->AddLiveNodes(lambdaSubregion, memoryNodes);
  Context_->AddLiveNodesAnnotatedLambda(tailLambdaNode);
}

bool
TopDownModRefEliminator::CheckInvariants(
    const rvsdg::RvsdgModule & rvsdgModule,
    const aa::ModRefSummary & seedModRefSummary,
    const aa::ModRefSummary & modRefSummary)
{
  std::function<void(
      const rvsdg::Region &,
      std::vector<const rvsdg::Region *> &,
      std::vector<const rvsdg::SimpleNode *> &)>
      collectRegionsAndCalls = [&](const rvsdg::Region & rootRegion,
                                   std::vector<const rvsdg::Region *> & regions,
                                   std::vector<const rvsdg::SimpleNode *> & callNodes)
  {
    for (auto & node : rootRegion.Nodes())
    {
      if (auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(&node))
      {
        auto lambdaSubregion = lambdaNode->subregion();
        regions.push_back(lambdaSubregion);
        collectRegionsAndCalls(*lambdaSubregion, regions, callNodes);
      }
      else if (auto phiNode = dynamic_cast<const rvsdg::PhiNode *>(&node))
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
      else if (is<CallOperation>(&node))
      {
        callNodes.push_back(util::AssertedCast<const rvsdg::SimpleNode>(&node));
      }
    }
  };

  std::vector<const rvsdg::SimpleNode *> callNodes;
  std::vector<const rvsdg::Region *> regions;
  collectRegionsAndCalls(rvsdgModule.Rvsdg().GetRootRegion(), regions, callNodes);

  for (auto region : regions)
  {
    auto & regionEntry = modRefSummary.GetRegionEntryNodes(*region);
    auto & seedRegionEntry = seedModRefSummary.GetRegionEntryNodes(*region);
    if (!regionEntry.IsSubsetOf(seedRegionEntry))
    {
      return false;
    }

    auto & regionExit = modRefSummary.GetRegionExitNodes(*region);
    auto & seedRegionExit = modRefSummary.GetRegionExitNodes(*region);
    if (!regionExit.IsSubsetOf(seedRegionExit))
    {
      return false;
    }
  }

  for (auto callNode : callNodes)
  {
    auto & callEntry = modRefSummary.GetCallEntryNodes(*callNode);
    auto & seedCallEntry = modRefSummary.GetCallEntryNodes(*callNode);
    if (!callEntry.IsSubsetOf(seedCallEntry))
    {
      return false;
    }

    auto & callExit = modRefSummary.GetCallExitNodes(*callNode);
    auto & seedCallExit = modRefSummary.GetCallExitNodes(*callNode);
    if (!callExit.IsSubsetOf(seedCallExit))
    {
      return false;
    }
  }

  return true;
}

}
