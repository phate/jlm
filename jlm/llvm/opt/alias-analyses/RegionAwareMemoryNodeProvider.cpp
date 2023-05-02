/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <typeindex>

namespace jlm::aa {

class RegionSummary final {
public:
  explicit
  RegionSummary(const jive::region & region)
  : Region_(&region)
  {}

  RegionSummary(const RegionSummary&) = delete;

  RegionSummary(RegionSummary&&) = delete;

  RegionSummary &
  operator=(const RegionSummary&) = delete;

  RegionSummary &
  operator=(RegionSummary&&) = delete;

  const HashSet<const PointsToGraph::MemoryNode*> &
  GetMemoryNodes() const
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const HashSet<const jive::simple_node*> &
  GetUnknownMemoryNodeReferences() const noexcept
  {
    return UnknownMemoryNodeReferences_;
  }

  const HashSet<const CallNode*> &
  GetNonRecursiveCalls() const
  {
    return NonRecursiveCalls_;
  }

  const HashSet<const CallNode*> &
  GetRecursiveCalls() const
  {
    return RecursiveCalls_;
  }

  const HashSet<const jive::structural_node*> &
  GetStructuralNodes() const
  {
    return StructuralNodes_;
  }

  void
  AddMemoryNodes(const HashSet<const PointsToGraph::MemoryNode*> & memoryNodes)
  {
    MemoryNodes_.UnionWith(memoryNodes);
  }

  void
  AddUnknownMemoryNodeReferences(const HashSet<const jive::simple_node*> & nodes)
  {
    UnknownMemoryNodeReferences_.UnionWith(nodes);
  }

  void
  AddNonRecursiveDirectCall(const CallNode & callNode)
  {
    JLM_ASSERT(CallNode::ClassifyCall(callNode)->GetCallType() == CallTypeClassifier::CallType::NonRecursiveDirectCall);
    NonRecursiveCalls_.Insert(&callNode);
  }

  void
  AddRecursiveDirectCall(const CallNode & callNode)
  {
    JLM_ASSERT(CallNode::ClassifyCall(callNode)->GetCallType() == CallTypeClassifier::CallType::RecursiveDirectCall);
    RecursiveCalls_.Insert(&callNode);
  }

  void
  AddStructuralNode(const jive::structural_node & structuralNode)
  {
    StructuralNodes_.Insert(&structuralNode);
  }

  [[nodiscard]] const jive::region &
  GetRegion() const noexcept
  {
    return *Region_;
  }

  static void
  Propagate(RegionSummary & dstSummary, const RegionSummary & srcSummary)
  {
    dstSummary.AddMemoryNodes(srcSummary.GetMemoryNodes());
    dstSummary.AddUnknownMemoryNodeReferences(srcSummary.GetUnknownMemoryNodeReferences());
  }

  static std::unique_ptr<RegionSummary>
  Create(const jive::region & region)
  {
    return std::make_unique<RegionSummary>(region);
  }

private:
  const jive::region * Region_;
  HashSet<const PointsToGraph::MemoryNode*> MemoryNodes_;
  HashSet<const jive::simple_node*> UnknownMemoryNodeReferences_;

  HashSet<const CallNode*> RecursiveCalls_;
  HashSet<const CallNode*> NonRecursiveCalls_;
  HashSet<const jive::structural_node*> StructuralNodes_;
};

/** \brief Memory node provisioning of region-aware memory node provider
 *
 */
class RegionAwareMemoryNodeProvisioning final : public MemoryNodeProvisioning {
  using RegionSummaryMap = std::unordered_map<const jive::region*, std::unique_ptr<RegionSummary>>;

  class RegionSummaryConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const RegionSummary*;
    using difference_type = std::ptrdiff_t;
    using pointer = const RegionSummary**;
    using reference = const RegionSummary*&;

  private:
    friend RegionAwareMemoryNodeProvisioning;

    explicit
    RegionSummaryConstIterator(const RegionSummaryMap::const_iterator & it)
      : it_(it)
    {}

  public:
    [[nodiscard]] const RegionSummary *
    GetRegionSummary() const noexcept
    {
      return it_->second.get();
    }

    const RegionSummary &
    operator*() const
    {
      JLM_ASSERT(GetRegionSummary() != nullptr);
      return *GetRegionSummary();
    }

    const RegionSummary *
    operator->() const
    {
      return GetRegionSummary();
    }

    RegionSummaryConstIterator &
    operator++()
    {
      ++it_;
      return *this;
    }

    RegionSummaryConstIterator
    operator++(int)
    {
      RegionSummaryConstIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const RegionSummaryConstIterator & other) const
    {
      return it_ == other.it_;
    }

    bool
    operator!=(const RegionSummaryConstIterator & other) const
    {
      return !operator==(other);
    }

  private:
    RegionSummaryMap::const_iterator  it_;
  };

  using RegionSummaryConstRange = iterator_range<RegionSummaryConstIterator>;

public:
  explicit
  RegionAwareMemoryNodeProvisioning(const PointsToGraph & pointsToGraph)
    : PointsToGraph_(pointsToGraph)
  {}

  RegionAwareMemoryNodeProvisioning(const RegionAwareMemoryNodeProvisioning&) = delete;

  RegionAwareMemoryNodeProvisioning(RegionAwareMemoryNodeProvisioning&&) = delete;

  RegionAwareMemoryNodeProvisioning &
  operator=(const RegionAwareMemoryNodeProvisioning&) = delete;

  RegionAwareMemoryNodeProvisioning &
  operator=(RegionAwareMemoryNodeProvisioning&&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionEntryNodes(const jive::region & region) const override
  {
    auto & regionSummary = GetRegionSummary(region);
    return regionSummary.GetMemoryNodes();
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetRegionExitNodes(const jive::region & region) const override
  {
    auto & regionSummary = GetRegionSummary(region);
    return regionSummary.GetMemoryNodes();
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallEntryNodes(const CallNode & callNode) const override
  {
    auto callTypeClassifier = CallNode::ClassifyCall(callNode);

    if (callTypeClassifier->IsNonRecursiveDirectCall())
    {
      auto & lambdaNode = *callTypeClassifier->GetLambdaOutput().node();
      return GetLambdaEntryNodes(lambdaNode);
    }
    else if (callTypeClassifier->IsRecursiveDirectCall())
    {
      auto & lambdaNode = *callTypeClassifier->GetLambdaOutput().node();
      return GetLambdaEntryNodes(lambdaNode);
    }
    else if (callTypeClassifier->IsExternalCall())
    {
      auto & import = callTypeClassifier->GetImport();
      return GetExternalFunctionNodes(import);
    }
    else if (callTypeClassifier->IsIndirectCall())
    {
      return GetIndirectCallNodes(callNode);
    }

    JLM_UNREACHABLE("Unhandled call type.");
  }

  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetCallExitNodes(const CallNode & callNode) const override
  {
    auto callTypeClassifier = CallNode::ClassifyCall(callNode);

    if (callTypeClassifier->IsNonRecursiveDirectCall())
    {
      auto & lambdaNode = *callTypeClassifier->GetLambdaOutput().node();
      return GetLambdaExitNodes(lambdaNode);
    }
    else if (callTypeClassifier->IsRecursiveDirectCall())
    {
      auto & lambdaNode = *callTypeClassifier->GetLambdaOutput().node();
      return GetLambdaExitNodes(lambdaNode);
    }
    else if (callTypeClassifier->IsExternalCall())
    {
      auto & import = callTypeClassifier->GetImport();
      return GetExternalFunctionNodes(import);
    }
    else if (callTypeClassifier->IsIndirectCall())
    {
      return GetIndirectCallNodes(callNode);
    }

    JLM_UNREACHABLE("Unhandled call type!");
  }

  [[nodiscard]] HashSet<const PointsToGraph::MemoryNode*>
  GetOutputNodes(const jive::output & output) const override
  {
    JLM_ASSERT(is<PointerType>(output.type()));
    auto & registerNode = PointsToGraph_.GetRegisterNode(output);

    HashSet<const PointsToGraph::MemoryNode*> memoryNodes;
    for (auto & memoryNode : registerNode.Targets())
      memoryNodes.Insert(&memoryNode);

    return memoryNodes;
  }

  RegionSummaryConstRange
  GetRegionSummaries() const
  {
    return {RegionSummaryConstIterator(RegionSummaries_.begin()), RegionSummaryConstIterator(RegionSummaries_.end())};
  }

  [[nodiscard]] bool
  ContainsRegionSummary(const jive::region & region) const
  {
    return RegionSummaries_.find(&region) != RegionSummaries_.end();
  }

  bool
  ContainsExternalFunctionNodes(const jive::argument & import) const
  {
    return ExternalFunctionNodes_.find(&import) != ExternalFunctionNodes_.end();
  }

  [[nodiscard]] RegionSummary &
  GetRegionSummary(const jive::region & region) const
  {
    JLM_ASSERT(ContainsRegionSummary(region));
    return *RegionSummaries_.find(&region)->second;
  }

  const HashSet<const PointsToGraph::MemoryNode*> &
  GetExternalFunctionNodes(const jive::argument & import) const
  {
    JLM_ASSERT(ContainsExternalFunctionNodes(import));

    return (*ExternalFunctionNodes_.find(&import)).second;
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

  void
  AddExternalFunctionNodes(
    const jive::argument & import,
    HashSet<const PointsToGraph::MemoryNode*> memoryNodes)
  {
    JLM_ASSERT(!ContainsExternalFunctionNodes(import));
    ExternalFunctionNodes_[&import] = std::move(memoryNodes);
  }

  static std::unique_ptr<RegionAwareMemoryNodeProvisioning>
  Create(const PointsToGraph & pointsToGraph)
  {
    return std::make_unique<RegionAwareMemoryNodeProvisioning>(pointsToGraph);
  }

  /**
   * This function checks the following two invariants:
   *
   * 1. The collections of memory nodes of all subregions of a structural node should be contained in the collection of
   * memory nodes of the region the structural node is contained in.
   *
   * 2. The collection of memory nodes of a lambda region should be contained in the collection of memory nodes of the
   * regions of all direct calls to this lambda.
   *
   * 3. The collections of unknown memory reference nodes of all subregions of a structural node should be contained in
   * the collection of unknown memory reference nodes of the region the structural node is contained in.
   *
   * 4. The collection of unknown memory reference nodes of a lambda region should be contained in the collection of
   * memory nodes of the regions of all direct calls to this lambda.
   *
   * @param provisioning \see RegionAwareMemoryNodeProvisioning
   * @return Returns true if all invariants are fulfilled, otherwise false.
   */
  static bool
  CheckInvariants(const RegionAwareMemoryNodeProvisioning & provisioning)
  {
    auto CheckInvariantsCall = [&](auto & callNode)
    {
      auto & regionSummary = provisioning.GetRegionSummary(*callNode.region());
      auto & regionMemoryNodes = regionSummary.GetMemoryNodes();
      auto & regionUnknownMemoryNodeReferences = regionSummary.GetUnknownMemoryNodeReferences();

      auto callTypeClassifier = CallNode::ClassifyCall(callNode);
      auto & lambdaRegion = *callTypeClassifier->GetLambdaOutput().node()->subregion();
      auto & lambdaRegionSummary = provisioning.GetRegionSummary(lambdaRegion);
      auto & lambdaRegionMemoryNodes = lambdaRegionSummary.GetMemoryNodes();
      auto & lambdaRegionUnknownMemoryNodeReferences = lambdaRegionSummary.GetUnknownMemoryNodeReferences();

      return lambdaRegionMemoryNodes.IsSubsetOf(regionMemoryNodes)
             && lambdaRegionUnknownMemoryNodeReferences.IsSubsetOf(regionUnknownMemoryNodeReferences);
    };

    auto CheckInvariantsStructuralNode = [&](auto & structuralNode)
    {
      auto & regionSummary = provisioning.GetRegionSummary(*structuralNode.region());
      auto & regionMemoryNodes = regionSummary.GetMemoryNodes();
      auto & regionUnknownMemoryNodeReferences = regionSummary.GetUnknownMemoryNodeReferences();

      for (size_t n = 0; n < structuralNode.nsubregions(); n++)
      {
        auto & subregion = *structuralNode.subregion(n);
        auto & subregionSummary = provisioning.GetRegionSummary(subregion);
        auto & subregionMemoryNodes = subregionSummary.GetMemoryNodes();
        auto & subregionUnknownMemoryNodeReferences = subregionSummary.GetUnknownMemoryNodeReferences();

        if (!subregionMemoryNodes.IsSubsetOf(regionMemoryNodes)
            || !subregionUnknownMemoryNodeReferences.IsSubsetOf(regionUnknownMemoryNodeReferences))
        {
          return false;
        }
      }

      return true;
    };

    for (auto & regionSummary : provisioning.GetRegionSummaries())
    {
      for (auto & structuralNode : regionSummary.GetStructuralNodes().Items())
      {
        if (!CheckInvariantsStructuralNode(*structuralNode))
        {
          return false;
        }
      }

      for (auto & callNode : regionSummary.GetNonRecursiveCalls().Items())
      {
        if (!CheckInvariantsCall(*callNode))
        {
          return false;
        }
      }

      for (auto & callNode : regionSummary.GetRecursiveCalls().Items())
      {
        if (!CheckInvariantsCall(*callNode))
        {
          return false;
        }
      }
    }

    return true;
  }

private:
  [[nodiscard]] const HashSet<const PointsToGraph::MemoryNode*> &
  GetIndirectCallNodes(const CallNode & callNode) const
  {
    /*
     * We have no idea about the function of an indirect call. This means that we have to be conservative and
     * sequentialize this indirect call with respect to all memory references that came before and after it. These
     * references should have been routed through the region the indirect call node lives in. Thus, we can just use
     * here the memory nodes of the region of the indirect call node.
     */
    auto & regionSummary = GetRegionSummary(*callNode.region());
    return regionSummary.GetMemoryNodes();
  }

  RegionSummaryMap RegionSummaries_;
  const PointsToGraph & PointsToGraph_;
  std::unordered_map<const jive::argument*, HashSet<const PointsToGraph::MemoryNode*>> ExternalFunctionNodes_;
};

RegionAwareMemoryNodeProvider::~RegionAwareMemoryNodeProvider() noexcept
= default;

std::unique_ptr<MemoryNodeProvisioning>
RegionAwareMemoryNodeProvider::ProvisionMemoryNodes(
  const jlm::RvsdgModule & rvsdgModule,
  const PointsToGraph & pointsToGraph,
  StatisticsCollector & statisticsCollector)
{
  Provisioning_ = RegionAwareMemoryNodeProvisioning::Create(pointsToGraph);

  auto statistics = Statistics::Create(statisticsCollector, rvsdgModule, pointsToGraph);

  statistics->StartAnnotationStatistics();
  AnnotateRegion(*rvsdgModule.Rvsdg().root());
  statistics->StopAnnotationStatistics();

  statistics->StartPropagationPass1Statistics();
  Propagate(rvsdgModule);
  statistics->StopPropagationPass1Statistics();

  statistics->StartResolveUnknownMemoryNodeReferencesStatistics();
  ResolveUnknownMemoryNodeReferences(rvsdgModule);
  statistics->StopResolveUnknownMemoryNodeReferencesStatistics();

  statistics->StartPropagationPass2Statistics();
  Propagate(rvsdgModule);
  statistics->StopPropagationPass2Statistics();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return std::unique_ptr<MemoryNodeProvisioning>(Provisioning_.release());
}

std::unique_ptr<MemoryNodeProvisioning>
RegionAwareMemoryNodeProvider::Create(
  const RvsdgModule & rvsdgModule,
  const PointsToGraph & pointsToGraph,
  StatisticsCollector & statisticsCollector)
{
  RegionAwareMemoryNodeProvider provider;
  return provider.ProvisionMemoryNodes(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<MemoryNodeProvisioning>
RegionAwareMemoryNodeProvider::Create(
  const RvsdgModule & rvsdgModule,
  const PointsToGraph & pointsToGraph)
{
  StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

void
RegionAwareMemoryNodeProvider::AnnotateRegion(jive::region & region)
{
  auto shouldCreateRegionSummary = [](auto & region)
  {
    return !region.IsRootRegion()
           && !jive::is<phi_op>(region.node())
           && !jive::is<delta::operation>(region.node());
  };

  RegionSummary * regionSummary = nullptr;
  if (shouldCreateRegionSummary(region))
  {
    regionSummary = &Provisioning_->AddRegionSummary(RegionSummary::Create(region));
  }

  for (auto & node : region.nodes)
  {
    if (auto structuralNode = dynamic_cast<const jive::structural_node*>(&node))
    {
      if (regionSummary)
      {
        regionSummary->AddStructuralNode(*structuralNode);
      }

      AnnotateStructuralNode(*structuralNode);
    }
    else if (auto simpleNode = dynamic_cast<const jive::simple_node*>(&node))
    {
      AnnotateSimpleNode(*simpleNode);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }
}

void
RegionAwareMemoryNodeProvider::AnnotateSimpleNode(const jive::simple_node & simpleNode)
{
  auto annotateLoad = [](auto & provider, auto & simpleNode)
  {
    provider.AnnotateLoad(*AssertedCast<const LoadNode>(&simpleNode));
  };
  auto annotateStore = [](auto & provider, auto & simpleNode)
  {
    provider.AnnotateStore(*AssertedCast<const StoreNode>(&simpleNode));
  };
  auto annotateAlloca = [](auto & provider, auto & simpleNode)
  {
    provider.AnnotateAlloca(simpleNode);
  };
  auto annotateMalloc = [](auto & provider, auto & simpleNode)
  {
    provider.AnnotateMalloc(simpleNode);
  };
  auto annotateFree = [](auto & provider, auto & simpleNode)
  {
    provider.AnnotateFree(simpleNode);
  };
  auto annotateCall = [](auto & provider, auto & simpleNode)
  {
    provider.AnnotateCall(*AssertedCast<const CallNode>(&simpleNode));
  };
  auto annotateMemcpy = [](auto & provider, auto & simpleNode)
  {
    provider.AnnotateMemcpy(simpleNode);
  };

  static std::unordered_map<
    std::type_index,
    std::function<void(RegionAwareMemoryNodeProvider&, const jive::simple_node&)>
  > nodes
    ({
       {typeid(LoadOperation),  annotateLoad},
       {typeid(StoreOperation), annotateStore},
       {typeid(alloca_op),      annotateAlloca},
       {typeid(malloc_op),      annotateMalloc},
       {typeid(free_op),        annotateFree},
       {typeid(CallOperation),  annotateCall},
       {typeid(Memcpy),         annotateMemcpy}
     });

  auto & operation = simpleNode.operation();
  if (nodes.find(typeid(operation)) == nodes.end())
    return;

  nodes[typeid(operation)](*this, simpleNode);
}

void
RegionAwareMemoryNodeProvider::AnnotateLoad(const jlm::LoadNode & loadNode)
{
  auto memoryNodes = Provisioning_->GetOutputNodes(*loadNode.GetAddressInput()->origin());
  auto & regionSummary = Provisioning_->GetRegionSummary(*loadNode.region());
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateStore(const jlm::StoreNode & storeNode)
{
  auto memoryNodes = Provisioning_->GetOutputNodes(*storeNode.GetAddressInput()->origin());
  auto & regionSummary = Provisioning_->GetRegionSummary(*storeNode.region());
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateAlloca(const jive::simple_node & allocaNode)
{
  JLM_ASSERT(jive::is<alloca_op>(allocaNode.operation()));

  auto & memoryNode = Provisioning_->GetPointsToGraph().GetAllocaNode(allocaNode);
  auto & regionSummary = Provisioning_->GetRegionSummary(*allocaNode.region());
  regionSummary.AddMemoryNodes({&memoryNode});
}

void
RegionAwareMemoryNodeProvider::AnnotateMalloc(const jive::simple_node & mallocNode)
{
  JLM_ASSERT(jive::is<malloc_op>(mallocNode.operation()));

  auto & memoryNode = Provisioning_->GetPointsToGraph().GetMallocNode(mallocNode);
  auto & regionSummary = Provisioning_->GetRegionSummary(*mallocNode.region());
  regionSummary.AddMemoryNodes({&memoryNode});
}

void
RegionAwareMemoryNodeProvider::AnnotateFree(const jive::simple_node & freeNode)
{
  JLM_ASSERT(jive::is<free_op>(freeNode.operation()));

  auto memoryNodes = Provisioning_->GetOutputNodes(*freeNode.input(0)->origin());
  auto & regionSummary = Provisioning_->GetRegionSummary(*freeNode.region());
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateCall(const CallNode & callNode)
{
  auto annotateNonRecursiveDirectCall = [](auto & provider, auto & callNode, auto & callTypeClassifier)
  {
    JLM_ASSERT(callTypeClassifier.GetCallType()  == CallTypeClassifier::CallType::NonRecursiveDirectCall);

    auto & regionSummary = provider.Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddNonRecursiveDirectCall(callNode);
  };
  auto annotateRecursiveDirectCall = [](auto & provider, auto & callNode, auto & callTypeClassifier)
  {
    JLM_ASSERT(callTypeClassifier.GetCallType() == CallTypeClassifier::CallType::RecursiveDirectCall);

    auto & regionSummary = provider.Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddRecursiveDirectCall(callNode);
  };
  auto annotateExternalCall = [](auto & provider, auto & callNode, auto & callTypeClassifier)
  {
    JLM_ASSERT(callTypeClassifier.GetCallType() == CallTypeClassifier::CallType::ExternalCall);

    auto & pointsToGraph = provider.Provisioning_->GetPointsToGraph();

    HashSet<const PointsToGraph::MemoryNode*> memoryNodes;
    memoryNodes.UnionWith(pointsToGraph.GetEscapedMemoryNodes());
    memoryNodes.Insert(&pointsToGraph.GetExternalMemoryNode());

    auto & import = callTypeClassifier.GetImport();
    auto & regionSummary = provider.Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddMemoryNodes(memoryNodes);
    if (!provider.Provisioning_->ContainsExternalFunctionNodes(import))
    {
      provider.Provisioning_->AddExternalFunctionNodes(import, memoryNodes);
    }
  };
  auto annotateIndirectCall = [](auto & provider, auto & callNode, auto & callTypeClassifier)
  {
    JLM_ASSERT(callTypeClassifier.GetCallType() == CallTypeClassifier::CallType::IndirectCall);

    auto & regionSummary = provider.Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddMemoryNodes({&provider.Provisioning_->GetPointsToGraph().GetExternalMemoryNode()});
    regionSummary.AddUnknownMemoryNodeReferences({&callNode});
  };

  static std::unordered_map<
    CallTypeClassifier::CallType,
    std::function<void(RegionAwareMemoryNodeProvider&, const CallNode&, const CallTypeClassifier&)>
  > callTypes
    ({
       {CallTypeClassifier::CallType::NonRecursiveDirectCall, annotateNonRecursiveDirectCall},
       {CallTypeClassifier::CallType::RecursiveDirectCall,    annotateRecursiveDirectCall},
       {CallTypeClassifier::CallType::IndirectCall,           annotateIndirectCall},
       {CallTypeClassifier::CallType::ExternalCall,           annotateExternalCall}
     });

  auto callTypeClassifier = CallNode::ClassifyCall(callNode);
  JLM_ASSERT(callTypes.find(callTypeClassifier->GetCallType()) != callTypes.end());
  callTypes[callTypeClassifier->GetCallType()](*this, callNode, *callTypeClassifier);
}

void
RegionAwareMemoryNodeProvider::AnnotateMemcpy(const jive::simple_node & memcpyNode)
{
  JLM_ASSERT(jive::is<Memcpy>(memcpyNode.operation()));

  auto & regionSummary = Provisioning_->GetRegionSummary(*memcpyNode.region());

  auto dstNodes = Provisioning_->GetOutputNodes(*memcpyNode.input(0)->origin());
  regionSummary.AddMemoryNodes(dstNodes);

  auto srcNodes = Provisioning_->GetOutputNodes(*memcpyNode.input(1)->origin());
  regionSummary.AddMemoryNodes(srcNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateStructuralNode(const jive::structural_node & structuralNode)
{
  if (jive::is<delta::operation>(&structuralNode))
  {
    /*
     * Nothing needs to be done for delta nodes.
     */
    return;
  }

  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    AnnotateRegion(*structuralNode.subregion(n));
  }
}

void
RegionAwareMemoryNodeProvider::Propagate(const jlm::RvsdgModule & rvsdgModule)
{
  jive::topdown_traverser traverser(rvsdgModule.Rvsdg().root());
  for (auto & node : traverser)
  {
    if (auto lambdaNode = dynamic_cast<const lambda::node*>(node))
    {
      PropagateRegion(*lambdaNode->subregion());
    }
    else if (auto phiNode = dynamic_cast<const phi::node*>(node))
    {
      PropagatePhi(*phiNode);
    }
    else if (dynamic_cast<const delta::node*>(node))
    {
      /*
       * Nothing needs to be done for delta nodes.
       */
      continue;
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }

  JLM_ASSERT(RegionAwareMemoryNodeProvisioning::CheckInvariants(*Provisioning_));
}

void
RegionAwareMemoryNodeProvider::PropagatePhi(const phi::node & phiNode)
{
  std::function<void(
    const jive::region&,
    const HashSet<const PointsToGraph::MemoryNode*>&,
    const HashSet<const jive::simple_node*>&
  )> assignAndPropagateMemoryNodes = [&](
    const jive::region & region,
    const HashSet<const PointsToGraph::MemoryNode*> & memoryNodes,
    const HashSet<const jive::simple_node*> & unknownMemoryNodeReferences)
  {
    auto & regionSummary = Provisioning_->GetRegionSummary(region);
    for (auto structuralNode : regionSummary.GetStructuralNodes().Items())
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        auto & subregion = *structuralNode->subregion(n);
        assignAndPropagateMemoryNodes(subregion, memoryNodes, unknownMemoryNodeReferences);

        auto & subregionSummary = Provisioning_->GetRegionSummary(subregion);
        if (subregionSummary.GetRecursiveCalls().Size() != 0)
        {
          subregionSummary.AddMemoryNodes(memoryNodes);
          subregionSummary.AddUnknownMemoryNodeReferences(unknownMemoryNodeReferences);
          RegionSummary::Propagate(regionSummary, subregionSummary);
        }
      }
    }
  };

  auto & phiNodeSubregion = *phiNode.subregion();
  PropagateRegion(phiNodeSubregion);

  auto lambdaNodes = ExtractLambdaNodes(phiNode);

  HashSet<const PointsToGraph::MemoryNode*> memoryNodes;
  HashSet<const jive::simple_node*> unknownMemoryNodeReferences;
  for (auto & lambdaNode : lambdaNodes)
  {
    auto & regionSummary = Provisioning_->GetRegionSummary(*lambdaNode->subregion());
    memoryNodes.UnionWith(regionSummary.GetMemoryNodes());
    unknownMemoryNodeReferences.UnionWith(regionSummary.GetUnknownMemoryNodeReferences());
  }

  assignAndPropagateMemoryNodes(phiNodeSubregion, memoryNodes, unknownMemoryNodeReferences);
}

void
RegionAwareMemoryNodeProvider::PropagateRegion(const jive::region & region)
{
  auto & regionSummary = Provisioning_->GetRegionSummary(region);
  for (auto & structuralNode : regionSummary.GetStructuralNodes().Items())
  {
    for (size_t n = 0; n < structuralNode->nsubregions(); n++)
    {
      auto & subregion = *structuralNode->subregion(n);
      PropagateRegion(subregion);

      auto & subregionSummary = Provisioning_->GetRegionSummary(subregion);
      RegionSummary::Propagate(regionSummary, subregionSummary);
    }
  }

  for (auto & callNode : regionSummary.GetNonRecursiveCalls().Items())
  {
    auto callTypeClassifier = CallNode::ClassifyCall(*callNode);
    auto & lambdaRegion = *callTypeClassifier->GetLambdaOutput().node()->subregion();
    auto & lambdaRegionSummary = Provisioning_->GetRegionSummary(lambdaRegion);

    RegionSummary::Propagate(regionSummary, lambdaRegionSummary);
  }
}

void
RegionAwareMemoryNodeProvider::ResolveUnknownMemoryNodeReferences(const jlm::RvsdgModule & rvsdgModule)
{
  auto ResolveLambda = [&](const lambda::node & lambda)
  {
    auto & lambdaRegionSummary = Provisioning_->GetRegionSummary(*lambda.subregion());

    for (auto node : lambdaRegionSummary.GetUnknownMemoryNodeReferences().Items())
    {
      auto & nodeRegion = *node->region();
      auto & nodeRegionSummary = Provisioning_->GetRegionSummary(nodeRegion);
      nodeRegionSummary.AddMemoryNodes(lambdaRegionSummary.GetMemoryNodes());
    }
  };

  auto nodes = ExtractRvsdgTailNodes(rvsdgModule);
  for (auto & node : nodes)
  {
    if (auto lambdaNode = dynamic_cast<const lambda::node*>(node))
    {
      ResolveLambda(*lambdaNode);
    }
    else if (auto phiNode = dynamic_cast<const phi::node*>(node))
    {
      auto lambdaNodes = ExtractLambdaNodes(*phiNode);
      for (auto & lambda : lambdaNodes)
      {
        ResolveLambda(*lambda);
      }
    }
    else if (dynamic_cast<const delta::node*>(node))
    {
      /*
       * Nothing needs to be done for delta nodes.
       */
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }
}

std::vector<const lambda::node*>
RegionAwareMemoryNodeProvider::ExtractLambdaNodes(const phi::node & phiNode)
{
  std::function<void(const phi::node&, std::vector<const lambda::node*>&)> extractLambdaNodes = [&](
      auto & phiNode,
      auto & lambdaNodes)
  {
    for (auto & node : phiNode.subregion()->nodes)
    {
      if (auto lambdaNode = dynamic_cast<const lambda::node*>(&node))
      {
        lambdaNodes.push_back(lambdaNode);
      }
      else if (auto innerPhiNode = dynamic_cast<const phi::node*>(&node))
      {
        extractLambdaNodes(*innerPhiNode, lambdaNodes);
      }
    }
  };

  std::vector<const lambda::node*> lambdaNodes;
  extractLambdaNodes(phiNode, lambdaNodes);

  return lambdaNodes;
}

std::vector<const jive::node*>
RegionAwareMemoryNodeProvider::ExtractRvsdgTailNodes(const jlm::RvsdgModule & rvsdgModule)
{
  auto IsOnlyExported = [](const jive::output & output)
  {
    auto IsRootRegionExport = [](const jive::input * input)
    {
      if (!input->region()->IsRootRegion())
      {
        return false;
      }

      if (jive::node_input::node(*input))
      {
        return false;
      }

      return true;
    };

    return std::all_of(
      output.begin(),
      output.end(),
      IsRootRegionExport);
  };

  auto & rootRegion = *rvsdgModule.Rvsdg().root();

  std::vector<const jive::node*> nodes;
  for (auto & node : rootRegion.bottom_nodes)
  {
      nodes.push_back(&node);
  }

  for (size_t n = 0; n < rootRegion.nresults(); n++)
  {
    auto output = rootRegion.result(n)->origin();
    if (IsOnlyExported(*output))
    {
      nodes.push_back(jive::node_output::node(output));
    }
  }

  return nodes;
}

}