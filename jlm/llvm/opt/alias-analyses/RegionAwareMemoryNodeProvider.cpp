/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/FunctionPointer.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/RegionAwareMemoryNodeProvider.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm::aa
{

/** \brief Region-aware memory node provider statistics
 *
 * The statistics collected when running the region-aware memory node provider.
 *
 * @see RegionAwareMemoryNodeProvider
 */
class RegionAwareMemoryNodeProvider::Statistics final : public util::Statistics
{
  const char * NumRvsdgRegionsLabel_ = "#RvsdgRegions";

  const char * AnnotationTimerLabel_ = "AnnotationTime";
  const char * PropagationPass1TimerLabel_ = "PropagationPass1Time";
  const char * PropagationPass2TimerLabel_ = "PropagationPass2Time";
  const char * ResolveUnknownMemoryReferenceTimerLabel_ = "ResolveUnknownMemoryReferenceTime";

public:
  ~Statistics() override = default;

  explicit Statistics(
      const util::StatisticsCollector & statisticsCollector,
      const rvsdg::RvsdgModule & rvsdgModule,
      const PointsToGraph & pointsToGraph)
      : util::Statistics(
            Statistics::Id::RegionAwareMemoryNodeProvisioning,
            rvsdgModule.SourceFilePath().value()),
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
  StartAnnotationStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(AnnotationTimerLabel_).start();
  }

  void
  StopAnnotationStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(AnnotationTimerLabel_).stop();
  }

  void
  StartPropagationPass1Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(PropagationPass1TimerLabel_).start();
  }

  void
  StopPropagationPass1Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(PropagationPass1TimerLabel_).stop();
  }

  void
  StartResolveUnknownMemoryNodeReferencesStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(ResolveUnknownMemoryReferenceTimerLabel_).start();
  }

  void
  StopResolveUnknownMemoryNodeReferencesStatistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(ResolveUnknownMemoryReferenceTimerLabel_).stop();
  }

  void
  StartPropagationPass2Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    AddTimer(PropagationPass2TimerLabel_).start();
  }

  void
  StopPropagationPass2Statistics() noexcept
  {
    if (!IsDemanded())
      return;

    GetTimer(PropagationPass2TimerLabel_).stop();
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

class RegionSummary final
{
public:
  explicit RegionSummary(const rvsdg::Region & region)
      : Region_(&region)
  {}

  RegionSummary(const RegionSummary &) = delete;

  RegionSummary(RegionSummary &&) = delete;

  RegionSummary &
  operator=(const RegionSummary &) = delete;

  RegionSummary &
  operator=(RegionSummary &&) = delete;

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetMemoryNodes() const
  {
    return MemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const rvsdg::SimpleNode *> &
  GetUnknownMemoryNodeReferences() const noexcept
  {
    return UnknownMemoryNodeReferences_;
  }

  const util::HashSet<const CallNode *> &
  GetNonRecursiveCalls() const
  {
    return NonRecursiveCalls_;
  }

  const util::HashSet<const CallNode *> &
  GetRecursiveCalls() const
  {
    return RecursiveCalls_;
  }

  const util::HashSet<const rvsdg::StructuralNode *> &
  GetStructuralNodes() const
  {
    return StructuralNodes_;
  }

  void
  AddMemoryNodes(const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes)
  {
    MemoryNodes_.UnionWith(memoryNodes);
  }

  void
  AddUnknownMemoryNodeReferences(const util::HashSet<const rvsdg::SimpleNode *> & nodes)
  {
    UnknownMemoryNodeReferences_.UnionWith(nodes);
  }

  void
  AddNonRecursiveDirectCall(const CallNode & callNode)
  {
    JLM_ASSERT(
        CallNode::ClassifyCall(callNode)->GetCallType()
        == CallTypeClassifier::CallType::NonRecursiveDirectCall);
    NonRecursiveCalls_.Insert(&callNode);
  }

  void
  AddRecursiveDirectCall(const CallNode & callNode)
  {
    JLM_ASSERT(
        CallNode::ClassifyCall(callNode)->GetCallType()
        == CallTypeClassifier::CallType::RecursiveDirectCall);
    RecursiveCalls_.Insert(&callNode);
  }

  void
  AddStructuralNode(const rvsdg::StructuralNode & structuralNode)
  {
    StructuralNodes_.Insert(&structuralNode);
  }

  [[nodiscard]] const rvsdg::Region &
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
  Create(const rvsdg::Region & region)
  {
    return std::make_unique<RegionSummary>(region);
  }

private:
  const rvsdg::Region * Region_;
  util::HashSet<const PointsToGraph::MemoryNode *> MemoryNodes_;
  util::HashSet<const rvsdg::SimpleNode *> UnknownMemoryNodeReferences_;

  util::HashSet<const CallNode *> RecursiveCalls_;
  util::HashSet<const CallNode *> NonRecursiveCalls_;
  util::HashSet<const rvsdg::StructuralNode *> StructuralNodes_;
};

/** \brief Memory node provisioning of region-aware memory node provider
 *
 */
class RegionAwareMemoryNodeProvisioning final : public MemoryNodeProvisioning
{
  using RegionSummaryMap =
      std::unordered_map<const rvsdg::Region *, std::unique_ptr<RegionSummary>>;

  class RegionSummaryConstIterator final
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const RegionSummary *;
    using difference_type = std::ptrdiff_t;
    using pointer = const RegionSummary **;
    using reference = const RegionSummary *&;

  private:
    friend RegionAwareMemoryNodeProvisioning;

    explicit RegionSummaryConstIterator(const RegionSummaryMap::const_iterator & it)
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
    RegionSummaryMap::const_iterator it_;
  };

  using RegionSummaryConstRange = util::IteratorRange<RegionSummaryConstIterator>;

public:
  explicit RegionAwareMemoryNodeProvisioning(const PointsToGraph & pointsToGraph)
      : PointsToGraph_(pointsToGraph)
  {}

  RegionAwareMemoryNodeProvisioning(const RegionAwareMemoryNodeProvisioning &) = delete;

  RegionAwareMemoryNodeProvisioning(RegionAwareMemoryNodeProvisioning &&) = delete;

  RegionAwareMemoryNodeProvisioning &
  operator=(const RegionAwareMemoryNodeProvisioning &) = delete;

  RegionAwareMemoryNodeProvisioning &
  operator=(RegionAwareMemoryNodeProvisioning &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionEntryNodes(const rvsdg::Region & region) const override
  {
    auto & regionSummary = GetRegionSummary(region);
    return regionSummary.GetMemoryNodes();
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetRegionExitNodes(const rvsdg::Region & region) const override
  {
    auto & regionSummary = GetRegionSummary(region);
    return regionSummary.GetMemoryNodes();
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
      auto & import = callTypeClassifier->GetImport();
      return GetExternalFunctionNodes(import);
    }
    else if (callTypeClassifier->IsIndirectCall())
    {
      return GetIndirectCallNodes(callNode);
    }

    JLM_UNREACHABLE("Unhandled call type.");
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
      auto & import = callTypeClassifier->GetImport();
      return GetExternalFunctionNodes(import);
    }
    else if (callTypeClassifier->IsIndirectCall())
    {
      return GetIndirectCallNodes(callNode);
    }

    JLM_UNREACHABLE("Unhandled call type!");
  }

  [[nodiscard]] util::HashSet<const PointsToGraph::MemoryNode *>
  GetOutputNodes(const rvsdg::output & output) const override
  {
    JLM_ASSERT(is<PointerType>(output.type()));

    util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
    auto registerNode = &PointsToGraph_.GetRegisterNode(output);
    for (auto & memoryNode : registerNode->Targets())
      memoryNodes.Insert(&memoryNode);

    return memoryNodes;
  }

  RegionSummaryConstRange
  GetRegionSummaries() const
  {
    return { RegionSummaryConstIterator(RegionSummaries_.begin()),
             RegionSummaryConstIterator(RegionSummaries_.end()) };
  }

  [[nodiscard]] bool
  ContainsRegionSummary(const rvsdg::Region & region) const
  {
    return RegionSummaries_.find(&region) != RegionSummaries_.end();
  }

  bool
  ContainsExternalFunctionNodes(const rvsdg::RegionArgument & import) const
  {
    return ExternalFunctionNodes_.find(&import) != ExternalFunctionNodes_.end();
  }

  [[nodiscard]] RegionSummary &
  GetRegionSummary(const rvsdg::Region & region) const
  {
    JLM_ASSERT(ContainsRegionSummary(region));
    return *RegionSummaries_.find(&region)->second;
  }

  [[nodiscard]] RegionSummary *
  TryGetRegionSummary(const rvsdg::Region & region) const
  {
    return ContainsRegionSummary(region) ? &GetRegionSummary(region) : nullptr;
  }

  const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetExternalFunctionNodes(const rvsdg::RegionArgument & import) const
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
      const rvsdg::RegionArgument & import,
      util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes)
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
   * 1. The collections of memory nodes of all subregions of a structural node should be contained
   * in the collection of memory nodes of the region the structural node is contained in.
   *
   * 2. The collections of unknown memory reference nodes of all subregions of a structural node
   * should be contained in the collection of unknown memory reference nodes of the region the
   * structural node is contained in.
   *
   * @param provisioning \see RegionAwareMemoryNodeProvisioning
   * @return Returns true if all invariants are fulfilled, otherwise false.
   */
  static bool
  CheckStructuralNodeInvariants(const RegionAwareMemoryNodeProvisioning & provisioning)
  {
    for (auto & regionSummary : provisioning.GetRegionSummaries())
    {
      for (auto & structuralNode : regionSummary.GetStructuralNodes().Items())
      {
        auto & regionMemoryNodes = regionSummary.GetMemoryNodes();
        auto & regionUnknownMemoryNodeReferences = regionSummary.GetUnknownMemoryNodeReferences();

        for (size_t n = 0; n < structuralNode->nsubregions(); n++)
        {
          auto & subregion = *structuralNode->subregion(n);
          auto & subregionSummary = provisioning.GetRegionSummary(subregion);
          auto & subregionMemoryNodes = subregionSummary.GetMemoryNodes();
          auto & subregionUnknownMemoryNodeReferences =
              subregionSummary.GetUnknownMemoryNodeReferences();

          if (!subregionMemoryNodes.IsSubsetOf(regionMemoryNodes)
              || !subregionUnknownMemoryNodeReferences.IsSubsetOf(
                  regionUnknownMemoryNodeReferences))
          {
            return false;
          }
        }
      }
    }

    return true;
  }

  /**
   * This function checks the following two invariants:
   *
   * 1. The collection of memory nodes of a lambda region should be contained in the collection of
   * memory nodes of the regions of all direct calls to this lambda.
   *
   * 2. The collection of unknown memory reference nodes of a lambda region should be contained in
   * the collection of memory nodes of the regions of all direct calls to this lambda.
   *
   * @param provisioning \see RegionAwareMemoryNodeProvisioning
   * @return Returns true if all invariants are fulfilled, otherwise false.
   */
  static bool
  CheckCallInvariants(const RegionAwareMemoryNodeProvisioning & provisioning)
  {
    auto checkInvariants = [&](auto & callNode)
    {
      auto & regionSummary = provisioning.GetRegionSummary(*callNode.region());
      auto & regionMemoryNodes = regionSummary.GetMemoryNodes();
      auto & regionUnknownMemoryNodeReferences = regionSummary.GetUnknownMemoryNodeReferences();

      auto callTypeClassifier = CallNode::ClassifyCall(callNode);
      auto & lambdaRegion =
          *rvsdg::AssertGetOwnerNode<llvm::lambda::node>(callTypeClassifier->GetLambdaOutput())
               .subregion();
      auto & lambdaRegionSummary = provisioning.GetRegionSummary(lambdaRegion);
      auto & lambdaRegionMemoryNodes = lambdaRegionSummary.GetMemoryNodes();
      auto & lambdaRegionUnknownMemoryNodeReferences =
          lambdaRegionSummary.GetUnknownMemoryNodeReferences();

      return lambdaRegionMemoryNodes.IsSubsetOf(regionMemoryNodes)
          && lambdaRegionUnknownMemoryNodeReferences.IsSubsetOf(regionUnknownMemoryNodeReferences);
    };

    for (auto & regionSummary : provisioning.GetRegionSummaries())
    {
      for (auto & callNode : regionSummary.GetNonRecursiveCalls().Items())
      {
        if (!checkInvariants(*callNode))
        {
          return false;
        }
      }

      for (auto & callNode : regionSummary.GetRecursiveCalls().Items())
      {
        if (!checkInvariants(*callNode))
        {
          return false;
        }
      }
    }

    return true;
  }

private:
  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetIndirectCallNodes(const CallNode & callNode) const
  {
    // We have no idea about the function of an indirect call. This means that we have to be
    // conservative and sequentialize this indirect call with respect to all memory references that
    // came before and after it. These references should have been routed through the region the
    // indirect call node lives in. Thus, we can just use here the memory nodes of the region of the
    // indirect call node.
    auto & regionSummary = GetRegionSummary(*callNode.region());
    return regionSummary.GetMemoryNodes();
  }

  RegionSummaryMap RegionSummaries_;
  const PointsToGraph & PointsToGraph_;
  std::
      unordered_map<const rvsdg::RegionArgument *, util::HashSet<const PointsToGraph::MemoryNode *>>
          ExternalFunctionNodes_;
};

RegionAwareMemoryNodeProvider::~RegionAwareMemoryNodeProvider() noexcept = default;

RegionAwareMemoryNodeProvider::RegionAwareMemoryNodeProvider() = default;

std::unique_ptr<MemoryNodeProvisioning>
RegionAwareMemoryNodeProvider::ProvisionMemoryNodes(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  Provisioning_ = RegionAwareMemoryNodeProvisioning::Create(pointsToGraph);
  auto statistics = Statistics::Create(statisticsCollector, rvsdgModule, pointsToGraph);

  statistics->StartAnnotationStatistics();
  AnnotateRegion(rvsdgModule.Rvsdg().GetRootRegion());
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
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  RegionAwareMemoryNodeProvider provider;
  return provider.ProvisionMemoryNodes(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<MemoryNodeProvisioning>
RegionAwareMemoryNodeProvider::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph)
{
  util::StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

void
RegionAwareMemoryNodeProvider::AnnotateRegion(rvsdg::Region & region)
{
  if (ShouldCreateRegionSummary(region))
  {
    Provisioning_->AddRegionSummary(RegionSummary::Create(region));
  }

  for (auto & node : region.Nodes())
  {
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      AnnotateStructuralNode(*structuralNode);
    }
    else if (auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
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
RegionAwareMemoryNodeProvider::AnnotateSimpleNode(const rvsdg::SimpleNode & simpleNode)
{
  if (auto loadNode = dynamic_cast<const LoadNode *>(&simpleNode))
  {
    AnnotateLoad(*loadNode);
  }
  else if (auto storeNode = dynamic_cast<const StoreNode *>(&simpleNode))
  {
    AnnotateStore(*storeNode);
  }
  else if (auto callNode = dynamic_cast<const CallNode *>(&simpleNode))
  {
    AnnotateCall(*callNode);
  }
  else if (is<alloca_op>(&simpleNode))
  {
    AnnotateAlloca(simpleNode);
  }
  else if (is<malloc_op>(&simpleNode))
  {
    AnnotateMalloc(simpleNode);
  }
  else if (is<FreeOperation>(&simpleNode))
  {
    AnnotateFree(simpleNode);
  }
  else if (is<MemCpyOperation>(&simpleNode))
  {
    AnnotateMemcpy(simpleNode);
  }
}

void
RegionAwareMemoryNodeProvider::AnnotateLoad(const LoadNode & loadNode)
{
  auto memoryNodes = Provisioning_->GetOutputNodes(*loadNode.GetAddressInput().origin());
  auto & regionSummary = Provisioning_->GetRegionSummary(*loadNode.region());
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateStore(const StoreNode & storeNode)
{
  auto memoryNodes = Provisioning_->GetOutputNodes(*storeNode.GetAddressInput().origin());
  auto & regionSummary = Provisioning_->GetRegionSummary(*storeNode.region());
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateAlloca(const rvsdg::SimpleNode & allocaNode)
{
  JLM_ASSERT(is<alloca_op>(allocaNode.GetOperation()));

  auto & memoryNode = Provisioning_->GetPointsToGraph().GetAllocaNode(allocaNode);
  auto & regionSummary = Provisioning_->GetRegionSummary(*allocaNode.region());
  regionSummary.AddMemoryNodes({ &memoryNode });
}

void
RegionAwareMemoryNodeProvider::AnnotateMalloc(const rvsdg::SimpleNode & mallocNode)
{
  JLM_ASSERT(is<malloc_op>(mallocNode.GetOperation()));

  auto & memoryNode = Provisioning_->GetPointsToGraph().GetMallocNode(mallocNode);
  auto & regionSummary = Provisioning_->GetRegionSummary(*mallocNode.region());
  regionSummary.AddMemoryNodes({ &memoryNode });
}

void
RegionAwareMemoryNodeProvider::AnnotateFree(const rvsdg::SimpleNode & freeNode)
{
  JLM_ASSERT(is<FreeOperation>(freeNode.GetOperation()));

  auto memoryNodes = Provisioning_->GetOutputNodes(*freeNode.input(0)->origin());
  auto & regionSummary = Provisioning_->GetRegionSummary(*freeNode.region());
  regionSummary.AddMemoryNodes(memoryNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateCall(const CallNode & callNode)
{
  auto callTypeClassifier = CallNode::ClassifyCall(callNode);
  auto callType = callTypeClassifier->GetCallType();

  if (callType == CallTypeClassifier::CallType::NonRecursiveDirectCall)
  {
    auto & regionSummary = Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddNonRecursiveDirectCall(callNode);
  }
  else if (callType == CallTypeClassifier::CallType::RecursiveDirectCall)
  {
    auto & regionSummary = Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddRecursiveDirectCall(callNode);
  }
  else if (callType == CallTypeClassifier::CallType::IndirectCall)
  {
    auto & regionSummary = Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddMemoryNodes({ &Provisioning_->GetPointsToGraph().GetExternalMemoryNode() });
    regionSummary.AddUnknownMemoryNodeReferences({ &callNode });
  }
  else if (callType == CallTypeClassifier::CallType::ExternalCall)
  {
    auto & pointsToGraph = Provisioning_->GetPointsToGraph();

    util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
    memoryNodes.UnionWith(pointsToGraph.GetEscapedMemoryNodes());
    memoryNodes.Insert(&pointsToGraph.GetExternalMemoryNode());

    auto & import = callTypeClassifier->GetImport();
    auto & regionSummary = Provisioning_->GetRegionSummary(*callNode.region());
    regionSummary.AddMemoryNodes(memoryNodes);
    if (!Provisioning_->ContainsExternalFunctionNodes(import))
    {
      Provisioning_->AddExternalFunctionNodes(import, memoryNodes);
    }
  }
  else
  {
    JLM_UNREACHABLE("Unhandled call type!");
  }
}

void
RegionAwareMemoryNodeProvider::AnnotateMemcpy(const rvsdg::SimpleNode & memcpyNode)
{
  JLM_ASSERT(is<MemCpyOperation>(memcpyNode.GetOperation()));

  auto & regionSummary = Provisioning_->GetRegionSummary(*memcpyNode.region());

  auto dstNodes = Provisioning_->GetOutputNodes(*memcpyNode.input(0)->origin());
  regionSummary.AddMemoryNodes(dstNodes);

  auto srcNodes = Provisioning_->GetOutputNodes(*memcpyNode.input(1)->origin());
  regionSummary.AddMemoryNodes(srcNodes);
}

void
RegionAwareMemoryNodeProvider::AnnotateStructuralNode(const rvsdg::StructuralNode & structuralNode)
{
  if (is<delta::operation>(&structuralNode))
  {
    // Nothing needs to be done for delta nodes.
    return;
  }

  if (auto regionSummary = Provisioning_->TryGetRegionSummary(*structuralNode.region()))
  {
    regionSummary->AddStructuralNode(structuralNode);
  }

  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    AnnotateRegion(*structuralNode.subregion(n));
  }
}

void
RegionAwareMemoryNodeProvider::Propagate(const rvsdg::RvsdgModule & rvsdgModule)
{
  rvsdg::TopDownTraverser traverser(&rvsdgModule.Rvsdg().GetRootRegion());
  for (auto & node : traverser)
  {
    if (auto lambdaNode = dynamic_cast<const lambda::node *>(node))
    {
      PropagateRegion(*lambdaNode->subregion());
    }
    else if (auto phiNode = dynamic_cast<const phi::node *>(node))
    {
      PropagatePhi(*phiNode);
    }
    else if (dynamic_cast<const delta::node *>(node))
    {
      // Nothing needs to be done for delta nodes.
      continue;
    }
    else if (
        is<FunctionToPointerOperation>(node->GetOperation())
        || is<PointerToFunctionOperation>(node->GetOperation()))
    {
      // Few operators may appear as top-level constructs and simply must
      // be ignored.
      continue;
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type!");
    }
  }

  JLM_ASSERT(RegionAwareMemoryNodeProvisioning::CheckStructuralNodeInvariants(*Provisioning_));
  JLM_ASSERT(RegionAwareMemoryNodeProvisioning::CheckCallInvariants(*Provisioning_));
}

void
RegionAwareMemoryNodeProvider::PropagatePhi(const phi::node & phiNode)
{
  auto lambdaNodes = phi::node::ExtractLambdaNodes(phiNode);
  if (lambdaNodes.empty())
  {
    // Nothing needs to be done if the phi node only contains delta nodes.
    return;
  }

  auto & phiNodeSubregion = *phiNode.subregion();
  PropagateRegion(phiNodeSubregion);

  util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
  util::HashSet<const rvsdg::SimpleNode *> unknownMemoryNodeReferences;
  for (auto & lambdaNode : lambdaNodes)
  {
    auto & regionSummary = Provisioning_->GetRegionSummary(*lambdaNode->subregion());
    memoryNodes.UnionWith(regionSummary.GetMemoryNodes());
    unknownMemoryNodeReferences.UnionWith(regionSummary.GetUnknownMemoryNodeReferences());
  }

  AssignAndPropagateMemoryNodes(phiNodeSubregion, memoryNodes, unknownMemoryNodeReferences);
}

void
RegionAwareMemoryNodeProvider::AssignAndPropagateMemoryNodes(
    const rvsdg::Region & region,
    const util::HashSet<const PointsToGraph::MemoryNode *> & memoryNodes,
    const util::HashSet<const rvsdg::SimpleNode *> & unknownMemoryNodeReferences)
{
  auto & regionSummary = Provisioning_->GetRegionSummary(region);
  for (auto structuralNode : regionSummary.GetStructuralNodes().Items())
  {
    for (size_t n = 0; n < structuralNode->nsubregions(); n++)
    {
      auto & subregion = *structuralNode->subregion(n);
      AssignAndPropagateMemoryNodes(subregion, memoryNodes, unknownMemoryNodeReferences);

      auto & subregionSummary = Provisioning_->GetRegionSummary(subregion);
      if (subregionSummary.GetRecursiveCalls().Size() != 0)
      {
        subregionSummary.AddMemoryNodes(memoryNodes);
        subregionSummary.AddUnknownMemoryNodeReferences(unknownMemoryNodeReferences);
      }
      RegionSummary::Propagate(regionSummary, subregionSummary);
    }
  }
}

void
RegionAwareMemoryNodeProvider::PropagateRegion(const rvsdg::Region & region)
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
    auto & lambdaRegion =
        *rvsdg::AssertGetOwnerNode<lambda::node>(callTypeClassifier->GetLambdaOutput()).subregion();
    auto & lambdaRegionSummary = Provisioning_->GetRegionSummary(lambdaRegion);

    RegionSummary::Propagate(regionSummary, lambdaRegionSummary);
  }
}

void
RegionAwareMemoryNodeProvider::ResolveUnknownMemoryNodeReferences(
    const rvsdg::RvsdgModule & rvsdgModule)
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

  auto nodes = rvsdg::Graph::ExtractTailNodes(rvsdgModule.Rvsdg());
  for (auto & node : nodes)
  {
    if (auto lambdaNode = dynamic_cast<const lambda::node *>(node))
    {
      ResolveLambda(*lambdaNode);
    }
    else if (auto phiNode = dynamic_cast<const phi::node *>(node))
    {
      auto lambdaNodes = phi::node::ExtractLambdaNodes(*phiNode);
      for (auto & lambda : lambdaNodes)
      {
        ResolveLambda(*lambda);
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

bool
RegionAwareMemoryNodeProvider::ShouldCreateRegionSummary(const rvsdg::Region & region)
{
  return !region.IsRootRegion() && !is<phi_op>(region.node())
      && !is<delta::operation>(region.node());
}

std::string
RegionAwareMemoryNodeProvider::ToRegionTree(
    const rvsdg::Graph & rvsdg,
    const RegionAwareMemoryNodeProvisioning & provisioning)
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
    if (provisioning.ContainsRegionSummary(*region))
    {
      auto & regionSummary = provisioning.GetRegionSummary(*region);
      auto & memoryNodes = regionSummary.GetMemoryNodes();
      subtree += util::strfmt(indent(depth), "MemoryNodes: ", toString(memoryNodes), "\n");
    }

    for (const auto & node : region->Nodes())
    {
      if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
      {
        subtree += util::strfmt(indent(depth), structuralNode->GetOperation().debug_string(), "\n");
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
