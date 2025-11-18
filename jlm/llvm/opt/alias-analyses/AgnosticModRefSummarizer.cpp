/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/MatchType.hpp>

namespace jlm::llvm::aa
{

/** \brief Mod/Ref summary of agnostic mod/ref summarizer
 *
 */
class AgnosticModRefSummary final : public ModRefSummary
{
public:
  using SimpleNodeModRefMap = std::
      unordered_map<const rvsdg::SimpleNode *, util::HashSet<PointsToGraph::NodeIndex>>;

  ~AgnosticModRefSummary() noexcept override = default;

private:
  AgnosticModRefSummary(
      const PointsToGraph & pointsToGraph,
      util::HashSet<PointsToGraph::NodeIndex> allMemoryNodes)
      : PointsToGraph_(pointsToGraph),
        AllMemoryNodes_(std::move(allMemoryNodes))
  {}

public:
  AgnosticModRefSummary(const AgnosticModRefSummary &) = delete;

  AgnosticModRefSummary(AgnosticModRefSummary &&) = delete;

  AgnosticModRefSummary &
  operator=(const AgnosticModRefSummary &) = delete;

  AgnosticModRefSummary &
  operator=(AgnosticModRefSummary &&) = delete;

  [[nodiscard]] const PointsToGraph &
  GetPointsToGraph() const noexcept override
  {
    return PointsToGraph_;
  }

  void
  SetSimpleNodeModRef(
      const rvsdg::SimpleNode & node,
      util::HashSet<PointsToGraph::NodeIndex> modRefSet)
  {
    JLM_ASSERT(SimpleNodeModRefs_.find(&node) == SimpleNodeModRefs_.end());
    SimpleNodeModRefs_[&node] = std::move(modRefSet);
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const override
  {
    if (const auto it = SimpleNodeModRefs_.find(&node); it != SimpleNodeModRefs_.end())
    {
      return it->second;
    }
    if (is<CallOperation>(&node))
    {
      return AllMemoryNodes_;
    }
    throw std::logic_error("Unhandled node type.");
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetGammaEntryModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetGammaExitModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetThetaModRef([[maybe_unused]] const rvsdg::ThetaNode & theta) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetLambdaEntryModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<PointsToGraph::NodeIndex> &
  GetLambdaExitModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    return AllMemoryNodes_;
  }

  static std::unique_ptr<AgnosticModRefSummary>
  Create(
      const PointsToGraph & pointsToGraph,
      util::HashSet<PointsToGraph::NodeIndex> memoryNodes)
  {
    return std::unique_ptr<AgnosticModRefSummary>(
        new AgnosticModRefSummary(pointsToGraph, std::move(memoryNodes)));
  }

private:
  const PointsToGraph & PointsToGraph_;
  SimpleNodeModRefMap SimpleNodeModRefs_;
  util::HashSet<PointsToGraph::NodeIndex> AllMemoryNodes_;
};

AgnosticModRefSummarizer::AgnosticModRefSummarizer() = default;

AgnosticModRefSummarizer::~AgnosticModRefSummarizer() = default;

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::SummarizeModRefs(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  auto statistics =
      Statistics::Create(rvsdgModule.SourceFilePath().value(), statisticsCollector, pointsToGraph);
  statistics->StartCollecting();

  auto allMemoryNodes = GetAllMemoryNodes(pointsToGraph);

  ModRefSummary_ = AgnosticModRefSummary::Create(pointsToGraph, std::move(allMemoryNodes));

  // Create ModRefSets for SimpleNodes that affect memory
  AnnotateRegion(rvsdgModule.Rvsdg().GetRootRegion());

  statistics->StopCollecting();
  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return std::move(ModRefSummary_);
}

util::HashSet<PointsToGraph::NodeIndex>
AgnosticModRefSummarizer::GetAllMemoryNodes(const PointsToGraph & pointsToGraph)
{
  util::HashSet<PointsToGraph::NodeIndex> memoryNodes;
  for (const auto allocaNode : pointsToGraph.allocaNodes())
    memoryNodes.insert(allocaNode);

  for (const auto deltaNode : pointsToGraph.deltaNodes())
    memoryNodes.insert(deltaNode);

  for (const auto lambdaNode : pointsToGraph.lambdaNodes())
    memoryNodes.insert(lambdaNode);

  for (const auto mallocNode : pointsToGraph.mallocNodes())
    memoryNodes.insert(mallocNode);

  for (const auto importNode : pointsToGraph.importNodes())
    memoryNodes.insert(importNode);

  JLM_ASSERT(memoryNodes.Size() == pointsToGraph.numMemoryNodes());

  return memoryNodes;
}

void
AgnosticModRefSummarizer::AnnotateRegion(const rvsdg::Region & region)
{
  for (const auto & node : region.Nodes())
  {
    rvsdg::MatchTypeOrFail(
        node,
        [&](const rvsdg::SimpleNode & simpleNode)
        {
          AnnotateSimpleNode(simpleNode);
        },
        [&](const rvsdg::StructuralNode & structuralNode)
        {
          for (const auto & subregion : structuralNode.Subregions())
          {
            AnnotateRegion(subregion);
          }
        });
  }
}

void
AgnosticModRefSummarizer::AddPointerTargetsToModRefSet(
    const rvsdg::Output & output,
    util::HashSet<PointsToGraph::NodeIndex> & modRefSet) const
{
  const auto & pointsToGraph = ModRefSummary_->GetPointsToGraph();
  JLM_ASSERT(IsPointerCompatible(output));
  const auto & addressReg = pointsToGraph.getNodeForRegister(output);
  for (const auto target : pointsToGraph.getExplicitTargets(addressReg).Items())
  {
    modRefSet.insert(target);
  }
  if (pointsToGraph.isTargetingAllExternallyAvailable(addressReg))
  {
    for (const auto implicitTarget : pointsToGraph.getExternallyAvailableNodes())
    {
      modRefSet.insert(implicitTarget);
    }
  }
}

void
AgnosticModRefSummarizer::AnnotateSimpleNode(const rvsdg::SimpleNode & node)
{
  if (is<StoreOperation>(&node))
  {
    const auto & address = *StoreOperation::AddressInput(node).origin();
    util::HashSet<PointsToGraph::NodeIndex> modRefSet;
    AddPointerTargetsToModRefSet(address, modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<LoadOperation>(&node))
  {
    const auto & address = *LoadOperation::AddressInput(node).origin();
    util::HashSet<PointsToGraph::NodeIndex> modRefSet;
    AddPointerTargetsToModRefSet(address, modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<MemCpyOperation>(&node))
  {
    util::HashSet<PointsToGraph::NodeIndex> modRefSet;
    const auto & srcAddress = *MemCpyOperation::sourceInput(node).origin();
    const auto & dstAddress = *MemCpyOperation::destinationInput(node).origin();
    AddPointerTargetsToModRefSet(srcAddress, modRefSet);
    AddPointerTargetsToModRefSet(dstAddress, modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<FreeOperation>(&node))
  {
    util::HashSet<PointsToGraph::NodeIndex> modRefSet;
    const auto & freeAddress = *FreeOperation::addressInput(node).origin();
    AddPointerTargetsToModRefSet(freeAddress, modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<AllocaOperation>(&node))
  {
    const auto allocaMemoryNode = ModRefSummary_->GetPointsToGraph().getNodeForAlloca(node);
    ModRefSummary_->SetSimpleNodeModRef(node, { allocaMemoryNode });
  }
  else if (is<MallocOperation>(&node))
  {
    const auto mallocMemoryNode = ModRefSummary_->GetPointsToGraph().getNodeForMalloc(node);
    ModRefSummary_->SetSimpleNodeModRef(node, { mallocMemoryNode });
  }
  else if (is<CallOperation>(&node))
  {
    // CallOperations are omitted on purpose, as calls use the AllMemoryNodes as their ModRef set.
  }
  else if (is<MemoryStateOperation>(&node))
  {
    // Memory state operations are only used to route memory state edges
  }
  else
  {
    // Any remaining type of node should not involve any memory states
    JLM_ASSERT(!hasMemoryState(node));
  }
}

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph,
    util::StatisticsCollector & statisticsCollector)
{
  AgnosticModRefSummarizer summarizer;
  return summarizer.SummarizeModRefs(rvsdgModule, pointsToGraph, statisticsCollector);
}

std::unique_ptr<ModRefSummary>
AgnosticModRefSummarizer::Create(
    const rvsdg::RvsdgModule & rvsdgModule,
    const PointsToGraph & pointsToGraph)
{
  util::StatisticsCollector statisticsCollector;
  return Create(rvsdgModule, pointsToGraph, statisticsCollector);
}

}
