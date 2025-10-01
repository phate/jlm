/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>

namespace jlm::llvm::aa
{

/** \brief Mod/Ref summary of agnostic mod/ref summarizer
 *
 */
class AgnosticModRefSummary final : public ModRefSummary
{
public:
  using SimpleNodeModRefMap = std::
      unordered_map<const rvsdg::SimpleNode *, util::HashSet<const PointsToGraph::MemoryNode *>>;

  ~AgnosticModRefSummary() noexcept override = default;

private:
  AgnosticModRefSummary(
      const PointsToGraph & pointsToGraph,
      util::HashSet<const PointsToGraph::MemoryNode *> allMemoryNodes)
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
      util::HashSet<const PointsToGraph::MemoryNode *> modRefSet)
  {
    JLM_ASSERT(SimpleNodeModRefs_.find(&node) == SimpleNodeModRefs_.end());
    SimpleNodeModRefs_[&node] = std::move(modRefSet);
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
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
    JLM_UNREACHABLE("Unhandled node type.");
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaEntryModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetGammaExitModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetThetaModRef([[maybe_unused]] const rvsdg::ThetaNode & theta) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaEntryModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const util::HashSet<const PointsToGraph::MemoryNode *> &
  GetLambdaExitModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    return AllMemoryNodes_;
  }

  static std::unique_ptr<AgnosticModRefSummary>
  Create(
      const PointsToGraph & pointsToGraph,
      util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes)
  {
    return std::unique_ptr<AgnosticModRefSummary>(
        new AgnosticModRefSummary(pointsToGraph, std::move(memoryNodes)));
  }

private:
  const PointsToGraph & PointsToGraph_;
  SimpleNodeModRefMap SimpleNodeModRefs_;
  util::HashSet<const PointsToGraph::MemoryNode *> AllMemoryNodes_;
};

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

util::HashSet<const PointsToGraph::MemoryNode *>
AgnosticModRefSummarizer::GetAllMemoryNodes(const PointsToGraph & pointsToGraph)
{
  util::HashSet<const PointsToGraph::MemoryNode *> memoryNodes;
  for (auto & allocaNode : pointsToGraph.AllocaNodes())
    memoryNodes.Insert(&allocaNode);

  for (auto & deltaNode : pointsToGraph.DeltaNodes())
    memoryNodes.Insert(&deltaNode);

  for (auto & lambdaNode : pointsToGraph.LambdaNodes())
    memoryNodes.Insert(&lambdaNode);

  for (auto & mallocNode : pointsToGraph.MallocNodes())
    memoryNodes.Insert(&mallocNode);

  for (auto & importNode : pointsToGraph.ImportNodes())
    memoryNodes.Insert(&importNode);

  memoryNodes.Insert(&pointsToGraph.GetExternalMemoryNode());

  return memoryNodes;
}

void
AgnosticModRefSummarizer::AnnotateRegion(const rvsdg::Region & region)
{
  for (const auto & node : region.Nodes())
  {
    if (const auto simpleNode = dynamic_cast<const rvsdg::SimpleNode *>(&node))
    {
      AnnotateSimpleNode(*simpleNode);
    }
    else if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      AnnotateStructuralNode(*structuralNode);
    }
    else
    {
      JLM_UNREACHABLE("Unknown node type");
    }
  }
}

void
AgnosticModRefSummarizer::AddPointerToModRefSet(
    const rvsdg::Output & output,
    util::HashSet<const PointsToGraph::MemoryNode *> & modRefSet)
{
  JLM_ASSERT(IsPointerCompatible(output));
  const auto & addressReg = ModRefSummary_->GetPointsToGraph().GetRegisterNode(output);
  for (auto & target : addressReg.Targets())
  {
    modRefSet.Insert(&target);
  }
}

void
AgnosticModRefSummarizer::AnnotateSimpleNode(const rvsdg::SimpleNode & node)
{
  if (is<StoreOperation>(&node))
  {
    const auto & address = *StoreOperation::AddressInput(node).origin();
    util::HashSet<const PointsToGraph::MemoryNode *> modRefSet;
    AddPointerToModRefSet(address, modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<LoadOperation>(&node))
  {
    const auto & address = *LoadOperation::AddressInput(node).origin();
    util::HashSet<const PointsToGraph::MemoryNode *> modRefSet;
    AddPointerToModRefSet(address, modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<MemCpyOperation>(&node))
  {
    util::HashSet<const PointsToGraph::MemoryNode *> modRefSet;
    AddPointerToModRefSet(*node.input(0)->origin(), modRefSet);
    AddPointerToModRefSet(*node.input(1)->origin(), modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<FreeOperation>(&node))
  {
    util::HashSet<const PointsToGraph::MemoryNode *> modRefSet;
    AddPointerToModRefSet(*node.input(0)->origin(), modRefSet);
    ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
  }
  else if (is<AllocaOperation>(&node))
  {
    const auto & allocaMemoryNode = ModRefSummary_->GetPointsToGraph().GetAllocaNode(node);
    ModRefSummary_->SetSimpleNodeModRef(node, { &allocaMemoryNode });
  }
  else if (is<MallocOperation>(&node))
  {
    const auto & mallocMemoryNode = ModRefSummary_->GetPointsToGraph().GetMallocNode(node);
    ModRefSummary_->SetSimpleNodeModRef(node, { &mallocMemoryNode });
  }

  // CallOperations are omitted on purpose, as calls use the AllMemoryNodes as their ModRefSet.
}

void
AgnosticModRefSummarizer::AnnotateStructuralNode(const rvsdg::StructuralNode & node)
{
  for (const auto & subregion : node.Subregions())
  {
    AnnotateRegion(subregion);
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
