/*
 * Copyright 2022 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "jlm/util/common.hpp"
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/StdLibIntrinsicOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/opt/alias-analyses/AgnosticModRefSummarizer.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/llvm/opt/alias-analyses/ModRefSummary.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/MatchType.hpp>

#include <unordered_map>

namespace jlm::llvm::aa
{

/** \brief class returned by the Agnostic ModRefSummarizer
 *
 */
class AgnosticModRefSet final : public ModRefSet
{
public:
  void
  addMemoryNode(PointsToGraph::NodeIndex memoryNode, ModRefEffect modRefEffect)
  {
    JLM_ASSERT(modRefEffect != ModRefEffect::NoEffect);
    modRefNodes_[memoryNode] |= modRefEffect;
  }
};

/** \brief Mod/Ref summary of agnostic mod/ref summarizer
 *
 */
class AgnosticModRefSummary final : public ModRefSummary
{
public:
  using SimpleNodeModRefMap = std::unordered_map<const rvsdg::SimpleNode *, AgnosticModRefSet>;

  ~AgnosticModRefSummary() noexcept override = default;

private:
  AgnosticModRefSummary(const PointsToGraph & pointsToGraph, AgnosticModRefSet allMemoryNodes)
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
  SetSimpleNodeModRef(const rvsdg::SimpleNode & node, AgnosticModRefSet modRefSet)
  {
    JLM_ASSERT(SimpleNodeModRefs_.find(&node) == SimpleNodeModRefs_.end());
    SimpleNodeModRefs_.insert({ &node, std::move(modRefSet) });
  }

  [[nodiscard]] const ModRefSet &
  GetSimpleNodeModRef(const rvsdg::SimpleNode & node) const override
  {
    if (const auto it = SimpleNodeModRefs_.find(&node); it != SimpleNodeModRefs_.end())
    {
      return it->second;
    }
    if (is<CallOperation>(node.GetOperation()))
    {
      return AllMemoryNodes_;
    }
    throw std::logic_error("Unhandled node type.");
  }

  [[nodiscard]] const ModRefSet &
  GetGammaEntryModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const ModRefSet &
  GetGammaExitModRef([[maybe_unused]] const rvsdg::GammaNode & gamma) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const ModRefSet &
  GetThetaModRef([[maybe_unused]] const rvsdg::ThetaNode & theta) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const ModRefSet &
  GetLambdaEntryModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    return AllMemoryNodes_;
  }

  [[nodiscard]] const ModRefSet &
  GetLambdaExitModRef([[maybe_unused]] const rvsdg::LambdaNode & lambda) const override
  {
    return AllMemoryNodes_;
  }

  static std::unique_ptr<AgnosticModRefSummary>
  Create(const PointsToGraph & pointsToGraph, AgnosticModRefSet allMemoryNodes)
  {
    return std::unique_ptr<AgnosticModRefSummary>(
        new AgnosticModRefSummary(pointsToGraph, std::move(allMemoryNodes)));
  }

private:
  const PointsToGraph & PointsToGraph_;
  SimpleNodeModRefMap SimpleNodeModRefs_;
  AgnosticModRefSet AllMemoryNodes_;
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

AgnosticModRefSet
AgnosticModRefSummarizer::GetAllMemoryNodes(const PointsToGraph & pointsToGraph)
{
  AgnosticModRefSet modRefSet;
  for (const auto allocaNode : pointsToGraph.allocaNodes())
    modRefSet.addMemoryNode(allocaNode, ModRefEffect::ModRef);

  for (const auto deltaNode : pointsToGraph.deltaNodes())
    modRefSet.addMemoryNode(deltaNode, ModRefEffect::ModRef);

  for (const auto lambdaNode : pointsToGraph.lambdaNodes())
    modRefSet.addMemoryNode(lambdaNode, ModRefEffect::ModRef);

  for (const auto mallocNode : pointsToGraph.mallocNodes())
    modRefSet.addMemoryNode(mallocNode, ModRefEffect::ModRef);

  for (const auto importNode : pointsToGraph.importNodes())
    modRefSet.addMemoryNode(importNode, ModRefEffect::ModRef);

  modRefSet.addMemoryNode(pointsToGraph.getExternalMemoryNode(), ModRefEffect::ModRef);

  JLM_ASSERT(modRefSet.getModRefNodes().size() == pointsToGraph.numMemoryNodes());

  return modRefSet;
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
    ModRefEffect modRefEffect,
    AgnosticModRefSet & modRefSet) const
{
  const auto & pointsToGraph = ModRefSummary_->GetPointsToGraph();
  JLM_ASSERT(IsPointerCompatible(output));
  const auto & addressReg = pointsToGraph.getNodeForRegister(output);
  for (const auto target : pointsToGraph.getExplicitTargets(addressReg).Items())
  {
    modRefSet.addMemoryNode(target, modRefEffect);
  }
  if (pointsToGraph.isTargetingAllExternallyAvailable(addressReg))
  {
    // Add all externally available memory nodes
    for (const auto implicitTarget : pointsToGraph.getExternallyAvailableNodes())
    {
      modRefSet.addMemoryNode(implicitTarget, modRefEffect);
    }
  }
}

void
AgnosticModRefSummarizer::AnnotateSimpleNode(const rvsdg::SimpleNode & node)
{
  MatchTypeWithDefault(
      node.GetOperation(),
      [&](const StoreOperation &)
      {
        const auto & address = *StoreOperation::AddressInput(node).origin();
        AgnosticModRefSet modRefSet;
        AddPointerTargetsToModRefSet(address, ModRefEffect::ModOnly, modRefSet);
        ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
      },
      [&](const LoadOperation &)
      {
        const auto & address = *LoadOperation::AddressInput(node).origin();
        AgnosticModRefSet modRefSet;
        AddPointerTargetsToModRefSet(address, ModRefEffect::RefOnly, modRefSet);
        ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
      },
      [&](const MemCpyOperation &)
      {
        AgnosticModRefSet modRefSet;
        const auto & srcAddress = *MemCpyOperation::sourceInput(node).origin();
        const auto & dstAddress = *MemCpyOperation::destinationInput(node).origin();
        AddPointerTargetsToModRefSet(srcAddress, ModRefEffect::RefOnly, modRefSet);
        AddPointerTargetsToModRefSet(dstAddress, ModRefEffect::ModOnly, modRefSet);
        ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
      },
      [&](const MemSetOperation &)
      {
        AgnosticModRefSet modRefSet;
        const auto & dstAddress = *MemSetOperation::destinationInput(node).origin();
        AddPointerTargetsToModRefSet(dstAddress, ModRefEffect::ModOnly, modRefSet);
        ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
      },
      [&](const FreeOperation &)
      {
        AgnosticModRefSet modRefSet;
        const auto & freeAddress = *FreeOperation::addressInput(node).origin();
        AddPointerTargetsToModRefSet(freeAddress, ModRefEffect::ModOnly, modRefSet);
        ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
      },
      [&](const AllocaOperation &)
      {
        const auto allocaMemoryNode = ModRefSummary_->GetPointsToGraph().getNodeForAlloca(node);
        AgnosticModRefSet modRefSet;
        // The alloca operation does not assign to the allocated memory, so Ref is more precise
        modRefSet.addMemoryNode(allocaMemoryNode, ModRefEffect::RefOnly);
        ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
      },
      [&](const MallocOperation &)
      {
        const auto mallocMemoryNode = ModRefSummary_->GetPointsToGraph().getNodeForMalloc(node);
        AgnosticModRefSet modRefSet;
        // The malloc operation does not assign to the allocated memory, so Ref is more precise
        modRefSet.addMemoryNode(mallocMemoryNode, ModRefEffect::RefOnly);
        ModRefSummary_->SetSimpleNodeModRef(node, std::move(modRefSet));
      },
      [&](const CallOperation &)
      {
        // CallOperations are omitted on purpose, as calls use the AllMemoryNodes as their ModRef
        // set.
      },
      [&](const MemoryStateOperation &)
      {
        // Memory state operations are only used to route memory state edges
      },
      [&]()
      {
        // Any remaining type of node should not involve any memory states
        JLM_ASSERT(!hasMemoryState(node));
      });
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
