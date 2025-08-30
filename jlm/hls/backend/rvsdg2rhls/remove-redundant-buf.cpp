/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/remove-redundant-buf.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

RedundantBufferElimination::~RedundantBufferElimination() noexcept = default;

void
RedundantBufferElimination::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  HandleRegion(module.Rvsdg().GetRootRegion());
}

void
RedundantBufferElimination::CreateAndRun(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  RedundantBufferElimination redundantBufferElimination;
  redundantBufferElimination.Run(module, statisticsCollector);
}

void
RedundantBufferElimination::HandleRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    // Handle innermost regions first
    if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        HandleRegion(subregion);
      }
      continue;
    }

    auto bufferOperation = dynamic_cast<const BufferOperation *>(&node.GetOperation());
    if (!bufferOperation)
      continue;

    if (!rvsdg::is<llvm::MemoryStateType>(node.input(0)->Type()))
      continue;

    if (bufferOperation->IsPassThrough())
      continue;

    if (!CanTraceToLoadOrStore(*node.input(0)->origin()))
      continue;

    // Replace the BufferOperation node with a passthrough BufferOperation node
    auto result =
        BufferOperation::create(*node.input(0)->origin(), bufferOperation->Capacity(), true)[0];
    node.output(0)->divert_users(result);
  }

  // Prune dead nodes
  region.prune(false);
}

bool
RedundantBufferElimination::CanTraceToLoadOrStore(const rvsdg::Output & output)
{
  JLM_ASSERT(rvsdg::is<llvm::MemoryStateType>(output.Type()));

  if (rvsdg::IsOwnerNodeOperation<LoadOperation>(output))
    return true;

  if (rvsdg::IsOwnerNodeOperation<LocalLoadOperation>(output))
    return true;

  if (rvsdg::IsOwnerNodeOperation<StoreOperation>(output))
    return true;

  if (rvsdg::IsOwnerNodeOperation<LocalStoreOperation>(output))
    return true;

  auto [forkNode, forkOperation] = rvsdg::TryGetSimpleNodeAndOptionalOp<ForkOperation>(output);
  if (forkNode && forkOperation)
    return CanTraceToLoadOrStore(*forkNode->input(0)->origin());

  auto [branchNode, branchOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BranchOperation>(output);
  if (branchNode && branchOperation)
    return CanTraceToLoadOrStore(*branchNode->input(1)->origin());

  return false;
}

} // namespace jlm::hls
