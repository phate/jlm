/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/memstate-conv.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>

namespace jlm::hls
{

MemoryStateSplitConversion::~MemoryStateSplitConversion() noexcept = default;

void
MemoryStateSplitConversion::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  ConvertMemoryStateSplitsInRegion(module.Rvsdg().GetRootRegion());
}

void
MemoryStateSplitConversion::CreateAndRun(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  MemoryStateSplitConversion memoryStateSplitConversion;
  memoryStateSplitConversion.Run(module, statisticsCollector);
}

void
MemoryStateSplitConversion::ConvertMemoryStateSplitsInRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    // Handle innermost regions first
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        ConvertMemoryStateSplitsInRegion(subregion);
      }
    }

    // Replace split nodes with fork nodes
    if (rvsdg::is<llvm::LambdaEntryMemoryStateSplitOperation>(&node)
        || rvsdg::is<llvm::MemoryStateSplitOperation>(&node))
    {
      JLM_ASSERT(node.ninputs() == 1);
      auto results = ForkOperation::create(node.noutputs(), *node.input(0)->origin());
      divert_users(&node, results);
    }
  }

  // Prune dead nodes
  region.prune(false);
}

} // namespace jlm::hls
