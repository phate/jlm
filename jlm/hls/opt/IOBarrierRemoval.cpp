/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/opt/IOBarrierRemoval.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/rvsdg/structural-node.hpp>

namespace jlm::hls
{

IOBarrierRemoval::~IOBarrierRemoval() noexcept = default;

void
IOBarrierRemoval::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  RemoveIOBarrierFromRegion(module.Rvsdg().GetRootRegion());
}

void
IOBarrierRemoval::RemoveIOBarrierFromRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    // Handle subregions first
    if (const auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        RemoveIOBarrierFromRegion(*structuralNode->subregion(n));
      }
    }

    // Render all IOBarrier nodes dead
    if (rvsdg::is<llvm::IOBarrierOperation>(&node))
    {
      node.output(0)->divert_users(node.input(0)->origin());
    }
  }

  // Remove all dead IOBarrier nodes
  region.prune(false);
}

}
