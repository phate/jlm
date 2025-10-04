/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-sinks.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>

namespace jlm::hls
{

SinkInsertion::~SinkInsertion() noexcept = default;

void
SinkInsertion::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  HandleRootRegion(module.Rvsdg().GetRootRegion());
}

void
SinkInsertion::CreateAndRun(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  SinkInsertion sinkInsertion;
  sinkInsertion.Run(module, statisticsCollector);
}

void
SinkInsertion::HandleRootRegion(rvsdg::Region & region)
{
  for (auto & node : region.Nodes())
  {
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        AddSinksToRegion(subregion);
      }
    }
  }
}

void
SinkInsertion::AddSinksToRegion(rvsdg::Region & region)
{
  // Add sinks to region arguments
  for (const auto argument : region.Arguments())
  {
    if (argument->IsDead())
    {
      SinkOperation::create(*argument);
    }
  }

  for (auto & node : region.Nodes())
  {
    // Add sinks to subregions of structural nodes
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        AddSinksToRegion(subregion);
      }
    }

    // Add sinks to outputs of nodes
    for (size_t n = 0; n < node.noutputs(); n++)
    {
      const auto output = node.output(n);
      if (output->IsDead())
      {
        SinkOperation::create(*output);
      }
    }
  }
}

}
