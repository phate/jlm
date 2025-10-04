/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-forks.hpp>
#include <jlm/hls/ir/hls.hpp>

namespace jlm::hls
{

ForkInsertion::~ForkInsertion() noexcept = default;

void
ForkInsertion::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  AddForksToRegion(module.Rvsdg().GetRootRegion());
}

void
ForkInsertion::CreateAndRun(
    rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  ForkInsertion forkInsertion;
  forkInsertion.Run(module, statisticsCollector);
}

void
ForkInsertion::AddForksToRegion(rvsdg::Region & region)
{
  // Add forks to region arguments
  for (const auto argument : region.Arguments())
  {
    if (argument->nusers() > 1)
      AddForkToOutput(*argument);
  }

  for (auto & node : region.Nodes())
  {
    // Add forks to subregions of structural nodes
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
      {
        AddForksToRegion(subregion);
      }
    }

    // Add forks to outputs of nodes
    for (size_t n = 0; n < node.noutputs(); n++)
    {
      const auto output = node.output(n);
      if (output->nusers() > 1)
      {
        AddForkToOutput(*output);
      }
    }
  }
}

void
ForkInsertion::AddForkToOutput(rvsdg::Output & output)
{
  JLM_ASSERT(output.nusers() > 1);

  const auto isConstant = IsConstantFork(output);
  const auto & forkNode = ForkOperation::CreateNode(output.nusers(), output, isConstant);

  size_t currentForkOutput = 0;
  while (output.nusers() != 1) // The fork node should be the only user left in the end
  {
    auto userIt = output.Users().begin();
    if (&*userIt == forkNode.input(0))
    {
      // Ignore the added fork node
      userIt = std::next(userIt);
    }

    JLM_ASSERT(currentForkOutput < forkNode.noutputs());
    userIt->divert_to(forkNode.output(currentForkOutput));
    currentForkOutput++;
  }

  JLM_ASSERT(currentForkOutput == forkNode.noutputs());
}

bool
ForkInsertion::IsConstantFork(const rvsdg::Output & output)
{
  const auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(output);
  return node != nullptr ? node->ninputs() == 0 : false;
}

}
