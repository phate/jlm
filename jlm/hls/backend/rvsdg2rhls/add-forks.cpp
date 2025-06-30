/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-forks.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
ForkInsertion::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  AddForksToRegion(module.Rvsdg().GetRootRegion());
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
  JLM_ASSERT(output.nusers() > 1 && output.nusers() != 0);

  const auto isConstant = IsConstantFork(output);
  const auto & forkNode = ForkOperation::CreateNode(output.nusers(), output, isConstant);

  size_t currentForkOutput = 0;
  for (auto & user : output.Users())
  {
    if (&user == forkNode.input(0))
    {
      // Ignore the just added fork node
      continue;
    }

    JLM_ASSERT(currentForkOutput < forkNode.noutputs());
    user.divert_to(forkNode.output(currentForkOutput));
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

void
add_forks(rvsdg::Region * region)
{
  for (size_t i = 0; i < region->narguments(); ++i)
  {
    auto arg = region->argument(i);
    if (arg->nusers() > 1)
    {
      std::vector<jlm::rvsdg::Input *> users;
      users.insert(users.begin(), arg->begin(), arg->end());
      auto fork = ForkOperation::create(arg->nusers(), *arg);
      for (size_t j = 0; j < users.size(); j++)
      {
        users[j]->divert_to(fork[j]);
      }
    }
  }
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        add_forks(structnode->subregion(n));
      }
    }
    // If a node has no inputs it is a constant
    bool isConstant = node->ninputs() == 0;
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      auto out = node->output(i);
      if (out->nusers() > 1)
      {
        std::vector<rvsdg::Input *> users(out->begin(), out->end());
        auto fork = ForkOperation::create(out->nusers(), *out, isConstant);
        for (size_t j = 0; j < users.size(); j++)
        {
          users[j]->divert_to(fork[j]);
        }
      }
    }
  }
}

void
add_forks(llvm::RvsdgModule & rvsdgModule)
{
  auto & graph = rvsdgModule.Rvsdg();
  auto root = &graph.GetRootRegion();
  add_forks(root);
}

}
