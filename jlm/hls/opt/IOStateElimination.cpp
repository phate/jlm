/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/opt/IOStateElimination.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/types.hpp>

namespace jlm::hls
{

void
IOStateElimination::eliminateIOStates(rvsdg::Region & region, rvsdg::Output & ioStateArgument)
{
  // FIXME: This method routes a lot of superfluous IO states into regions. It is unnecessary work,
  // even though DNE + CNE eventually removes them again.

  for (auto & node : region.Nodes())
  {
    // Handle innermost regions first
    if (const auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(&node))
    {
      for (auto & subregion : structuralNode->Subregions())
        eliminateIOStates(subregion, ioStateArgument);
    }

    // Ensure all IO state outputs become dead
    for (auto & output : node.Outputs())
    {
      if (rvsdg::is<llvm::IOStateType>(output.Type()))
      {
        auto & ioStateOperand = rvsdg::RouteToRegion(ioStateArgument, region);
        output.divert_users(&ioStateOperand);
      }
    }
  }
}

IOStateElimination::~IOStateElimination() noexcept = default;

void
IOStateElimination::Run(rvsdg::RvsdgModule & module, util::StatisticsCollector &)
{
  const auto & graph = module.Rvsdg();
  const auto rootRegion = &graph.GetRootRegion();
  if (rootRegion->numNodes() != 1)
  {
    throw std::logic_error("Root should have only one node now");
  }

  const auto lambdaNode =
      dynamic_cast<const rvsdg::LambdaNode *>(rootRegion->Nodes().begin().ptr());
  if (!lambdaNode)
  {
    throw std::logic_error("Node needs to be a lambda");
  }

  eliminateIOStates(
      *lambdaNode->subregion(),
      llvm::LlvmLambdaOperation::getIOStateArgument(*lambdaNode));
}

}
