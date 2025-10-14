/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/opt/IOStateElimination.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static rvsdg::RegionArgument *
GetIoStateArgument(const rvsdg::LambdaNode & lambda)
{
  auto subregion = lambda.subregion();
  for (size_t n = 0; n < subregion->narguments(); n++)
  {
    auto argument = subregion->argument(n);
    if (jlm::rvsdg::is<jlm::llvm::IOStateType>(argument->Type()))
      return argument;
  }
  return nullptr;
}

static void
eliminate_io_state(rvsdg::RegionArgument * iostate, rvsdg::Region * region)
{
  // eliminates iostate fromm all calls, as well as removes iostate from node outputs
  // this leaves a pseudo-dependecy routed to the respective argument
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        eliminate_io_state(iostate, structnode->subregion(n));
    }
    else if (auto simplenode = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (dynamic_cast<const llvm::CallOperation *>(&simplenode->GetOperation()))
      {
        auto io_routed = &rvsdg::RouteToRegion(*iostate, *region);
        auto io_in = node->input(node->ninputs() - 2);
        io_in->divert_to(io_routed);
      }
    }
    // make sure iostate outputs are not used to break dependencies
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      auto out = node->output(i);
      if (!jlm::rvsdg::is<jlm::llvm::IOStateType>(out->Type()))
        continue;
      auto routed = &rvsdg::RouteToRegion(*iostate, *region);
      out->divert_users(routed);
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

  eliminate_io_state(GetIoStateArgument(*lambdaNode), lambdaNode->subregion());
}

}
