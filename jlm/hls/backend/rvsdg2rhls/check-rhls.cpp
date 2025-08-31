/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/check-rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/hls-function-util.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{
void
CheckAddrQueue(rvsdg::Node * node)
{
  auto [addrQueueNode, addrQueueOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<AddressQueueOperation>(*node->output(0));
  JLM_ASSERT(rvsdg::is<AddressQueueOperation>(node));
  // Ensure that there is no buffer between state_gate and addr_queue enq.
  // This is SG1 in the paper. Otherwise, there might be a race condition in the disambiguation
  JLM_ASSERT(
      rvsdg::IsOwnerNodeOperation<StateGateOperation>(*FindSourceNode(node->input(1)->origin())));
  // make sure there is enough buffer space on the output, so there can be no race condition with
  // SG3
  auto [bufferNode, bufferOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BufferOperation>(*node->output(0)->Users().begin());
  JLM_ASSERT(bufferOperation && bufferOperation->Capacity() >= addrQueueOperation->capacity);
}

void
check_rhls(rvsdg::Region * sr)
{
  for (auto & node : rvsdg::TopDownTraverser(sr))
  {
    if (rvsdg::is<rvsdg::StructuralOperation>(node))
    {
      if (auto ln = dynamic_cast<LoopNode *>(node))
      {
        check_rhls(ln->subregion());
      }
      else
      {
        throw util::Error("There should be only simple nodes and loop nodes");
      }
    }
    for (size_t i = 0; i < node->noutputs(); i++)
    {
      if (node->output(i)->nusers() == 0)
      {
        throw util::Error("Output has no users");
      }
      else if (node->output(i)->nusers() > 1)
      {
        throw util::Error("Output has more than one user");
      }
    }
    if (rvsdg::is<AddressQueueOperation>(node))
    {
      CheckAddrQueue(node);
    }
  }
}

void
check_rhls(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  if (root->nnodes() != 1)
  {
    throw util::Error("Root should have only one node now");
  }
  auto ln = dynamic_cast<const rvsdg::LambdaNode *>(root->Nodes().begin().ptr());
  if (!ln)
  {
    throw util::Error("Node needs to be a lambda");
  }
  check_rhls(ln->subregion());
}

}
