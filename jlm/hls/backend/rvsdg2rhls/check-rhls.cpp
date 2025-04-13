/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include "hls-function-util.hpp"
#include <jlm/hls/backend/rvsdg2rhls/check-rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{
void
CheckAddrQueue(rvsdg::Node * node)
{
  auto addrQ = TryGetOwnerOp<addr_queue_op>(*node->output(0));
  JLM_ASSERT(rvsdg::is<addr_queue_op>(node));
  // Ensure that there is no buffer between state_gate and addr_queue enq.
  // This is SG1 in the paper. Otherwise, there might be a race condition in the disambiguation
  JLM_ASSERT(TryGetOwnerOp<state_gate_op>(*FindSourceNode(node->input(1)->origin())));
  // make sure there is enough buffer space on the output, so there can be no race condition with SG3
  auto buf = TryGetOwnerOp<buffer_op>(**node->output(0)->begin());
  JLM_ASSERT(buf && buf->capacity >= addrQ->capacity);
}

void
check_rhls(rvsdg::Region * sr)
{
  for (auto & node : rvsdg::TopDownTraverser(sr))
  {
    if (rvsdg::is<rvsdg::StructuralOperation>(node))
    {
      if (auto ln = dynamic_cast<hls::loop_node *>(node))
      {
        check_rhls(ln->subregion());
      }
      else
      {
        throw jlm::util::error("There should be only simple nodes and loop nodes");
      }
    }
    for (size_t i = 0; i < node->noutputs(); i++)
    {
      if (node->output(i)->nusers() == 0)
      {
        throw jlm::util::error("Output has no users");
      }
      else if (node->output(i)->nusers() > 1)
      {
        throw jlm::util::error("Output has more than one user");
      }
    }
    if (rvsdg::is<addr_queue_op>(node))
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
    throw jlm::util::error("Root should have only one node now");
  }
  auto ln = dynamic_cast<const rvsdg::LambdaNode *>(root->Nodes().begin().ptr());
  if (!ln)
  {
    throw jlm::util::error("Node needs to be a lambda");
  }
  check_rhls(ln->subregion());
}

}
