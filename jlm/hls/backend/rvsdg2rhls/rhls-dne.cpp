/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static bool
remove_unused_loop_outputs(hls::loop_node & loopNode)
{
  auto numRemovedOutputs = loopNode.PruneLoopOutputs();
  return numRemovedOutputs != 0;
}

static bool
remove_unused_loop_inputs(hls::loop_node & loopNode)
{
  size_t numRemovedInputs = loopNode.PruneLoopInputs();

  auto match = [](const backedge_argument & argument)
  {
    auto & result = *argument.result();
    auto isPassthrough = argument.nusers() == 1 && result.origin() == &argument;

    return argument.IsDead() || isPassthrough;
  };
  loopNode.RemoveBackEdgeArgumentsWhere(match);

  return numRemovedInputs != 0;
}

bool
dne(jlm::rvsdg::region * sr)
{
  bool any_changed = false;
  bool changed;
  do
  {
    changed = false;
    for (auto & node : jlm::rvsdg::bottomup_traverser(sr))
    {
      if (!node->has_users())
      {
        remove(node);
        changed = true;
      }
      else if (auto ln = dynamic_cast<hls::loop_node *>(node))
      {
        changed |= remove_unused_loop_outputs(*ln);
        changed |= remove_unused_loop_inputs(*ln);
        changed |= dne(ln->subregion());
      }
    }
    any_changed |= changed;
  } while (changed);
  assert(sr->bottom_nodes.empty());
  return any_changed;
}

void
dne(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  if (root->nodes.size() != 1)
  {
    throw util::error("Root should have only one node now");
  }
  auto ln = dynamic_cast<const llvm::lambda::node *>(root->nodes.begin().ptr());
  if (!ln)
  {
    throw util::error("Node needs to be a lambda");
  }
  dne(ln->subregion());
}

}
