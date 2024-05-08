/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/remove-redundant-buf.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/opt/alias-analyses/Operators.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>

namespace jlm::hls
{

bool
eliminate_buf(jlm::rvsdg::output * o)
{
  if (auto so = dynamic_cast<jlm::rvsdg::simple_output *>(o))
  {
    auto node = so->node();
    if (dynamic_cast<const branch_op *>(&node->operation()))
    {
      return eliminate_buf(node->input(1)->origin());
    }
    else if (dynamic_cast<const local_load_op *>(&node->operation()))
    {
      return true;
    }
    else if (dynamic_cast<const local_store_op *>(&node->operation()))
    {
      return true;
    }
  }
  return false;
}

void
remove_redundant_buf(jlm::rvsdg::region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        remove_redundant_buf(structnode->subregion(n));
      }
    }
    else if (dynamic_cast<jlm::rvsdg::simple_node *>(node))
    {
      if (auto buf = dynamic_cast<const buffer_op *>(&node->operation()))
      {
        if (dynamic_cast<const jlm::llvm::MemoryStateType *>(&buf->argument(0).type()))
        {
          if (!buf->pass_through && eliminate_buf(node->input(0)->origin()))
          {
            auto new_out = buffer_op::create(*node->input(0)->origin(), buf->capacity, true)[0];
            node->output(0)->divert_users(new_out);
            remove(node);
          }
        }
      }
    }
  }
}

void
remove_redundant_buf(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  remove_redundant_buf(root);
}

} // namespace jlm::hls
