/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/remove-redundant-buf.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

bool
eliminate_buf(jlm::rvsdg::Output * o)
{
  if (auto so = dynamic_cast<rvsdg::SimpleOutput *>(o))
  {
    auto node = so->node();
    if (jlm::rvsdg::is<const BranchOperation>(node->GetOperation()))
    {
      return eliminate_buf(node->input(1)->origin());
    }
    else if (jlm::rvsdg::is<const ForkOperation>(node->GetOperation()))
    {
      // part of memory disambiguation
      return eliminate_buf(node->input(0)->origin());
    }
    else if (jlm::rvsdg::is<const local_load_op>(node->GetOperation()))
    {
      return true;
    }
    if (jlm::rvsdg::is<LocalStoreOperation>(node))
    {
      return true;
    }
    else if (jlm::rvsdg::is<const LoadOperation>(node->GetOperation()))
    {
      return true;
    }
    else if (jlm::rvsdg::is<const jlm::hls::store_op>(node->GetOperation()))
    {
      return true;
    }
  }

  return false;
}

void
remove_redundant_buf(rvsdg::Region * region)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        remove_redundant_buf(structnode->subregion(n));
      }
    }
    else if (dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (auto buf = dynamic_cast<const BufferOperation *>(&node->GetOperation()))
      {
        if (std::dynamic_pointer_cast<const jlm::llvm::MemoryStateType>(buf->argument(0)))
        {
          if (!buf->pass_through && eliminate_buf(node->input(0)->origin()))
          {
            auto new_out =
                BufferOperation::create(*node->input(0)->origin(), buf->capacity, true)[0];
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
  auto root = &graph.GetRootRegion();
  remove_redundant_buf(root);
}

} // namespace jlm::hls
