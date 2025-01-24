/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/memstate-conv.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
memstate_conv(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  memstate_conv(root);
}

void
memstate_conv(rvsdg::Region * region)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        memstate_conv(structnode->subregion(n));
    }
    else if (auto simplenode = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (dynamic_cast<const llvm::LambdaEntryMemoryStateSplitOperation *>(
              &simplenode->GetOperation())
          || dynamic_cast<const jlm::llvm::MemoryStateSplitOperation *>(
              &simplenode->GetOperation()))
      {
        auto new_outs =
            hls::fork_op::create(simplenode->noutputs(), *simplenode->input(0)->origin());
        for (size_t i = 0; i < simplenode->noutputs(); ++i)
        {
          simplenode->output(i)->divert_users(new_outs[i]);
        }
        remove(simplenode);
      }
      // exit is handled as normal SimpleOperation
    }
  }
}

} // namespace jlm::hls
