/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/check-rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

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
  auto ln = dynamic_cast<const llvm::lambda::node *>(root->Nodes().begin().ptr());
  if (!ln)
  {
    throw jlm::util::error("Node needs to be a lambda");
  }
  check_rhls(ln->subregion());
}

}
