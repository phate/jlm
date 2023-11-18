/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-sinks.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
add_sinks(jlm::rvsdg::region * region)
{
  for (size_t i = 0; i < region->narguments(); ++i)
  {
    auto arg = region->argument(i);
    if (!arg->nusers())
    {
      hls::sink_op::create(*arg);
    }
  }
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        add_sinks(structnode->subregion(n));
      }
    }

    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      auto out = node->output(i);
      if (!out->nusers())
      {
        hls::sink_op::create(*out);
      }
    }
  }
}

void
add_sinks(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  add_sinks(root);
}

}
