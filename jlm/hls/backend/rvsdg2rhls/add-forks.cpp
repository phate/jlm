/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-forks.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

void
add_forks(jlm::rvsdg::region * region)
{
  for (size_t i = 0; i < region->narguments(); ++i)
  {
    auto arg = region->argument(i);
    if (arg->nusers() > 1)
    {
      std::vector<jlm::rvsdg::input *> users;
      users.insert(users.begin(), arg->begin(), arg->end());
      auto fork = hls::fork_op::create(arg->nusers(), *arg);
      for (size_t j = 0; j < users.size(); j++)
      {
        users[j]->divert_to(fork[j]);
      }
    }
  }
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        add_forks(structnode->subregion(n));
      }
    }
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      auto out = node->output(i);
      if (out->nusers() > 1)
      {
        std::vector<jlm::rvsdg::input *> users;
        users.insert(users.begin(), out->begin(), out->end());
        auto fork = hls::fork_op::create(out->nusers(), *out);
        for (size_t j = 0; j < users.size(); j++)
        {
          users[j]->divert_to(fork[j]);
        }
      }
    }
  }
}

void
add_forks(llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  add_forks(root);
}

} // namespace hls::hls
