/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/theta-conv.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static void
theta_conv(jlm::rvsdg::theta_node * theta)
{
  jlm::rvsdg::substitution_map smap;

  auto loop = hls::loop_node::create(theta->region());
  std::vector<jlm::rvsdg::input *> branches;

  // add loopvars and populate the smap
  for (size_t i = 0; i < theta->ninputs(); i++)
  {
    jlm::rvsdg::output * buffer;
    loop->add_loopvar(theta->input(i)->origin(), &buffer);
    smap.insert(theta->input(i)->argument(), buffer);
    // buffer out is only used by branch
    branches.push_back(*buffer->begin());
    // divert theta outputs
    theta->output(i)->divert_users(loop->output(i));
  }

  // copy contents of theta
  theta->subregion()->copy(loop->subregion(), smap, false, false);

  // connect predicate/branches in loop to results
  loop->set_predicate(smap.lookup(theta->predicate()->origin()));
  for (size_t i = 0; i < theta->ninputs(); i++)
  {
    branches[i]->divert_to(smap.lookup(theta->input(i)->result()->origin()));
  }

  remove(theta);
}

static void
theta_conv(jlm::rvsdg::region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      if (auto theta = dynamic_cast<jlm::rvsdg::theta_node *>(node))
      {
        theta_conv(theta->subregion());
        theta_conv(theta);
        //                        theta_conv(region);
      }
      else
      {
        for (size_t n = 0; n < structnode->nsubregions(); n++)
          theta_conv(structnode->subregion(n));
      }
    }
  }
}

void
theta_conv(jlm::llvm::RvsdgModule & rm)
{
  auto & graph = rm.Rvsdg();
  auto root = graph.root();
  theta_conv(root);
}

}
