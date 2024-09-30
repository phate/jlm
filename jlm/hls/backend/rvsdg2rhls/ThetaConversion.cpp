/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static void
ConvertThetaNode(rvsdg::ThetaNode & theta)
{
  rvsdg::SubstitutionMap smap;

  auto loop = hls::loop_node::create(theta.region());
  std::vector<jlm::rvsdg::input *> branches;

  // add loopvars and populate the smap
  for (size_t i = 0; i < theta.ninputs(); i++)
  {
    jlm::rvsdg::output * buffer;
    loop->add_loopvar(theta.input(i)->origin(), &buffer);
    smap.insert(theta.MapEntryLoopVar(*theta.input(i)).pre, buffer);
    // buffer out is only used by branch
    branches.push_back(*buffer->begin());
    // divert theta outputs
    theta.output(i)->divert_users(loop->output(i));
  }

  // copy contents of theta
  theta.subregion()->copy(loop->subregion(), smap, false, false);

  // connect predicate/branches in loop to results
  loop->set_predicate(smap.lookup(theta.predicate()->origin()));
  for (size_t i = 0; i < theta.ninputs(); i++)
  {
    branches[i]->divert_to(smap.lookup(theta.MapEntryLoopVar(*theta.input(i)).post->origin()));
  }

  remove(&theta);
}

static void
ConvertThetaNodesInRegion(rvsdg::Region & region);

static void
ConvertThetaNodesInStructuralNode(jlm::rvsdg::structural_node & structuralNode)
{
  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    ConvertThetaNodesInRegion(*structuralNode.subregion(n));
  }

  if (auto thetaNode = dynamic_cast<rvsdg::ThetaNode *>(&structuralNode))
  {
    ConvertThetaNode(*thetaNode);
  }
}

static void
ConvertThetaNodesInRegion(rvsdg::Region & region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(&region))
  {
    if (auto structuralNode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      ConvertThetaNodesInStructuralNode(*structuralNode);
    }
  }
}

void
ConvertThetaNodes(jlm::llvm::RvsdgModule & rvsdgModule)
{
  ConvertThetaNodesInRegion(*rvsdgModule.Rvsdg().root());
}

}
