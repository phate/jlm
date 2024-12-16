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

  // Add loop variables, insert loop constant buffers for invariant variables, and populate the
  // smap.
  for (size_t i = 0; i < theta.ninputs(); i++)
  {
    // Check if the input is a loop invariant such that a loop constant buffer should be created.
    // Memory state inputs are not loop variables containting a value, so we ignor these.
    if (is_invariant(theta.input(i))
        && !jlm::rvsdg::is<jlm::llvm::MemoryStateType>(theta.input(i)->Type()))
    {
      smap.insert(theta.input(i)->argument(), loop->add_loopconst(theta.input(i)->origin()));
      branches.push_back(nullptr);
      // The HLS loop has no output for this input. The users of the theta output is
      // therefore redirected to the input origin, as the value is loop invariant.
      theta.output(i)->divert_users(theta.input(i)->origin());
    }
    else
    {
      jlm::rvsdg::output * buffer;
      loop->add_loopvar(theta.input(i)->origin(), &buffer);
      smap.insert(theta.input(i)->argument(), buffer);
      // buffer out is only used by branch
      branches.push_back(*buffer->begin());
      // divert theta outputs
      theta.output(i)->divert_users(loop->output(loop->noutputs() - 1));
    }
  }

  // copy contents of theta
  theta.subregion()->copy(loop->subregion(), smap, false, false);

  // connect predicate/branches in loop to results
  loop->set_predicate(smap.lookup(theta.predicate()->origin()));
  for (size_t i = 0; i < theta.ninputs(); i++)
  {
    if (branches[i])
    {
      branches[i]->divert_to(smap.lookup(theta.input(i)->result()->origin()));
    }
  }

  remove(&theta);
}

static void
ConvertThetaNodesInRegion(rvsdg::Region & region);

static void
ConvertThetaNodesInStructuralNode(rvsdg::StructuralNode & structuralNode)
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
    if (auto structuralNode = dynamic_cast<rvsdg::StructuralNode *>(node))
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
