/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/static/ThetaConversion.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <jlm/hls/ir/static/loop.hpp>

namespace jlm::static_hls
{

void
addSimpleNodes(jlm::rvsdg::region & region, jlm::static_hls::loop_node & loop)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(&region))
  {
    // FIXME now we only handle simple nodes
    if (dynamic_cast<const jlm::rvsdg::simple_node *>(node))
    {
      loop.add_node(node);
    }
    else
    {
      JLM_UNREACHABLE("Static HLS only support simple nodes in theta node at this point");
    }
  }
}

static void
ConvertThetaNode(jlm::rvsdg::theta_node & theta)
{
  std::cout << "***** Converting theta node *****" << std::endl;

  auto loop = static_hls::loop_node::create(theta.region());

  // add loopvars and populate the smap
  for (size_t i = 0; i < theta.ninputs(); i++)
  {
    loop->add_loopvar(theta.input(i));
    // divert theta outputs
    theta.output(i)->divert_users(loop->output(i));
  }

  // copy contents of theta
  addSimpleNodes(*theta.subregion(), *loop);

  for (size_t i = 0; i < theta.ninputs(); i++)
  {
    loop->add_loopback_arg(theta.input(i));
  }

  loop->print_nodes_registers();

  std::cout << "**** Printing operations users ****" << std::endl;
  for (auto & node : loop->compute_subregion()->nodes)
  {
    for (size_t i = 0; i < node.ninputs(); i++)
    {
      std::cout << "node " << node.operation().debug_string();
      std::cout << " input " << i;
      std::cout << " users: ";
      auto users = loop->get_users(node.input(i));
      for (auto & user : users)
      {
        if (auto node_out = dynamic_cast<jlm::rvsdg::node_output *>(user))
        {
          std::cout << node_out->node()->operation().debug_string() << ", ";
        }
        else if (auto arg = dynamic_cast<jlm::rvsdg::argument *>(user))
        {
          std::cout << "control arg " << arg->index() << ", ";
        }
      }
      std::cout << std::endl;
    }
  }

  loop->finalize();

  // // copy contents of theta
  // theta.subregion()->copy(loop->subregion(), smap, false, false);
  remove(&theta);
}

static void
ConvertThetaNodesInRegion(jlm::rvsdg::region & region);

static void
ConvertThetaNodesInStructuralNode(jlm::rvsdg::structural_node & structuralNode)
{
  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    ConvertThetaNodesInRegion(*structuralNode.subregion(n));
  }

  if (auto thetaNode = dynamic_cast<jlm::rvsdg::theta_node *>(&structuralNode))
  {
    ConvertThetaNode(*thetaNode);
  }
}

static void
ConvertThetaNodesInRegion(jlm::rvsdg::region & region)
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
