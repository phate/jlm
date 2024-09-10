/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/GammaConversion.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{

static void
ConvertGammaNodeWithoutSpeculation(rvsdg::GammaNode & gammaNode)
{
  rvsdg::substitution_map substitutionMap;

  // create a branch for each gamma input and map the corresponding argument of each subregion to an
  // output of the branch
  for (size_t i = 0; i < gammaNode.nentryvars(); i++)
  {
    auto branchResults =
        hls::branch_op::create(*gammaNode.predicate()->origin(), *gammaNode.entryvar(i)->origin());

    for (size_t s = 0; s < gammaNode.nsubregions(); s++)
    {
      substitutionMap.insert(gammaNode.subregion(s)->argument(i), branchResults[s]);
    }
  }

  for (size_t s = 0; s < gammaNode.nsubregions(); s++)
  {
    gammaNode.subregion(s)->copy(gammaNode.region(), substitutionMap, false, false);
  }

  for (size_t i = 0; i < gammaNode.nexitvars(); i++)
  {
    std::vector<rvsdg::output *> alternatives;
    for (size_t s = 0; s < gammaNode.nsubregions(); s++)
    {
      alternatives.push_back(substitutionMap.lookup(gammaNode.subregion(s)->result(i)->origin()));
    }
    // create mux nodes for each gamma output
    // use mux instead of merge in case of paths with different delay - otherwise one could overtake
    // the other see https://ieeexplore.ieee.org/abstract/document/9515491
    auto mux = hls::mux_op::create(*gammaNode.predicate()->origin(), alternatives, false);

    gammaNode.exitvar(i)->divert_users(mux[0]);
  }

  remove(&gammaNode);
}

static void
ConvertGammaNodeWithSpeculation(rvsdg::GammaNode & gammaNode)
{
  rvsdg::substitution_map substitutionMap;

  // Map arguments to origins of inputs. Forks will automatically be created later
  for (size_t i = 0; i < gammaNode.nentryvars(); i++)
  {
    auto gammaInput = gammaNode.entryvar(i);

    for (size_t s = 0; s < gammaNode.nsubregions(); s++)
    {
      substitutionMap.insert(gammaNode.subregion(s)->argument(i), gammaInput->origin());
    }
  }

  for (size_t s = 0; s < gammaNode.nsubregions(); s++)
  {
    gammaNode.subregion(s)->copy(gammaNode.region(), substitutionMap, false, false);
  }

  for (size_t i = 0; i < gammaNode.nexitvars(); i++)
  {
    std::vector<rvsdg::output *> alternatives;
    for (size_t s = 0; s < gammaNode.nsubregions(); s++)
    {
      alternatives.push_back(substitutionMap.lookup(gammaNode.subregion(s)->result(i)->origin()));
    }

    // create discarding mux for each gamma output
    auto merge = hls::mux_op::create(*gammaNode.predicate()->origin(), alternatives, true);

    gammaNode.exitvar(i)->divert_users(merge[0]);
  }

  remove(&gammaNode);
}

static bool
CanGammaNodeBeSpeculative(const rvsdg::GammaNode & gammaNode)
{
  for (size_t i = 0; i < gammaNode.noutputs(); ++i)
  {
    auto gammaOutput = gammaNode.output(i);
    if (rvsdg::is<rvsdg::statetype>(gammaOutput->type()))
    {
      // don't allow state outputs since they imply operations with side effects
      return false;
    }
  }

  for (size_t i = 0; i < gammaNode.nsubregions(); ++i)
  {
    for (auto & node : gammaNode.subregion(i)->nodes)
    {
      if (rvsdg::is<rvsdg::theta_op>(&node) || rvsdg::is<hls::loop_op>(&node))
      {
        // don't allow thetas or loops since they could potentially block forever
        return false;
      }
      else if (auto innerGammaNode = dynamic_cast<rvsdg::GammaNode *>(&node))
      {
        if (!CanGammaNodeBeSpeculative(*innerGammaNode))
        {
          // only allow gammas that can also be speculated on
          return false;
        }
      }
      else if (rvsdg::is<rvsdg::structural_op>(&node))
      {
        throw util::error("Unexpected structural node: " + node.operation().debug_string());
      }
    }
  }

  return true;
}

static void
ConvertGammaNodesInRegion(rvsdg::region & region);

static void
ConvertGammaNodesInStructuralNode(rvsdg::structural_node & structuralNode)
{
  for (size_t n = 0; n < structuralNode.nsubregions(); n++)
  {
    ConvertGammaNodesInRegion(*structuralNode.subregion(n));
  }

  if (auto gammaNode = dynamic_cast<rvsdg::GammaNode *>(&structuralNode))
  {
    if (CanGammaNodeBeSpeculative(*gammaNode))
    {
      ConvertGammaNodeWithSpeculation(*gammaNode);
    }
    else
    {
      ConvertGammaNodeWithoutSpeculation(*gammaNode);
    }
  }
}

static void
ConvertGammaNodesInRegion(rvsdg::region & region)
{
  for (auto & node : rvsdg::topdown_traverser(&region))
  {
    if (auto structuralNode = dynamic_cast<rvsdg::structural_node *>(node))
    {
      ConvertGammaNodesInStructuralNode(*structuralNode);
    }
  }
}

void
ConvertGammaNodes(llvm::RvsdgModule & rvsdgModule)
{
  ConvertGammaNodesInRegion(*rvsdgModule.Rvsdg().root());
}

}
