/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::rvsdg
{

/* structural input */

StructuralInput::~StructuralInput() noexcept
{
  JLM_ASSERT(arguments.empty());
}

StructuralInput::StructuralInput(
    rvsdg::StructuralNode * node,
    jlm::rvsdg::Output * origin,
    std::shared_ptr<const rvsdg::Type> type)
    : NodeInput(origin, node, std::move(type))
{}

/* structural output */

StructuralOutput::~StructuralOutput() noexcept
{
  JLM_ASSERT(results.empty());
}

StructuralOutput::StructuralOutput(StructuralNode * node, std::shared_ptr<const rvsdg::Type> type)
    : NodeOutput(node, std::move(type))
{}

/* structural node */

StructuralNode::~StructuralNode() noexcept
{
  region()->notifyNodeDestroy(this);

  subregions_.clear();
}

StructuralNode::StructuralNode(rvsdg::Region * region, size_t nsubregions)
    : Node(region)
{
  if (nsubregions == 0)
    throw util::Error("Number of subregions must be greater than zero.");

  for (size_t n = 0; n < nsubregions; n++)
    subregions_.emplace_back(std::unique_ptr<rvsdg::Region>(new jlm::rvsdg::Region(this, n)));

  region->notifyNodeCreate(this);
}

std::string
StructuralNode::DebugString() const
{
  return GetOperation().debug_string();
}

}
