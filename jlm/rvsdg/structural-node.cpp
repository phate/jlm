/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::rvsdg
{

/* structural input */

StructuralInput::~StructuralInput() noexcept
{
  JLM_ASSERT(arguments.empty());

  on_input_destroy(this);
}

StructuralInput::StructuralInput(
    rvsdg::StructuralNode * node,
    jlm::rvsdg::output * origin,
    std::shared_ptr<const rvsdg::Type> type)
    : node_input(origin, node, std::move(type))
{
  on_input_create(this);
}

/* structural output */

StructuralOutput::~StructuralOutput() noexcept
{
  JLM_ASSERT(results.empty());

  on_output_destroy(this);
}

StructuralOutput::StructuralOutput(StructuralNode * node, std::shared_ptr<const rvsdg::Type> type)
    : node_output(node, std::move(type))
{
  on_output_create(this);
}

/* structural node */

StructuralNode::~StructuralNode() noexcept
{
  on_node_destroy(this);

  subregions_.clear();
}

StructuralNode::StructuralNode(
    const StructuralOperation & op,
    rvsdg::Region * region,
    size_t nsubregions)
    : node(op.copy(), region)
{
  if (nsubregions == 0)
    throw jlm::util::error("Number of subregions must be greater than zero.");

  for (size_t n = 0; n < nsubregions; n++)
    subregions_.emplace_back(std::unique_ptr<rvsdg::Region>(new jlm::rvsdg::Region(this, n)));

  on_node_create(this);
}

StructuralInput *
StructuralNode::append_input(std::unique_ptr<StructuralInput> input)
{
  if (input->node() != this)
    throw jlm::util::error("Appending input to wrong node.");

  auto index = input->index();
  JLM_ASSERT(index == 0);
  if (index != 0 || (index == 0 && ninputs() > 0 && this->input(0) == input.get()))
    return this->input(index);

  auto sinput = std::unique_ptr<node_input>(input.release());
  return static_cast<StructuralInput *>(add_input(std::move(sinput)));
}

StructuralOutput *
StructuralNode::append_output(std::unique_ptr<StructuralOutput> output)
{
  if (output->node() != this)
    throw jlm::util::error("Appending output to wrong node.");

  auto index = output->index();
  JLM_ASSERT(index == 0);
  if (index != 0 || (index == 0 && noutputs() > 0 && this->output(0) == output.get()))
    return this->output(index);

  auto soutput = std::unique_ptr<node_output>(output.release());
  return static_cast<StructuralOutput *>(add_output(std::move(soutput)));
}

}
