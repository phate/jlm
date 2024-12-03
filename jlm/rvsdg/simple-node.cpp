/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::rvsdg
{

/* inputs */

simple_input::~simple_input() noexcept
{
  on_input_destroy(this);
}

simple_input::simple_input(
    jlm::rvsdg::simple_node * node,
    jlm::rvsdg::output * origin,
    std::shared_ptr<const rvsdg::Type> type)
    : node_input(origin, node, std::move(type))
{}

/* outputs */

simple_output::simple_output(
    jlm::rvsdg::simple_node * node,
    std::shared_ptr<const rvsdg::Type> type)
    : node_output(node, std::move(type))
{}

simple_output::~simple_output() noexcept
{
  on_output_destroy(this);
}

/* simple nodes */

simple_node::~simple_node()
{
  on_node_destroy(this);
}

simple_node::simple_node(
    rvsdg::Region * region,
    const SimpleOperation & op,
    const std::vector<jlm::rvsdg::output *> & operands)
    : Node(op.copy(), region)
{
  if (simple_node::GetOperation().narguments() != operands.size())
    throw jlm::util::error(jlm::util::strfmt(
        "Argument error - expected ",
        simple_node::GetOperation().narguments(),
        ", received ",
        operands.size(),
        " arguments."));

  for (size_t n = 0; n < simple_node::GetOperation().narguments(); n++)
  {
    add_input(
        std::make_unique<simple_input>(this, operands[n], simple_node::GetOperation().argument(n)));
  }

  for (size_t n = 0; n < simple_node::GetOperation().nresults(); n++)
    add_output(std::make_unique<simple_output>(this, simple_node::GetOperation().result(n)));

  on_node_create(this);
}

const SimpleOperation &
simple_node::GetOperation() const noexcept
{
  return *util::AssertedCast<const SimpleOperation>(&Node::GetOperation());
}

Node *
simple_node::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  auto node = create(region, GetOperation(), operands);
  graph()->mark_denormalized();
  return node;
}

Node *
simple_node::copy(rvsdg::Region * region, SubstitutionMap & smap) const
{
  std::vector<jlm::rvsdg::output *> operands;
  for (size_t n = 0; n < ninputs(); n++)
  {
    auto origin = input(n)->origin();
    auto operand = smap.lookup(origin);

    if (operand == nullptr)
    {
      if (region != this->region())
        throw jlm::util::error("Node operand not in substitution map.");

      operand = origin;
    }

    operands.push_back(operand);
  }

  auto node = copy(region, operands);

  JLM_ASSERT(node->noutputs() == noutputs());
  for (size_t n = 0; n < node->noutputs(); n++)
    smap.insert(output(n), node->output(n));

  return node;
}

}
