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
    std::shared_ptr<const rvsdg::type> type)
    : node_input(origin, node, std::move(type))
{}

/* outputs */

simple_output::simple_output(
    jlm::rvsdg::simple_node * node,
    std::shared_ptr<const rvsdg::type> type)
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
    const jlm::rvsdg::simple_op & op,
    const std::vector<jlm::rvsdg::output *> & operands)
    : node(op.copy(), region)
{
  if (operation().narguments() != operands.size())
    throw jlm::util::error(jlm::util::strfmt(
        "Argument error - expected ",
        operation().narguments(),
        ", received ",
        operands.size(),
        " arguments."));

  for (size_t n = 0; n < operation().narguments(); n++)
  {
    node::add_input(std::make_unique<simple_input>(this, operands[n], operation().argument(n)));
  }

  for (size_t n = 0; n < operation().nresults(); n++)
    node::add_output(std::make_unique<simple_output>(this, operation().result(n)));

  on_node_create(this);
}

jlm::rvsdg::node *
simple_node::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  auto node = create(region, *static_cast<const simple_op *>(&operation()), operands);
  graph()->mark_denormalized();
  return node;
}

jlm::rvsdg::node *
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
