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
    jlm::rvsdg::SimpleNode * node,
    jlm::rvsdg::output * origin,
    std::shared_ptr<const rvsdg::Type> type)
    : node_input(origin, node, std::move(type))
{}

/* outputs */

simple_output::simple_output(jlm::rvsdg::SimpleNode * node, std::shared_ptr<const rvsdg::Type> type)
    : node_output(node, std::move(type))
{}

simple_output::~simple_output() noexcept
{
  on_output_destroy(this);
}

SimpleNode::~SimpleNode()
{
  on_node_destroy(this);
}

SimpleNode::SimpleNode(
    rvsdg::Region & region,
    std::unique_ptr<SimpleOperation> operation,
    const std::vector<jlm::rvsdg::output *> & operands)
    : Node(&region),
      Operation_(std::move(operation))
{
  if (GetOperation().narguments() != operands.size())
    throw jlm::util::error(jlm::util::strfmt(
        "Argument error - expected ",
        SimpleNode::GetOperation().narguments(),
        ", received ",
        operands.size(),
        " arguments."));

  for (size_t n = 0; n < SimpleNode::GetOperation().narguments(); n++)
  {
    add_input(
        std::make_unique<simple_input>(this, operands[n], SimpleNode::GetOperation().argument(n)));
  }

  for (size_t n = 0; n < SimpleNode::GetOperation().nresults(); n++)
    add_output(std::make_unique<simple_output>(this, SimpleNode::GetOperation().result(n)));

  on_node_create(this);
}

const SimpleOperation &
SimpleNode::GetOperation() const noexcept
{
  return *Operation_;
}

Node *
SimpleNode::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  std::unique_ptr<SimpleOperation> operation(
      util::AssertedCast<SimpleOperation>(GetOperation().copy().release()));
  return &Create(*region, std::move(operation), operands);
}

Node *
SimpleNode::copy(rvsdg::Region * region, SubstitutionMap & smap) const
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

std::optional<std::vector<rvsdg::output *>>
NormalizeSimpleOperationCommonNodeElimination(
    Region & region,
    const SimpleOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  auto isCongruent = [&](const Node & node)
  {
    auto & nodeOperation = node.GetOperation();
    return nodeOperation == operation && operands == rvsdg::operands(&node)
        && &nodeOperation != &operation;
  };

  if (operands.empty())
  {
    for (auto & node : region.TopNodes())
    {
      if (isCongruent(node))
      {
        return outputs(&node);
      }
    }
  }
  else
  {
    for (const auto & user : *operands[0])
    {
      if (const auto node = TryGetOwnerNode<SimpleNode>(*user))
      {
        if (isCongruent(*node))
        {
          return outputs(node);
        }
      }
    }
  }

  return std::nullopt;
}

}
