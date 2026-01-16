/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::rvsdg
{

SimpleNode::~SimpleNode()
{
  region()->notifyNodeDestroy(this);
}

SimpleNode::SimpleNode(
    rvsdg::Region & region,
    std::unique_ptr<SimpleOperation> operation,
    const std::vector<jlm::rvsdg::Output *> & operands)
    : Node(&region),
      Operation_(std::move(operation))
{
  if (GetOperation().narguments() != operands.size())
    throw util::Error(jlm::util::strfmt(
        "Argument error - expected ",
        SimpleNode::GetOperation().narguments(),
        ", received ",
        operands.size(),
        " arguments."));

  for (size_t n = 0; n < SimpleNode::GetOperation().narguments(); n++)
  {
    addInput(
        std::make_unique<NodeInput>(operands[n], this, SimpleNode::GetOperation().argument(n)),
        false);
  }

  for (size_t n = 0; n < SimpleNode::GetOperation().nresults(); n++)
    addOutput(std::make_unique<NodeOutput>(this, SimpleNode::GetOperation().result(n)));

  region.notifyNodeCreate(this);
}

const SimpleOperation &
SimpleNode::GetOperation() const noexcept
{
  return *Operation_;
}

Node *
SimpleNode::copy(Region * region, const std::vector<Output *> & operands) const
{
  return &Create(*region, GetOperation().copy(), operands);
}

Node *
SimpleNode::copy(Region * region, SubstitutionMap & smap) const
{
  std::vector<Output *> operands;
  for (auto & input : Inputs())
  {
    auto & operand = smap.lookup(*input.origin());
    operands.push_back(&operand);
  }

  auto copiedNode = copy(region, operands);

  for (size_t n = 0; n < copiedNode->noutputs(); n++)
    smap.insert(output(n), copiedNode->output(n));

  return copiedNode;
}

std::string
SimpleNode::DebugString() const
{
  return GetOperation().debug_string();
}

std::optional<std::vector<rvsdg::Output *>>
NormalizeSimpleOperationCommonNodeElimination(
    Region & region,
    const SimpleOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  auto isCongruent = [&](const Node & node)
  {
    auto simpleNode = dynamic_cast<const SimpleNode *>(&node);
    return simpleNode && simpleNode->GetOperation() == operation
        && operands == rvsdg::operands(&node) && &simpleNode->GetOperation() != &operation;
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
    for (const auto & user : operands[0]->Users())
    {
      if (const auto node = TryGetOwnerNode<SimpleNode>(user))
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
