/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/unary.hpp>

namespace jlm::rvsdg
{

UnaryOperation::~UnaryOperation() noexcept
{}

std::optional<std::vector<rvsdg::Output *>>
NormalizeUnaryOperation(
    const UnaryOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  auto & operand = *operands[0];

  if (const auto reduction = operation.can_reduce_operand(&operand);
      reduction != unop_reduction_none)
  {
    return { { operation.reduce_operand(reduction, &operand) } };
  }

  return std::nullopt;
}

}
