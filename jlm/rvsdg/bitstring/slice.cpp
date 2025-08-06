/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/arithmetic.hpp>
#include <jlm/rvsdg/bitstring/concat.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/slice.hpp>

namespace jlm::rvsdg
{

BitSliceOperation::~BitSliceOperation() noexcept = default;

bool
BitSliceOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const BitSliceOperation *>(&other);
  return op && op->low() == low() && op->high() == high() && op->argument(0) == argument(0);
}

std::string
BitSliceOperation::debug_string() const
{
  return jlm::util::strfmt("SLICE[", low(), ":", high(), ")");
}

unop_reduction_path_t
BitSliceOperation::can_reduce_operand(const jlm::rvsdg::Output * arg) const noexcept
{
  auto node = TryGetOwnerNode<Node>(*arg);
  auto & arg_type = *std::dynamic_pointer_cast<const bittype>(arg->Type());

  if ((low() == 0) && (high() == arg_type.nbits()))
    return unop_reduction_idempotent;

  if (is<BitSliceOperation>(node))
    return unop_reduction_narrow;

  if (is<bitconstant_op>(node))
    return unop_reduction_constant;

  if (is<BitConcatOperation>(node))
    return unop_reduction_distribute;

  return unop_reduction_none;
}

jlm::rvsdg::Output *
BitSliceOperation::reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const
{
  if (path == unop_reduction_idempotent)
  {
    return arg;
  }

  auto node = static_cast<node_output *>(arg)->node();

  if (path == unop_reduction_narrow)
  {
    auto op = static_cast<const BitSliceOperation &>(node->GetOperation());
    return jlm::rvsdg::bitslice(node->input(0)->origin(), low() + op.low(), high() + op.low());
  }

  if (path == unop_reduction_constant)
  {
    auto op = static_cast<const bitconstant_op &>(node->GetOperation());
    std::string s(&op.value()[0] + low(), high() - low());
    return create_bitconstant(arg->region(), BitValueRepresentation(s.c_str()));
  }

  if (path == unop_reduction_distribute)
  {
    size_t pos = 0, n = 0;
    std::vector<jlm::rvsdg::Output *> arguments;
    for (n = 0; n < node->ninputs(); n++)
    {
      auto argument = node->input(n)->origin();
      size_t base = pos;
      size_t nbits = std::static_pointer_cast<const bittype>(argument->Type())->nbits();
      pos = pos + nbits;
      if (base < high() && pos > low())
      {
        size_t slice_low = (low() > base) ? (low() - base) : 0;
        size_t slice_high = (high() < pos) ? (high() - base) : (pos - base);
        argument = jlm::rvsdg::bitslice(argument, slice_low, slice_high);
        arguments.push_back(argument);
      }
    }

    return jlm::rvsdg::bitconcat(arguments);
  }

  return nullptr;
}

std::unique_ptr<Operation>
BitSliceOperation::copy() const
{
  return std::make_unique<BitSliceOperation>(*this);
}

jlm::rvsdg::Output *
bitslice(jlm::rvsdg::Output * argument, size_t low, size_t high)
{
  return CreateOpNode<BitSliceOperation>(
             { argument },
             std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(argument->Type()),
             low,
             high)
      .output(0);
}

}
