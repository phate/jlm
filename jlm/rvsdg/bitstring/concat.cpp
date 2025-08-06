/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/bitstring/concat.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/slice.hpp>
#include <jlm/rvsdg/reduction-helpers.hpp>

namespace jlm::rvsdg
{

jlm::rvsdg::Output *
bitconcat(const std::vector<jlm::rvsdg::Output *> & operands)
{
  std::vector<std::shared_ptr<const jlm::rvsdg::bittype>> types;
  for (const auto operand : operands)
    types.push_back(std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(operand->Type()));

  return CreateOpNode<BitConcatOperation>(operands, std::move(types)).output(0);
}

std::shared_ptr<const bittype>
BitConcatOperation::aggregate_arguments(
    const std::vector<std::shared_ptr<const bittype>> & types) noexcept
{
  size_t total = 0;
  for (const auto & t : types)
  {
    total += t->nbits();
  }
  return bittype::Create(total);
}

BitConcatOperation::~BitConcatOperation() noexcept = default;

bool
BitConcatOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const BitConcatOperation *>(&other);
  if (!op || op->narguments() != narguments())
    return false;

  for (size_t n = 0; n < narguments(); n++)
  {
    if (op->argument(n) != argument(n))
      return false;
  }

  return true;
}

binop_reduction_path_t
BitConcatOperation::can_reduce_operand_pair(
    const jlm::rvsdg::Output * arg1,
    const jlm::rvsdg::Output * arg2) const noexcept
{
  auto node1 = TryGetOwnerNode<Node>(*arg1);
  auto node2 = TryGetOwnerNode<Node>(*arg2);

  if (!node1 || !node2)
    return binop_reduction_none;

  auto arg1_constant = is<bitconstant_op>(node1);
  auto arg2_constant = is<bitconstant_op>(node2);

  if (arg1_constant && arg2_constant)
  {
    return binop_reduction_constants;
  }

  auto arg1_slice = dynamic_cast<const BitSliceOperation *>(&node1->GetOperation());
  auto arg2_slice = dynamic_cast<const BitSliceOperation *>(&node2->GetOperation());

  if (arg1_slice && arg2_slice)
  {
    auto origin1 = node1->input(0)->origin();
    auto origin2 = node2->input(0)->origin();

    if (origin1 == origin2 && arg1_slice->high() == arg2_slice->low())
    {
      return binop_reduction_merge;
    }

    /* FIXME: support sign bit */
  }

  return binop_reduction_none;
}

jlm::rvsdg::Output *
BitConcatOperation::reduce_operand_pair(
    binop_reduction_path_t path,
    jlm::rvsdg::Output * arg1,
    jlm::rvsdg::Output * arg2) const
{
  auto node1 = static_cast<node_output *>(arg1)->node();
  auto node2 = static_cast<node_output *>(arg2)->node();

  if (path == binop_reduction_constants)
  {
    auto & arg1_constant = static_cast<const bitconstant_op &>(node1->GetOperation());
    auto & arg2_constant = static_cast<const bitconstant_op &>(node2->GetOperation());

    BitValueRepresentation bits(arg1_constant.value());
    bits.Append(arg2_constant.value());
    return create_bitconstant(arg1->region(), std::move(bits));
  }

  if (path == binop_reduction_merge)
  {
    auto arg1_slice = static_cast<const BitSliceOperation *>(&node1->GetOperation());
    auto arg2_slice = static_cast<const BitSliceOperation *>(&node2->GetOperation());
    return jlm::rvsdg::bitslice(node1->input(0)->origin(), arg1_slice->low(), arg2_slice->high());

    /* FIXME: support sign bit */
  }

  return NULL;
}

enum BinaryOperation::flags
BitConcatOperation::flags() const noexcept
{
  return BinaryOperation::flags::associative;
}

std::string
BitConcatOperation::debug_string() const
{
  return "BitConcat";
}

std::unique_ptr<Operation>
BitConcatOperation::copy() const
{
  return std::make_unique<BitConcatOperation>(*this);
}

static std::vector<std::shared_ptr<const bittype>>
GetTypesFromOperands(const std::vector<rvsdg::Output *> & args)
{
  std::vector<std::shared_ptr<const bittype>> types;
  for (const auto arg : args)
  {
    types.push_back(std::dynamic_pointer_cast<const bittype>(arg->Type()));
  }
  return types;
}

std::optional<std::vector<rvsdg::Output *>>
FlattenBitConcatOperation(const BitConcatOperation &, const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(!operands.empty());

  const auto newOperands = base::detail::associative_flatten(
      operands,
      [](jlm::rvsdg::Output * arg)
      {
        // FIXME: switch to comparing operator, not just typeid, after
        // converting "concat" to not be a binary operator anymore
        return is<BitConcatOperation>(TryGetOwnerNode<Node>(*arg));
      });

  if (operands == newOperands)
  {
    JLM_ASSERT(newOperands.size() == 2);
    return std::nullopt;
  }

  JLM_ASSERT(newOperands.size() > 2);
  return outputs(&CreateOpNode<BitConcatOperation>(newOperands, GetTypesFromOperands(newOperands)));
}

}
