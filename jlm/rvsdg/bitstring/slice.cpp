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
  return unop_reduction_none;
}

jlm::rvsdg::Output *
BitSliceOperation::reduce_operand(unop_reduction_path_t path, jlm::rvsdg::Output * arg) const
{
  return nullptr;
}

std::optional<std::vector<Output *>>
BitSliceOperation::normalizeIdempotent(
    const BitSliceOperation & operation,
    const std::vector<Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto operand = operands[0];

  const auto bitType = util::assertedCast<const BitType>(operand->Type().get());
  if (operation.low() == 0 && operation.high() == bitType->nbits())
  {
    return std::vector{ operand };
  }

  return std::nullopt;
}

std::optional<std::vector<Output *>>
BitSliceOperation::foldConstant(
    const BitSliceOperation & operation,
    const std::vector<Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto operand = operands[0];

  auto [_, constantOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConstantOperation>(*operand);
  if (constantOperation)
  {
    const auto slicedValue = constantOperation->value().slice(operation.low(), operation.high());
    return std::vector{ &BitConstantOperation::create(*operand->region(), slicedValue) };
  }

  return std::nullopt;
}

std::optional<std::vector<Output *>>
BitSliceOperation::narrowSlice(
    const BitSliceOperation & operation,
    const std::vector<Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto operand = operands[0];

  auto [sliceNode, sliceOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitSliceOperation>(*operand);
  if (sliceOperation)
  {
    const auto newLow = operation.low() + sliceOperation->low();
    const auto newHigh = operation.high() + sliceOperation->low();
    return std::vector{ bitslice(sliceNode->input(0)->origin(), newLow, newHigh) };
  }

  return std::nullopt;
}

std::optional<std::vector<Output *>>
BitSliceOperation::distributeSlice(
    const BitSliceOperation & operation,
    const std::vector<Output *> & operands)
{
  JLM_ASSERT(operands.size() == 1);
  const auto operand = operands[0];
  const auto low = operation.low();
  const auto high = operation.high();

  auto [bitConcatNode, bitConcatOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<BitConcatOperation>(*operand);
  if (bitConcatOperation)
  {
    size_t pos = 0, n = 0;
    std::vector<Output *> newOperands;
    for (n = 0; n < bitConcatNode->ninputs(); n++)
    {
      auto argument = bitConcatNode->input(n)->origin();
      const auto base = pos;
      const auto numBits = std::static_pointer_cast<const BitType>(argument->Type())->nbits();
      pos = pos + numBits;
      if (base < high && pos > low)
      {
        const auto slice_low = (low > base) ? (low - base) : 0;
        const auto slice_high = (high < pos) ? (high - base) : (pos - base);
        argument = bitslice(argument, slice_low, slice_high);
        newOperands.push_back(argument);
      }
    }

    return std::vector{ bitconcat(newOperands) };
  }

  return std::nullopt;
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
             std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(argument->Type()),
             low,
             high)
      .output(0);
}

}
