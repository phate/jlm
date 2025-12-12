/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/trace.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/simple-node.hpp>

namespace jlm::llvm
{

rvsdg::Output &
traceOutput(rvsdg::Output & startingOutput)
{
  auto & output = rvsdg::traceOutput(startingOutput);

  if (const auto [node, ioBarrierOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(output);
      node && ioBarrierOp)
  {
    return llvm::traceOutput(*node->input(0)->origin());
  }

  return output;
}

const rvsdg::Output &
traceOutput(const rvsdg::Output & startingOutput)
{
  return llvm::traceOutput(const_cast<rvsdg::Output &>(startingOutput));
}

std::optional<int64_t>
tryGetConstantSignedInteger(const rvsdg::Output & output)
{
  const auto & normalized = llvm::traceOutput(output);

  if (const auto [_, constant] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IntegerConstantOperation>(normalized);
      constant)
  {
    const auto & rep = constant->Representation();
    if (rep.is_known() && rep.nbits() <= 64)
      return rep.to_int();
    return std::nullopt;
  }

  if (const auto [_, constant] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::BitConstantOperation>(normalized);
      constant)
  {
    const auto & rep = constant->value();
    if (rep.is_known() && rep.nbits() <= 64)
      return rep.to_int();
    return std::nullopt;
  }

  return std::nullopt;
}

}
