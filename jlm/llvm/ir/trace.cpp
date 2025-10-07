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

const rvsdg::Output &
TraceOutput(const rvsdg::Output & startingOutput)
{
  auto & output = rvsdg::TraceOutput(startingOutput);

  if (const auto [node, ioBarrierOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(output);
      node && ioBarrierOp)
  {
    return llvm::TraceOutput(*node->input(0)->origin());
  }

  return output;
}

std::optional<int64_t>
TryGetConstantSignedInteger(const rvsdg::Output & output)
{
  const auto & normalized = llvm::TraceOutput(output);

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
          rvsdg::TryGetSimpleNodeAndOptionalOp<rvsdg::bitconstant_op>(normalized);
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
