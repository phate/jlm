/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/Trace.hpp>

namespace jlm::llvm
{

llvm::OutputTracer::OutputTracer() = default;

rvsdg::Output &
OutputTracer::traceStep(rvsdg::Output & output, bool mayLeaveRegion)
{
  // FIXME: Needing to create a custom subclass of OutputTracer to make it handle a single LLVM
  // specific operation is not great, as we now have multiple choices for traceOutput.
  // It would be better to have a single tracing class that handles all operations,
  // and somehow marking the IOBarrier with a "trait" that makes the output map to the input.

  auto & trace1 = rvsdg::OutputTracer::traceStep(output, mayLeaveRegion);

  if (const auto [node, ioBarrierOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(trace1);
      node && ioBarrierOp)
  {
    return *IOBarrierOperation::BarredInput(*node).origin();
  }

  return trace1;
}

rvsdg::Output &
traceOutput(rvsdg::Output & output)
{
  OutputTracer tracer;
  return tracer.trace(output);
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
