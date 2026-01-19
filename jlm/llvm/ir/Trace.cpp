/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/rvsdg/type.hpp>

namespace jlm::llvm
{

llvm::OutputTracer::OutputTracer() = default;

rvsdg::Output &
OutputTracer::traceStep(rvsdg::Output & output, bool mayLeaveRegion)
{
  auto & trace1 = rvsdg::OutputTracer::traceStep(output, mayLeaveRegion);

  if (const auto [node, ioBarrierOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(trace1);
      node && ioBarrierOp)
  {
    return *IOBarrierOperation::BarredInput(*node).origin();
  }

  // If enabled, try tracing through the memory states of load nodes
  if (traceThroughLoadedStates_)
  {
    if (const auto [node, loadOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<LoadOperation>(trace1);
        node && loadOp)
    {
      if (is<MemoryStateType>(trace1.Type()))
      {
        // Map the memory state output to the corresponding memory state input
        auto & memoryStateInput = LoadOperation::MapMemoryStateOutputToInput(trace1);
        return *memoryStateInput.origin();
      }
    }
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
