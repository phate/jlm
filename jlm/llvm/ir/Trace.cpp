/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/ConversionOperations.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/bitstring/constant.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>

namespace jlm::llvm
{

OutputTracer::OutputTracer(const bool enableCaching)
    : rvsdg::OutputTracer(enableCaching)
{}

rvsdg::Output &
OutputTracer::traceStep(rvsdg::Output & output, const rvsdg::Region * withinRegion)
{
  auto & trace1 = rvsdg::OutputTracer::traceStep(output, withinRegion);

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
traceOutput(rvsdg::Output & output, const rvsdg::Region * withinRegion)
{
  constexpr bool enableCaching = false;
  OutputTracer tracer(enableCaching);
  return tracer.trace(output, withinRegion);
}

std::optional<int64_t>
tryGetConstantSignedInteger(const rvsdg::Output & output)
{
  const auto & normalized = llvm::traceOutput(output, nullptr);

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

  if (const auto [sextNode, sextOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<SExtOperation>(normalized);
      sextOp)
  {
    const auto inputValue = tryGetConstantSignedInteger(*sextNode->input(0)->origin());
    if (!inputValue.has_value())
      return std::nullopt;

    // When doing sign extensions, we only need to care about the size of the input type
    const auto inputType = sextOp->argument(0);
    const auto inputBits = util::assertedCast<const rvsdg::BitType>(inputType.get())->nbits();
    JLM_ASSERT(inputBits <= 64);
    const auto extendBits = 64 - inputBits;

    // Shift signed value left and right again to sign extend
    return (static_cast<int64_t>(*inputValue) << extendBits) >> extendBits;
  }

  if (const auto [zextNode, zextOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<ZExtOperation>(normalized);
      zextOp)
  {
    const auto inputValue = tryGetConstantSignedInteger(*zextNode->input(0)->origin());
    if (!inputValue.has_value())
      return std::nullopt;

    // When doing zero extensions, we only need to care about the size of the input type
    const auto inputType = zextOp->argument(0);
    const auto inputBits = util::assertedCast<const rvsdg::BitType>(inputType.get())->nbits();
    JLM_ASSERT(inputBits <= 64);
    const auto extendBits = 64 - inputBits;

    // Shift unsigned value left and right again to zero extend
    return (static_cast<uint64_t>(*inputValue) << extendBits) >> extendBits;
  }

  if (const auto [truncNode, truncOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<TruncOperation>(normalized);
      truncOp)
  {
    const auto inputValue = tryGetConstantSignedInteger(*truncNode->input(0)->origin());
    if (!inputValue.has_value())
      return std::nullopt;

    // When truncating, we keep only the desired output bits and sign extend the rest
    const auto outputType = truncOp->result(0);
    const auto outputBits = util::assertedCast<const rvsdg::BitType>(outputType.get())->nbits();
    JLM_ASSERT(outputBits <= 64);
    const auto eraseBits = 64 - outputBits;

    // Shift signed value left and right again to clear the top eraseBits with sign bits
    return (static_cast<int64_t>(*inputValue) << eraseBits) >> eraseBits;
  }

  return std::nullopt;
}

TracedPointerOrigin
TracePointerOriginPrecise(const rvsdg::Output & p)
{
  // The original pointer p is always equal to base + byte offset
  const rvsdg::Output * base = &p;
  int64_t offset = 0;

  while (true)
  {
    // Use normalization function to get past all trivially invariant operations
    base = &llvm::traceOutput(*base);

    if (const auto [node, gep] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<GetElementPtrOperation>(*base);
        gep)
    {
      auto calculatedOffset = GetElementPtrOperation::CalculateOffset(*node);

      // Only trace through GEPs with statically known offsets
      if (!calculatedOffset.has_value())
        break;

      base = node->input(0)->origin();
      offset += *calculatedOffset;

      continue;
    }

    // We were not able to trace further
    break;
  }

  return TracedPointerOrigin{ base, offset };
}

bool
TraceAllPointerOrigins(
    TracedPointerOrigin p,
    TraceCollection & traceCollection,
    const size_t maxTraceCollectionSize)
{
  if (traceCollection.AllTracedOutputs.size() >= maxTraceCollectionSize)
    return false;

  // Normalize the pointer first, to avoid tracing trivial temporary outputs
  p.BasePointer = &llvm::traceOutput(*p.BasePointer);

  auto it = traceCollection.AllTracedOutputs.find(p.BasePointer);
  if (it != traceCollection.AllTracedOutputs.end())
  {
    // If the base pointer has already been traced with an unknown offset, we have nothing to add
    if (!it->second.has_value())
      return true;

    // The offset used for the base pointer the last time it was traced
    const auto prevOffset = *it->second;

    // If we are visiting the same base pointer again with the same offset, we have nothing to add
    if (p.Offset.has_value() && *p.Offset == prevOffset)
      return true;

    // We have different offsets to last time, collapse to unknown offset
    p.Offset = std::nullopt;
  }

  traceCollection.AllTracedOutputs[p.BasePointer] = p.Offset;

  // If it is a GEP, we can trace through it, but possibly lose precise offset information
  if (const auto [node, gep] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<GetElementPtrOperation>(*p.BasePointer);
      gep)
  {
    // Update the base pointer and offset to represent the other side of the GEP
    p.BasePointer = node->input(0)->origin();

    // If we have precisely tracked the offset so far, try updating it with the GEPs offset
    if (p.Offset.has_value())
    {
      const auto gepOffset = GetElementPtrOperation::CalculateOffset(*node);
      if (gepOffset.has_value())
        p.Offset = *p.Offset + *gepOffset;
      else
        p.Offset = std::nullopt;
    }

    return TraceAllPointerOrigins(p, traceCollection, maxTraceCollectionSize);
  }

  // If the node is a \ref SelectOperation, trace through both possible inputs
  if (const auto [node, select] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<SelectOperation>(*p.BasePointer);
      select)
  {
    auto leftTrace = p;
    leftTrace.BasePointer = node->input(1)->origin();
    auto rightTrace = p;
    rightTrace.BasePointer = node->input(2)->origin();

    return TraceAllPointerOrigins(leftTrace, traceCollection, maxTraceCollectionSize)
        && TraceAllPointerOrigins(rightTrace, traceCollection, maxTraceCollectionSize);
  }

  // If we reach undef nodes, do not include them in the TopOrigins
  if (rvsdg::IsOwnerNodeOperation<UndefValueOperation>(*p.BasePointer))
  {
    return true;
  }

  // Trace into gamma nodes
  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*p.BasePointer))
  {
    auto exitVar = gamma->MapOutputExitVar(*p.BasePointer);
    for (auto result : exitVar.branchResult)
    {
      TracedPointerOrigin inside = { result->origin(), p.Offset };

      // If tracing gives up, we give up
      if (!TraceAllPointerOrigins(inside, traceCollection, maxTraceCollectionSize))
        return false;
    }

    return true;
  }

  // Trace into theta nodes
  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*p.BasePointer))
  {
    auto loopVar = theta->MapOutputLoopVar(*p.BasePointer);

    // Invariant loop variables should already have been handled by normalization
    JLM_ASSERT(!rvsdg::ThetaLoopVarIsInvariant(loopVar));

    TracedPointerOrigin inside = { loopVar.post->origin(), p.Offset };
    return TraceAllPointerOrigins(inside, traceCollection, maxTraceCollectionSize);
  }

  // We could not trace further, add p as a TopOrigin
  traceCollection.TopOrigins[p.BasePointer] = p.Offset;
  return true;
}

}
