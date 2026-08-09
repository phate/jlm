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
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/region.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/Trace.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/Math.hpp>

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
    const auto inputBits = sextOp->nsrcbits();
    return util::truncateAndSignExtend(*inputValue, inputBits);
  }

  if (const auto [zextNode, zextOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<ZExtOperation>(normalized);
      zextOp)
  {
    const auto inputValue = tryGetConstantSignedInteger(*zextNode->input(0)->origin());
    if (!inputValue.has_value())
      return std::nullopt;

    // When doing zero extensions, we only need to care about the size of the input type
    const auto inputBits = zextOp->nsrcbits();
    return util::truncateAndZeroExtend(*inputValue, inputBits);
  }

  if (const auto [truncNode, truncOp] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<TruncOperation>(normalized);
      truncOp)
  {
    const auto inputValue = tryGetConstantSignedInteger(*truncNode->input(0)->origin());
    if (!inputValue.has_value())
      return std::nullopt;

    const auto outputBits = truncOp->ndstbits();
    // When truncating, we still need to fill the high bits with something.
    // We chose to sign extend, but this is mainly an aestetic choice
    return util::truncateAndSignExtend(*inputValue, outputBits);
  }

  return std::nullopt;
}

std::optional<int64_t>
TracedPointerOrigin::getOffsetInBytes() const noexcept
{
  if (!gepConstants.has_value())
    return std::nullopt;

  int64_t offsetInBytes = 0;
  for (auto gepConstant : *gepConstants)
  {
    offsetInBytes += gepConstant.getOffsetInBytes();
  }

  return offsetInBytes;
}

TracedPointerOrigin
TracePointerOriginPrecise(const rvsdg::Output & p)
{
  const rvsdg::Output * base = &p;
  std::vector<GetElementPtrOperation::Constant> gepConstants;

  while (true)
  {
    // Use normalization function to get past all trivially invariant operations
    base = &llvm::traceOutput(*base);

    if (const auto [gepNode, gepOperation] =
            rvsdg::TryGetSimpleNodeAndOptionalOp<GetElementPtrOperation>(*base);
        gepOperation)
    {
      if (const auto gepConstantOpt = GetElementPtrOperation::tryGetAsConstant(*gepNode);
          gepConstantOpt.has_value())
      {
        base = gepNode->input(0)->origin();
        gepConstants.emplace_back(gepConstantOpt.value());
        continue;
      }
    }

    // We were not able to trace further
    break;
  }

  return TracedPointerOrigin{ base, gepConstants };
}

static bool
traceAllPointerOriginsInternal(
    const rvsdg::Output * basePointer,
    std::optional<int64_t> offsetInBytes,
    TraceCollection & traceCollection,
    const size_t maxTraceCollectionSize)
{
  if (traceCollection.AllTracedOutputs.size() >= maxTraceCollectionSize)
    return false;

  // Normalize the pointer first, to avoid tracing trivial temporary outputs
  basePointer = &llvm::traceOutput(*basePointer);

  auto it = traceCollection.AllTracedOutputs.find(basePointer);
  if (it != traceCollection.AllTracedOutputs.end())
  {
    // If the base pointer has already been traced with an unknown offset, we have nothing to add
    if (!it->second.has_value())
      return true;

    // The offset used for the base pointer the last time it was traced
    const auto prevOffset = *it->second;

    // If we are visiting the same base pointer again with the same offset, we have nothing to add
    if (offsetInBytes.has_value() && *offsetInBytes == prevOffset)
      return true;

    // We have different offsets to last time, collapse to unknown offset
    offsetInBytes = std::nullopt;
  }

  traceCollection.AllTracedOutputs[basePointer] = offsetInBytes;

  // If it is a GEP, we can trace through it, but possibly lose precise offset information
  if (const auto [gepNode, gepOperation] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<GetElementPtrOperation>(*basePointer);
      gepOperation)
  {
    // Update the base pointer and offset to represent the other side of the GEP
    basePointer = GetElementPtrOperation::getBaseAddressInput(*gepNode).origin();

    // If we have precisely tracked the offset so far, try updating it with the GEPs offset
    if (offsetInBytes.has_value())
    {
      if (const auto gepConstant = GetElementPtrOperation::tryGetAsConstant(*gepNode);
          gepConstant.has_value())
        offsetInBytes = *offsetInBytes + gepConstant->getOffsetInBytes();
      else
        offsetInBytes = std::nullopt;
    }

    return traceAllPointerOriginsInternal(
        basePointer,
        offsetInBytes,
        traceCollection,
        maxTraceCollectionSize);
  }

  // If the node is a \ref SelectOperation, trace through both possible inputs
  if (const auto [node, select] =
          rvsdg::TryGetSimpleNodeAndOptionalOp<SelectOperation>(*basePointer);
      select)
  {
    return traceAllPointerOriginsInternal(
               node->input(1)->origin(),
               offsetInBytes,
               traceCollection,
               maxTraceCollectionSize)
        && traceAllPointerOriginsInternal(
               node->input(2)->origin(),
               offsetInBytes,
               traceCollection,
               maxTraceCollectionSize);
  }

  // If we reach undef nodes, do not include them in the TopOrigins
  if (rvsdg::IsOwnerNodeOperation<UndefValueOperation>(*basePointer))
  {
    return true;
  }

  // Trace into gamma nodes
  if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*basePointer))
  {
    auto exitVar = gamma->MapOutputExitVar(*basePointer);
    for (auto result : exitVar.branchResult)
    {
      // If tracing gives up, we give up
      if (!traceAllPointerOriginsInternal(
              result->origin(),
              offsetInBytes,
              traceCollection,
              maxTraceCollectionSize))
        return false;
    }

    return true;
  }

  // Normalization never stops at a gamma entry variable
  JLM_ASSERT(!rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*basePointer));

  // Trace into theta nodes
  if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*basePointer))
  {
    auto loopVar = theta->MapOutputLoopVar(*basePointer);

    // Invariant loop variables should already have been handled by normalization
    JLM_ASSERT(!rvsdg::ThetaLoopVarIsInvariant(loopVar));
    return traceAllPointerOriginsInternal(
        loopVar.post->origin(),
        offsetInBytes,
        traceCollection,
        maxTraceCollectionSize);
  }

  // Trace loop variable pre arguments in theta nodes
  if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*basePointer))
  {
    auto loopVar = theta->MapPreLoopVar(*basePointer);

    // Invariant loop variables should already have been handled by normalization
    JLM_ASSERT(!rvsdg::ThetaLoopVarIsInvariant(loopVar));

    // Trace both from the post and the loop variable input
    return traceAllPointerOriginsInternal(
               loopVar.post->origin(),
               offsetInBytes,
               traceCollection,
               maxTraceCollectionSize)
        && traceAllPointerOriginsInternal(
               loopVar.input->origin(),
               offsetInBytes,
               traceCollection,
               maxTraceCollectionSize);
  }

  // We could not trace further, add p as a TopOrigin
  traceCollection.TopOrigins[basePointer] = offsetInBytes;
  return true;
}

bool
TraceAllPointerOrigins(
    TracedPointerOrigin p,
    TraceCollection & traceCollection,
    const size_t maxTraceCollectionSize)
{
  return traceAllPointerOriginsInternal(
      p.BasePointer,
      p.getOffsetInBytes(),
      traceCollection,
      maxTraceCollectionSize);
}

}
