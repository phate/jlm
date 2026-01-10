/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/opt/alias-analyses/LocalAliasAnalysis.hpp>

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/MatchType.hpp>
#include <jlm/rvsdg/theta.hpp>

#include <numeric>
#include <queue>

namespace jlm::llvm::aa
{

LocalAliasAnalysis::LocalAliasAnalysis() = default;

LocalAliasAnalysis::~LocalAliasAnalysis() noexcept = default;

std::string
LocalAliasAnalysis::ToString() const
{
  return "LocalAA";
}

/**
 * Represents the result of tracing a pointer p to some origin,
 * as a traced base pointer value plus an optional byte offset.
 *
 * If the offset is present, that means
 *  p = base pointer + offset
 *
 * If the offset is not present, that means
 *  p = base pointer + [unknown offset]
 */
struct LocalAliasAnalysis::TracedPointerOrigin
{
  const rvsdg::Output * BasePointer;
  std::optional<int64_t> Offset;
};

/**
 * Represents a collection of possible origins of a pointer value.
 */
struct LocalAliasAnalysis::TraceCollection
{
  /**
   * Contains all outputs visited while tracing, to avoid re-visiting.
   * If an output is visited first with a known offset, and later with a different offset,
   * the offset is collapsed to an unknown offset, and tracing continues with that.
   */
  std::unordered_map<const rvsdg::Output *, std::optional<int64_t>> AllTracedOutputs;

  /**
   * Contains the outputs that have been reached through tracing, which can not be traced further.
   * For example the output of an AllocaOperation, the result of a LoadNonVolatileOperation,
   * or the return value of a CallOperation.
   */
  std::unordered_map<const rvsdg::Output *, std::optional<int64_t>> TopOrigins;
};

AliasAnalysis::AliasQueryResponse
LocalAliasAnalysis::Query(const rvsdg::Output & p1, size_t s1, const rvsdg::Output & p2, size_t s2)
{
  const auto & p1Norm = llvm::traceOutput(p1);
  const auto & p2Norm = llvm::traceOutput(p2);

  // If the two pointers are the same value, they must alias
  if (&p1Norm == &p2Norm)
    return MustAlias;

  // Trace through GEP operations to get closer to the origins of the pointers
  // Only trace through GEPs where the offset is known at compile time,
  // to avoid giving up on MustAlias prematurely
  const auto p1Traced = TracePointerOriginPrecise(p1Norm);
  const auto p2Traced = TracePointerOriginPrecise(p2Norm);
  JLM_ASSERT(p1Traced.Offset.has_value() && p2Traced.Offset.has_value());

  if (p1Traced.BasePointer == p2Traced.BasePointer)
  {
    // The pointers share base, but may have different offsets
    // p1 = base + p1Offset
    // p2 = base + p2Offset

    return QueryOffsets(p1Traced.Offset, s1, p2Traced.Offset, s2);
  }

  // Keep tracing back to all sources
  TraceCollection p1TraceCollection;
  TraceCollection p2TraceCollection;

  // If tracing reaches too many possible outputs, it may give up
  if (!TraceAllPointerOrigins(p1Traced, p1TraceCollection))
    return MayAlias;

  if (!TraceAllPointerOrigins(p2Traced, p2TraceCollection))
    return MayAlias;

  // Removes top origins that can not possibly be valid targets due to being too small.
  // If p1 + s1 is outside the range of a top origin, then p1 can not target it
  RemoveTopOriginsWithRemainingSizeBelow(p1TraceCollection, s1);
  RemoveTopOriginsWithRemainingSizeBelow(p2TraceCollection, s2);

  // If each trace collection has only one top origin, check if they have the same base pointer
  if (p1TraceCollection.TopOrigins.size() == 1 && p2TraceCollection.TopOrigins.size() == 1)
  {
    const auto & [p1Base, p1Offset] = *p1TraceCollection.TopOrigins.begin();
    const auto & [p2Base, p2Offset] = *p2TraceCollection.TopOrigins.begin();
    if (p1Base == p2Base)
      return QueryOffsets(p1Offset, s1, p2Offset, s2);
  }

  // From this point on we give up on MustAlias

  // Since we only have inbound GEPs, a pointer p = b + 12 must point at least 12 bytes into
  // the memory region it points to
  auto minimumP1OffsetFromStart = GetMinimumOffsetFromStart(p1TraceCollection);
  auto minimumP2OffsetFromStart = GetMinimumOffsetFromStart(p2TraceCollection);

  // In case the trace collections contain unknown offsets, also try using the
  // precise p1Traced and p2Traced, which always have a known offset
  if (*p1Traced.Offset > 0)
    minimumP1OffsetFromStart =
        std::max(minimumP1OffsetFromStart, static_cast<size_t>(*p1Traced.Offset));
  if (*p2Traced.Offset > 0)
    minimumP2OffsetFromStart =
        std::max(minimumP2OffsetFromStart, static_cast<size_t>(*p2Traced.Offset));

  // Since we have given up on MustAlias, we can remove some targets even if they are valid.
  // Even if p1 might point to an 4-byte int foo,
  // if p2 is an 8-byte operation, or p2 is at least 4 bytes into its target,
  // we can safely discard that p1 might target foo.
  RemoveTopOriginsSmallerThanSize(p2TraceCollection, minimumP1OffsetFromStart + s1);
  RemoveTopOriginsSmallerThanSize(p1TraceCollection, minimumP2OffsetFromStart + s2);

  // If we know that p2 is at least 12 bytes into the memory region it targets,
  // then any use of p1 where p1 + s1 is within the first 12 bytes of its memory region can be
  // ignored.
  RemoveTopOriginsWithinTheFirstNBytes(p1TraceCollection, s1, minimumP2OffsetFromStart);
  RemoveTopOriginsWithinTheFirstNBytes(p2TraceCollection, s2, minimumP1OffsetFromStart);

  // Any direct overlap in the collections' top sets means there is a possibility of aliasing
  if (DoTraceCollectionsOverlap(p1TraceCollection, s1, p2TraceCollection, s2))
    return MayAlias;

  // Even if there is no direct overlap in the trace collections, the pointers may still alias
  // Take for example the top sets { ALLOCA[a]+40, ALLOCA[b] } and { ALLOCA[c], o4+20 }
  // o4 is some output that can not be traced further, but it is also not original.
  // It is possible for o4 to be a pointer to ALLOCA[a]+20, in which case there is aliasing.

  // We already know that there is no direct overlap in the top origin sets,
  // so if both trace collections only contain original pointers, there is NoAlias.
  const bool p1AllTopsOriginal = HasOnlyOriginalTopOrigins(p1TraceCollection);
  const bool p2AllTopsOriginal = HasOnlyOriginalTopOrigins(p2TraceCollection);

  if (p1AllTopsOriginal && p2AllTopsOriginal)
    return NoAlias;

  // If one of the pointers has a top origin set containing only fully traceable ALLOCAs,
  // it is not possible for the other pointer to target any of them,
  // as they would already be explicitly included in its top origin set.
  const bool p1OnlyTraceable = HasOnlyFullyTraceableTopOrigins(p1TraceCollection);
  const bool p2OnlyTraceable = HasOnlyFullyTraceableTopOrigins(p2TraceCollection);

  if (p1OnlyTraceable || p2OnlyTraceable)
    return NoAlias;

  return MayAlias;
}

/**
 * Calculates the byte offset inside the given type, starting at the given offset of GEP inputs.
 * Uses recursion to handle nested types.
 * If any indexing input is not a compile time constant, nullopt is returned.
 * @param gepNode the GEP node
 * @param inputIndex the index of the input that applies inside the given type
 * @param type the type the offset is inside
 * @return the byte offset within the given type, or nullopt if not possible.
 */
static std::optional<int64_t>
CalculateIntraTypeGepOffset(
    const rvsdg::SimpleNode & gepNode,
    size_t inputIndex,
    const rvsdg::Type & type)
{
  // If we have no more input index values, we are not offsetting into the type
  if (inputIndex >= gepNode.ninputs())
    return 0;

  // GEP input 0 is the pointer being offset
  // GEP input 1 is the number of whole types
  // Intra-type offsets start at input 2 and beyond
  JLM_ASSERT(inputIndex >= 2);

  auto & gepInput = *gepNode.input(inputIndex)->origin();
  auto indexingValue = tryGetConstantSignedInteger(gepInput);

  // Any unknown indexing value means the GEP offset is unknown overall
  if (!indexingValue.has_value())
    return std::nullopt;

  if (auto array = dynamic_cast<const ArrayType *>(&type))
  {
    const auto & elementType = array->GetElementType();
    int64_t offset = *indexingValue * GetTypeAllocSize(*elementType);

    // Get the offset into the element type as well, if any
    const auto subOffset = CalculateIntraTypeGepOffset(gepNode, inputIndex + 1, *elementType);
    if (subOffset.has_value())
      return offset + *subOffset;

    return std::nullopt;
  }
  if (auto strct = dynamic_cast<const StructType *>(&type))
  {
    if (*indexingValue < 0 || static_cast<size_t>(*indexingValue) >= strct->numElements())
      throw std::logic_error("Struct type has fewer fields than requested by GEP");

    const auto & fieldType = strct->getElementType(*indexingValue);
    int64_t offset = strct->GetFieldOffset(*indexingValue);

    const auto subOffset = CalculateIntraTypeGepOffset(gepNode, inputIndex + 1, *fieldType);
    if (subOffset.has_value())
      return offset + *subOffset;

    return std::nullopt;
  }

  JLM_UNREACHABLE("Unknown GEP type");
}

std::optional<int64_t>
LocalAliasAnalysis::CalculateGepOffset(const rvsdg::SimpleNode & gepNode)
{
  const auto gep = util::assertedCast<const GetElementPtrOperation>(&gepNode.GetOperation());

  // The pointee type. Gets updated by the loop below if the GEP has multiple levels of offsets
  const auto & pointeeType = gep->GetPointeeType();

  const auto & wholeTypeIndexingOrigin = *gepNode.input(1)->origin();
  const auto wholeTypeIndexing = tryGetConstantSignedInteger(wholeTypeIndexingOrigin);

  if (!wholeTypeIndexing.has_value())
    return std::nullopt;

  int64_t offset = *wholeTypeIndexing * GetTypeAllocSize(pointeeType);

  // In addition to offsetting by whole types, a GEP can also offset within a type
  const auto subOffset = CalculateIntraTypeGepOffset(gepNode, 2, pointeeType);
  if (!subOffset.has_value())
    return std::nullopt;

  return offset + *subOffset;
}

LocalAliasAnalysis::TracedPointerOrigin
LocalAliasAnalysis::TracePointerOriginPrecise(const rvsdg::Output & p)
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
      auto calculatedOffset = CalculateGepOffset(*node);

      // Only trace through GEPs with statically known offsets
      if (!calculatedOffset.has_value())
        break;

      base = node->input(0)->origin();
      offset += *calculatedOffset;
    }

    // We were not able to trace further
    break;
  }

  return TracedPointerOrigin{ base, offset };
}

AliasAnalysis::AliasQueryResponse
LocalAliasAnalysis::QueryOffsets(
    std::optional<int64_t> offset1,
    size_t s1,
    std::optional<int64_t> offset2,
    size_t s2)
{
  // If either offset is unknown, return MayAlias
  if (!offset1.has_value() || !offset2.has_value())
    return MayAlias;

  auto difference = *offset2 - *offset1;
  if (difference == 0)
    return MustAlias;

  // p2 starts at or after p1+s1
  if (difference >= 0 && static_cast<size_t>(difference) >= s1)
    return NoAlias;

  // p1 starts at or after p2+s2
  if (difference <= 0 && static_cast<size_t>(-difference) >= s2)
    return NoAlias;

  // We have a partial alias
  return MayAlias;
}

bool
LocalAliasAnalysis::TraceAllPointerOrigins(TracedPointerOrigin p, TraceCollection & traceCollection)
{
  if (traceCollection.AllTracedOutputs.size() >= MaxTraceCollectionSize)
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
      const auto gepOffset = CalculateGepOffset(*node);
      if (gepOffset.has_value())
        p.Offset = *p.Offset + *gepOffset;
      else
        p.Offset = std::nullopt;
    }

    return TraceAllPointerOrigins(p, traceCollection);
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

    return TraceAllPointerOrigins(leftTrace, traceCollection)
        && TraceAllPointerOrigins(rightTrace, traceCollection);
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
      if (!TraceAllPointerOrigins(inside, traceCollection))
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
    return TraceAllPointerOrigins(inside, traceCollection);
  }

  // We could not trace further, add p as a TopOrigin
  traceCollection.TopOrigins[p.BasePointer] = p.Offset;
  return true;
}

bool
LocalAliasAnalysis::IsOriginalOrigin(const rvsdg::Output & pointer)
{
  // Each GraphImport represents a unique object
  if (dynamic_cast<const GraphImport *>(&pointer))
    return true;

  if (rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(pointer))
    return true;

  if (rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(pointer))
    return true;

  // Is pointer the output of one of the nodes
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(pointer))
  {
    if (is<AllocaOperation>(node->GetOperation()))
      return true;

    if (is<MallocOperation>(node->GetOperation()))
      return true;
  }

  return false;
}

bool
LocalAliasAnalysis::HasOnlyOriginalTopOrigins(TraceCollection & traces)
{
  for (auto [output, offset] : traces.TopOrigins)
  {
    if (!IsOriginalOrigin(*output))
      return false;
  }
  return true;
}

std::optional<size_t>
LocalAliasAnalysis::GetOriginalOriginSize(const rvsdg::Output & pointer)
{
  if (auto delta = rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(pointer))
    return GetTypeAllocSize(*delta->GetOperation().Type());
  if (auto import = dynamic_cast<const GraphImport *>(&pointer))
  {
    auto size = GetTypeAllocSize(*import->ValueType());
    // Workaround for imported incomplete types appearing to have size 0 in the LLVM IR
    if (size == 0)
      return std::nullopt;

    return size;
  }
  if (const auto [node, allocaOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(pointer);
      allocaOp)
  {
    const auto elementCount = tryGetConstantSignedInteger(*node->input(0)->origin());
    if (elementCount.has_value())
      return *elementCount * GetTypeAllocSize(*allocaOp->ValueType());
  }
  if (const auto [node, mallocOp] = rvsdg::TryGetSimpleNodeAndOptionalOp<MallocOperation>(pointer);
      mallocOp)
  {
    const auto mallocSize =
        tryGetConstantSignedInteger(*MallocOperation::sizeInput(*node).origin());
    if (mallocSize.has_value())
      return *mallocSize;
  }

  return std::nullopt;
}

std::optional<size_t>
LocalAliasAnalysis::GetRemainingSize(TracedPointerOrigin trace)
{
  const auto totalSize = GetOriginalOriginSize(*trace.BasePointer);
  if (!totalSize.has_value())
    return std::nullopt;

  if (!trace.Offset.has_value())
    return *totalSize;

  // Avoid wrap-around by truncating remaining size to min 0
  if (*trace.Offset > 0 && static_cast<size_t>(*trace.Offset) > *totalSize)
    return 0;

  return *totalSize - *trace.Offset;
}

void
LocalAliasAnalysis::RemoveTopOriginsWithRemainingSizeBelow(TraceCollection & traces, size_t s)
{
  auto it = traces.TopOrigins.begin();
  while (it != traces.TopOrigins.end())
  {
    const auto remainingSize = GetRemainingSize({ it->first, it->second });
    if (remainingSize.has_value())
    {
      // This top origin leaves too little room, and can be fully removed
      if (*remainingSize < s)
      {
        it = traces.TopOrigins.erase(it);
        continue;
      }

      // If a top origin is exactly large enough for s, any unknown offset must be 0
      if (*remainingSize == s && !it->second.has_value())
        it->second = 0;
    }
    it++;
  }
}

size_t
LocalAliasAnalysis::GetMinimumOffsetFromStart(TraceCollection & traces)
{
  std::optional<size_t> minimumOffset;
  for (auto [output, offset] : traces.TopOrigins)
  {
    // If one of the possible targets has an unknown offset, just use the access size
    if (!offset.has_value())
      return 0;

    if (*offset < 0)
      return 0;

    if (minimumOffset.has_value())
      minimumOffset = std::min(*minimumOffset, static_cast<size_t>(*offset));
    else
      minimumOffset = offset;
  }

  if (minimumOffset.has_value())
    return *minimumOffset;

  // We only get here if the top origins is empty, in which case the return value
  // does not matter, as the query will return NoAlias anyway.
  return 0;
}

void
LocalAliasAnalysis::RemoveTopOriginsSmallerThanSize(TraceCollection & traces, size_t s)
{
  auto it = traces.TopOrigins.begin();
  while (it != traces.TopOrigins.end())
  {
    auto originSize = GetOriginalOriginSize(*it->first);
    if (originSize.has_value() && *originSize < s)
      it = traces.TopOrigins.erase(it);
    else
      it++;
  }
}

void
LocalAliasAnalysis::RemoveTopOriginsWithinTheFirstNBytes(
    TraceCollection & traces,
    size_t s,
    size_t N)
{
  auto it = traces.TopOrigins.begin();
  while (it != traces.TopOrigins.end())
  {
    const auto offset = it->second;

    // If the pointer is original, it is also pointing to the beginning of the memory region.
    // The offset thus tells us exactly which bytes within the memory region we touch.
    if (IsOriginalOrigin(*it->first) && offset.has_value() && *offset + s <= N)
      it = traces.TopOrigins.erase(it);
    else
      it++;
  }
}

bool
LocalAliasAnalysis::DoTraceCollectionsOverlap(
    TraceCollection & tc1,
    size_t s1,
    TraceCollection & tc2,
    size_t s2)
{
  for (auto [p1Origin, p1Offset] : tc1.TopOrigins)
  {
    auto p2Find = tc2.TopOrigins.find(p1Origin);
    if (p2Find == tc2.TopOrigins.end())
      continue;

    auto p2Offset = p2Find->second;
    if (QueryOffsets(p1Offset, s1, p2Offset, s2) != NoAlias)
      return true;
  }

  return false;
}

bool
LocalAliasAnalysis::IsOriginalOriginFullyTraceable(const rvsdg::Output & pointer)
{
  // The only original origins that can be fully traced for escaping are ALLOCAs
  if (!rvsdg::IsOwnerNodeOperation<AllocaOperation>(pointer))
    return false;

  // Check if the result for this ALLOCA is already memoized
  auto it = IsFullyTraceable_.find(&pointer);
  if (it != IsFullyTraceable_.end())
    return it->second;

  // Use a queue to find all users of the ALLOCA's address
  std::queue<const rvsdg::Output *> qu;
  std::unordered_set<const rvsdg::Output *> added;

  const auto Enqueue = [&](const rvsdg::Output & p)
  {
    // Only enqueue new outputs
    auto [_, inserted] = added.insert(&p);
    if (inserted)
      qu.push(&p);
  };

  Enqueue(pointer);
  while (!qu.empty())
  {
    auto & p = *qu.front();
    qu.pop();

    // Handle all inputs that are users of p
    for (auto & user : p.Users())
    {
      if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(user))
      {
        auto input = gamma->MapInput(user);

        // A pointer must always be an EntryVar, as the MatchVar has a ControlType
        auto entry = std::get_if<rvsdg::GammaNode::EntryVar>(&input);
        JLM_ASSERT(entry);

        for (auto output : entry->branchArgument)
          Enqueue(*output);

        continue;
      }
      if (auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(user))
      {
        // user is a gamma result, find the corresponding gamma output
        auto exitVar = gamma->MapBranchResultExitVar(user);
        Enqueue(*exitVar.output);

        continue;
      }

      if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(user))
      {
        auto loopVar = theta->MapInputLoopVar(user);

        // The loop always runs at least once, so map it to the inside
        Enqueue(*loopVar.pre);

        continue;
      }
      if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(user))
      {
        // user is a theta result, find the corresponding loop variable
        auto loopVar = theta->MapPostLoopVar(user);
        Enqueue(*loopVar.pre);
        Enqueue(*loopVar.output);

        continue;
      }

      if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(user))
      {
        bool do_continue = MatchTypeWithDefault(
            node->GetOperation(),
            [&](const IOBarrierOperation &)
            {
              // The pointer input must be the node's first input
              JLM_ASSERT(user.index() == 0);
              Enqueue(*node->output(0));
              return true;
            },
            [&](const GetElementPtrOperation &)
            {
              // The pointer input must be the node's first input
              JLM_ASSERT(user.index() == 0);
              Enqueue(*node->output(0));
              return true;
            },
            [&](const SelectOperation &)
            {
              // Select operations are fine, if the output is still fully traceable
              Enqueue(*node->output(0));
              return true;
            },
            [&](const LoadOperation &)
            {
              // Loads are always fine
              return true;
            },
            [&](const StoreOperation &)
            {
              // Stores are only fine if the pointer itself is not being stored somewhere
              if (&user == &StoreOperation::AddressInput(*node))
                return true;
              else
                return false;
            },
            []()
            {
              return false;
            });
        if (do_continue)
          continue;
      }

      // We were unable to handle this user, so the original pointer escapes tracing
      IsFullyTraceable_[&pointer] = false;
      return false;
    }
  }

  // The entire queue was processed without reaching a single untraceable user of the pointer
  IsFullyTraceable_[&pointer] = true;
  return true;
}

bool
LocalAliasAnalysis::HasOnlyFullyTraceableTopOrigins(TraceCollection & traces)
{
  for (auto [topOrigin, _] : traces.TopOrigins)
  {
    if (!IsOriginalOriginFullyTraceable(*topOrigin))
      return false;
  }

  return true;
}

}
