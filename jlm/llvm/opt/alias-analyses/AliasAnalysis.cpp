/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "MemoryStateEncoder.hpp"
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/lambda.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/util/Math.hpp>

#include <numeric>
#include <queue>

namespace jlm::llvm::aa
{

AliasAnalysis::AliasAnalysis() = default;

AliasAnalysis::~AliasAnalysis() = default;

PointsToGraphAliasAnalysis::PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph)
    : PointsToGraph_(pointsToGraph)
{}

PointsToGraphAliasAnalysis::~PointsToGraphAliasAnalysis() = default;

std::string
PointsToGraphAliasAnalysis::ToString() const
{
  return "PointsToGraphAA";
}

AliasAnalysis::AliasQueryResponse
PointsToGraphAliasAnalysis::Query(
    const rvsdg::Output & p1,
    const size_t s1,
    const rvsdg::Output & p2,
    const size_t s2)
{
  // If the two pointers are the same value, they must alias
  if (&p1 == &p2)
    return MustAlias;

  // Assume that all pointers actually exist in the PointsToGraph
  auto & p1RegisterNode = PointsToGraph_.GetRegisterNode(p1);
  auto & p2RegisterNode = PointsToGraph_.GetRegisterNode(p2);

  // Check if both pointers may target the external node, to avoid iterating over large sets
  const auto & externalNode = PointsToGraph_.GetExternalMemoryNode();
  if (p1RegisterNode.HasTarget(externalNode) && p2RegisterNode.HasTarget(externalNode))
    return MayAlias;

  // Quickly checks if the given register node has only one possible target
  const auto GetSingleTarget = [&](const PointsToGraph::RegisterNode & node,
                                   size_t size) -> std::optional<const PointsToGraph::MemoryNode *>
  {
    std::optional<const PointsToGraph::MemoryNode *> singleTarget;
    for (auto & target : node.Targets())
    {
      // Skip memory locations that are too small to hold size
      const auto targetSize = GetMemoryNodeSize(target);
      if (targetSize.has_value() && *targetSize < size)
        continue;

      if (singleTarget.has_value())
        return std::nullopt;

      singleTarget = &target;
    }

    return singleTarget;
  };

  // If both p1 and p2 have exactly one possible target, which is the same for both,
  // and is the same size as the operations, and only represents one concrete memory location,
  // we can respond with MustAlias
  if (s1 == s2)
  {
    const auto p1SingleTarget = GetSingleTarget(p1RegisterNode, s1);
    const auto p2SingleTarget = GetSingleTarget(p2RegisterNode, s2);
    if (p1SingleTarget.has_value() && p2SingleTarget.has_value())
    {
      if (*p1SingleTarget == *p2SingleTarget)
      {
        if (IsRepresentingSingleMemoryLocation(**p1SingleTarget))
        {
          const auto targetSize = GetMemoryNodeSize(**p1SingleTarget);
          if (targetSize.has_value() && *targetSize == s1)
            return MustAlias;
        }
      }
    }
  }

  // For a memory location to be the target of both p1 and p2, it needs to be large enough
  // to represent both [p1, p1+s1) and [p2, p2+s2)
  const auto neededSize = std::max(s1, s2);

  for (auto & target : p1RegisterNode.Targets())
  {
    // Skip memory locations that are too small
    const auto targetSize = GetMemoryNodeSize(target);
    if (targetSize.has_value() && *targetSize < neededSize)
      continue;

    if (p2RegisterNode.HasTarget(target))
      return MayAlias;
  }

  return NoAlias;
}

std::optional<size_t>
PointsToGraphAliasAnalysis::GetMemoryNodeSize(const PointsToGraph::MemoryNode & node)
{
  if (auto delta = dynamic_cast<const PointsToGraph::DeltaNode *>(&node))
    return GetLlvmTypeSize(*delta->GetDeltaNode().GetOperation().Type());
  if (auto import = dynamic_cast<const PointsToGraph::ImportNode *>(&node))
  {
    auto size = GetLlvmTypeSize(*import->GetArgument().ValueType());
    // Workaround for imported incomplete types appearing to have size 0 in the LLVM IR
    if (size == 0)
      return std::nullopt;
  }
  if (auto alloca = dynamic_cast<const PointsToGraph::AllocaNode *>(&node))
  {
    const auto & allocaNode = alloca->GetAllocaNode();
    const auto allocaOp = util::AssertedCast<const alloca_op>(&allocaNode.GetOperation());

    // An alloca has a count parameter, which on rare occasions is not just the constant 1.
    const auto elementCount = GetConstantIntegerValue(*allocaNode.input(0)->origin());
    if (elementCount.has_value())
      return *elementCount * GetLlvmTypeSize(*allocaOp->ValueType());
  }
  if (auto malloc = dynamic_cast<const PointsToGraph::MallocNode *>(&node))
  {
    const auto & mallocNode = malloc->GetMallocNode();

    const auto mallocSize = GetConstantIntegerValue(*mallocNode.input(0)->origin());
    if (mallocSize.has_value())
      return *mallocSize;
  }

  return std::nullopt;
}

bool
PointsToGraphAliasAnalysis::IsRepresentingSingleMemoryLocation(
    const PointsToGraph::MemoryNode & node)
{
  return PointsToGraph::Node::Is<PointsToGraph::DeltaNode>(node)
      || PointsToGraph::Node::Is<PointsToGraph::ImportNode>(node)
      || PointsToGraph::Node::Is<PointsToGraph::LambdaNode>(node);
}

ChainedAliasAnalysis::ChainedAliasAnalysis(AliasAnalysis & first, AliasAnalysis & second)
    : First_(first),
      Second_(second)
{}

ChainedAliasAnalysis::~ChainedAliasAnalysis() = default;

AliasAnalysis::AliasQueryResponse
ChainedAliasAnalysis::Query(
    const rvsdg::Output & p1,
    size_t s1,
    const rvsdg::Output & p2,
    size_t s2)
{
  const auto firstResponse = First_.Query(p1, s1, p2, s2);

  // Anything other than MayAlias is precise, and can be returned right away
  if (firstResponse != MayAlias)
  {
    [[maybe_unused]] AliasQueryResponse opposite = firstResponse == MustAlias ? NoAlias : MustAlias;
    JLM_ASSERT(Second_.Query(p1, s1, p2, s2) != opposite);
    return firstResponse;
  }

  return Second_.Query(p1, s1, p2, s2);
}

std::string
ChainedAliasAnalysis::ToString() const
{
  return util::strfmt("ChainedAA(", First_.ToString(), ",", Second_.ToString(), ")");
}

LocalAliasAnalysis::LocalAliasAnalysis()
{}

LocalAliasAnalysis::~LocalAliasAnalysis() = default;

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
 *  p = base pointer + <unknown>
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
   * For example the output of an ALLOCA, the result of a LOAD, or the return value of a function.
   */
  std::unordered_map<const rvsdg::Output *, std::optional<int64_t>> TopOrigins;
};

AliasAnalysis::AliasQueryResponse
LocalAliasAnalysis::Query(
    const rvsdg::Output & p1,
    size_t s1,
    const rvsdg::Output & p2,
    size_t s2)
{
  const auto & p1Norm = NormalizePointerValue(p1);
  const auto & p2Norm = NormalizePointerValue(p2);

  // If the two pointers are the same value, they must alias
  if (&p1Norm == &p2Norm)
    return MustAlias;

  // Trace through GEP operations to get closer to the origins of the pointers
  // Only traces through GEPs where the offset is known at compile time,
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
    const rvsdg::ValueType & type)
{
  // If we have no more input index values, we are not offsetting into the type
  if (inputIndex >= gepNode.ninputs())
    return 0;

  // GEP input 0 is the pointer being offset
  // GEP input 1 is the number of whole types
  // Intra-type offsets start at input 2 and beyond
  JLM_ASSERT(inputIndex >= 2);

  auto & gepInput = *gepNode.input(inputIndex)->origin();
  auto indexingValue = GetConstantIntegerValue(gepInput);

  // Any unknown indexing value means the GEP offset is unknown overall
  if (!indexingValue.has_value())
    return std::nullopt;

  if (auto array = dynamic_cast<const ArrayType *>(&type))
  {
    const auto & elementType = array->GetElementType();
    int64_t offset = *indexingValue * GetLlvmTypeSize(*elementType);

    // Get the offset into the element type as well, if any
    const auto subOffset = CalculateIntraTypeGepOffset(gepNode, inputIndex + 1, *elementType);
    if (subOffset.has_value())
      return offset + *subOffset;

    return std::nullopt;
  }
  if (auto strct = dynamic_cast<const StructType *>(&type))
  {
    if (*indexingValue < 0
        || static_cast<size_t>(*indexingValue) >= strct->GetDeclaration().NumElements())
      throw std::logic_error("Struct type has fewer fields than requested by GEP");

    const auto & fieldType = strct->GetDeclaration().GetElement(*indexingValue);
    int64_t offset = GetStructFieldOffset(*strct, *indexingValue);

    const auto subOffset = CalculateIntraTypeGepOffset(gepNode, inputIndex + 1, fieldType);
    if (subOffset.has_value())
      return offset + *subOffset;

    return std::nullopt;
  }

  JLM_UNREACHABLE("Unknown GEP type");
}

std::optional<int64_t>
LocalAliasAnalysis::CalculateGepOffset(const rvsdg::SimpleNode & gepNode)
{
  const auto gep = util::AssertedCast<const GetElementPtrOperation>(&gepNode.GetOperation());

  // The pointee type. Gets updated by the loop below if the GEP has multiple levels of offsets
  const auto & pointeeType = gep->GetPointeeType();

  const auto & wholeTypeIndexingOrigin = *gepNode.input(1)->origin();
  const auto wholeTypeIndexing = GetConstantIntegerValue(wholeTypeIndexingOrigin);

  if (!wholeTypeIndexing.has_value())
    return std::nullopt;

  int64_t offset = *wholeTypeIndexing * GetLlvmTypeSize(pointeeType);

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
    base = &NormalizePointerValue(*base);

    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*base))
    {
      if (is<GetElementPtrOperation>(node->GetOperation()))
      {
        auto calculatedOffset = CalculateGepOffset(*node);

        // Only trace through GEPs with statically known offsets
        if (!calculatedOffset.has_value())
          break;

        base = node->input(0)->origin();
        offset += *calculatedOffset;
      }
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
  p.BasePointer = &NormalizePointerValue(*p.BasePointer);

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

  if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*p.BasePointer))
  {
    // If it is a GEP, we can trace through it, but possibly lose precise offset information
    if (is<GetElementPtrOperation>(node))
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

    // If we reach undef nodes, do not include them in the TopOrigins
    if (is<UndefValueOperation>(node))
    {
      return true;
    }
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

  if (rvsdg::TryGetOwnerNode<delta::node>(pointer))
    return true;

  if (rvsdg::TryGetOwnerNode<rvsdg::LambdaNode>(pointer))
    return true;

  // Is pointer the output of one of the nodes
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(pointer))
  {
    if (is<alloca_op>(node))
      return true;

    if (is<malloc_op>(node))
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
  if (auto delta = rvsdg::TryGetOwnerNode<delta::node>(pointer))
    return GetLlvmTypeSize(*delta->GetOperation().Type());
  if (auto import = dynamic_cast<const GraphImport *>(&pointer))
  {
    auto size = GetLlvmTypeSize(*import->ValueType());
    // Workaround for imported incomplete types appearing to have size 0 in the LLVM IR
    if (size == 0)
      return std::nullopt;
  }
  if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(pointer))
  {
    if (auto allocaOp = dynamic_cast<const alloca_op *>(&node->GetOperation()))
    {
      const auto elementCount = GetConstantIntegerValue(*node->input(0)->origin());
      if (elementCount.has_value())
        return *elementCount * GetLlvmTypeSize(*allocaOp->ValueType());
    }
    if (is<malloc_op>(node))
    {
      const auto mallocSize = GetConstantIntegerValue(*node->input(0)->origin());
      if (mallocSize.has_value())
        return *mallocSize;
    }
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
    if (remainingSize.has_value() && *remainingSize < s)
      it = traces.TopOrigins.erase(it);
    else
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
  const auto originalNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(pointer);
  if (!is<alloca_op>(originalNode))
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
    for (auto user : p)
    {
      if (auto gamma = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*user))
      {
        auto input = gamma->MapInput(*user);

        // A pointer must always be an EntryVar, as the MatchVar has a ControlType
        auto entry = std::get_if<rvsdg::GammaNode::EntryVar>(&input);
        JLM_ASSERT(entry);

        for (auto output : entry->branchArgument)
          Enqueue(*output);

        continue;
      }
      if (auto gamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*user))
      {
        // user is a gamma result, find the corresponding gamma output
        auto exitVar = gamma->MapBranchResultExitVar(*user);
        Enqueue(*exitVar.output);

        continue;
      }

      if (auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*user))
      {
        auto loopVar = theta->MapInputLoopVar(*user);

        // The loop always runs at least once, so map it to the inside
        Enqueue(*loopVar.pre);

        continue;
      }
      if (auto theta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*user))
      {
        // user is a theta result, find the corresponding loop variable
        auto loopVar = theta->MapPostLoopVar(*user);
        Enqueue(*loopVar.pre);
        Enqueue(*loopVar.output);

        continue;
      }

      if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user))
      {
        // Pointers go straight through IO barriers and GEPs
        if (is<IOBarrierOperation>(node) || is<GetElementPtrOperation>(node))
        {
          // The pointer input must be the node's first input
          JLM_ASSERT(user->index() == 0);
          Enqueue(*node->output(0));
          continue;
        }

        // Loads are always fine
        if (is<LoadOperation>(node))
          continue;

        // Stores are only fine if the pointer itself is not being stored somewhere
        if (is<StoreOperation>(node))
        {
          if (user == &StoreOperation::AddressInput(*node))
            continue;
        }
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

bool
IsPointerCompatible(const rvsdg::Output & value)
{
  return IsOrContains<PointerType>(*value.Type());
}

const rvsdg::Output &
NormalizeOutput(const rvsdg::Output & output)
{
  if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(output))
  {
    if (is<IOBarrierOperation>(node))
    {
      return NormalizeOutput(*node->input(0)->origin());
    }
  }
  else if (rvsdg::TryGetOwnerNode<rvsdg::StructuralNode>(output))
  {
    // If the output is a phi recursion variable, continue tracing inside the phi
    if (const auto phiNode = rvsdg::TryGetOwnerNode<rvsdg::PhiNode>(output))
    {
      const auto fixVar = phiNode->MapOutputFixVar(output);
      return NormalizeOutput(*fixVar.result->origin());
    }

    // If the output is a theta output, check if it is invariant
    if (const auto theta = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(output))
    {
      const auto loopVar = theta->MapOutputLoopVar(output);
      if (rvsdg::ThetaLoopVarIsInvariant(loopVar))
        return NormalizeOutput(*loopVar.input->origin());
    }
  }
  else if (const auto outerGamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(output))
  {
    // Follow the gamma input
    auto input = outerGamma->MapBranchArgument(output);
    std::get_if<rvsdg::GammaNode::EntryVar>(&input);

    if (const auto entryVar = std::get_if<rvsdg::GammaNode::EntryVar>(&input))
    {
      return NormalizeOutput(*entryVar->input->origin());
    }
  }
  else if (const auto outerTheta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(output))
  {
    const auto loopVar = outerTheta->GetLoopVars()[output.index()];

    // If the loop variable is invariant, continue normalizing
    if (ThetaLoopVarIsInvariant(loopVar))
      return NormalizeOutput(*loopVar.input->origin());
  }
  else if (const auto outerLambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(output))
  {
    const auto ctxVar = outerLambda->MapBinderContextVar(output);

    // If the argument is a contex variable, continue normalizing
    if (ctxVar)
      return NormalizeOutput(*ctxVar->input->origin());
  }
  else if (const auto phiNode = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(output))
  {
    const auto argument = phiNode->MapArgument(output);
    if (const auto cvArg = std::get_if<rvsdg::PhiNode::ContextVar>(&argument))
    {
      // Follow the context variable to outside the phi
      return NormalizeOutput(*cvArg->input->origin());
    }
    if (const auto fixArg = std::get_if<rvsdg::PhiNode::FixVar>(&argument))
    {
      // Follow to the recursion variable's definition
      return NormalizeOutput(*fixArg->result->origin());
    }

    JLM_UNREACHABLE("Unknown phi argument type");
  }

  return output;
}

const rvsdg::Output &
NormalizePointerValue(const rvsdg::Output & pointer)
{
  JLM_ASSERT(IsPointerCompatible(pointer));
  return NormalizeOutput(pointer);
}

std::optional<int64_t>
GetConstantIntegerValue(const rvsdg::Output & output)
{
  const auto & normalized = NormalizeOutput(output);
  if (auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(normalized))
  {
    if (auto constant = dynamic_cast<const IntegerConstantOperation *>(&node->GetOperation()))
    {
      return constant->Representation().to_int();
    }
  }

  return std::nullopt;
}

/**
 * @return the given \p size, rounded up to be a multiple of the given \p alignment
 */
size_t
RoundUpToMultipleOf(size_t size, size_t alignment)
{
  const auto miss = size % alignment;
  if (miss == 0)
    return size;
  return size + alignment - miss;
}

size_t
GetLlvmTypeSize(const rvsdg::ValueType & type)
{
  if (const auto bits = dynamic_cast<const rvsdg::bittype *>(&type))
  {
    // Assume 8 bits per byte, and round up to a power of 2 bytes
    const auto bytes = (bits->nbits() + 7) / 8;
    return util::RoundUpToPowerOf2(bytes);
  }
  if (is<PointerType>(type))
  {
    // Assume 64-bit pointers
    return 8;
  }
  if (const auto arrayType = dynamic_cast<const ArrayType *>(&type))
  {
    return arrayType->nelements() * GetLlvmTypeSize(*arrayType->GetElementType());
  }
  if (const auto floatType = dynamic_cast<const FloatingPointType *>(&type))
  {
    switch (floatType->size())
    {
    case fpsize::half:
      return 2;
    case fpsize::flt:
      return 4;
    case fpsize::dbl:
      return 8;
    case fpsize::fp128:
      return 16;
    case fpsize::x86fp80:
      return 16; // Will never actually be written to memory, but we round up
    default:
      JLM_UNREACHABLE("Unknown float size");
    }
  }
  if (const auto structType = dynamic_cast<const StructType *>(&type))
  {
    size_t totalSize = 0;
    size_t alignment = 1;

    const auto & decl = structType->GetDeclaration();
    // A packed struct has alignment 1, and all fields are tightly packed
    const auto isPacked = structType->IsPacked();

    for (size_t i = 0; i < decl.NumElements(); i++)
    {
      auto & field = decl.GetElement(i);
      auto fieldSize = GetLlvmTypeSize(field);
      auto fieldAlignment = isPacked ? 1 : GetLlvmTypeAlignment(field);

      // Add the size of the field, including any needed padding
      totalSize = RoundUpToMultipleOf(totalSize, fieldAlignment);
      totalSize += fieldSize;

      // The struct as a whole must be at least as aligned as each field
      alignment = std::lcm(alignment, fieldAlignment);
    }

    // Round size up to a multiple of alignment
    totalSize = RoundUpToMultipleOf(totalSize, alignment);

    // TODO: We assume C and allow empty structs. In C++, the size of an empty struct is 1 byte.

    return totalSize;
  }
  if (const auto vectorType = dynamic_cast<const VectorType *>(&type))
  {
    return vectorType->size() * GetLlvmTypeSize(*vectorType->Type());
  }
  if (is<rvsdg::FunctionType>(type))
  {
    // We should never read from or write to functions, so give the size 0
    return 0;
  }

  std::cerr << "unknown type: " << typeid(type).name() << std::endl;
  JLM_UNREACHABLE("Unknown type");
}

size_t
GetLlvmTypeAlignment(const rvsdg::ValueType & type)
{
  if (is<rvsdg::bittype>(type))
  {
    return GetLlvmTypeSize(type);
  }
  if (is<PointerType>(type) || is<FloatingPointType>(type))
  {
    return GetLlvmTypeSize(type);
  }
  if (const auto arrayType = dynamic_cast<const ArrayType *>(&type))
  {
    return GetLlvmTypeAlignment(*arrayType->GetElementType());
  }
  if (const auto structType = dynamic_cast<const StructType *>(&type))
  {
    const auto & decl = structType->GetDeclaration();
    // A packed struct has alignment 1, and all fields are tightly packed
    if (structType->IsPacked())
      return 1;

    size_t alignment = 1;

    for (size_t i = 0; i < decl.NumElements(); i++)
    {
      auto & field = decl.GetElement(i);
      auto fieldAlignment = GetLlvmTypeAlignment(field);

      // The struct as a whole must be at least as aligned as each field
      alignment = std::lcm(alignment, fieldAlignment);
    }

    return alignment;
  }
  if (const auto vectorType = dynamic_cast<const VectorType *>(&type))
  {
    return GetLlvmTypeAlignment(*vectorType->Type());
  }

  JLM_UNREACHABLE("Unknown type");
}

size_t
GetStructFieldOffset(const StructType & structType, size_t fieldIndex)
{
  const auto & decl = structType.GetDeclaration();
  const auto isPacked = structType.IsPacked();

  size_t offset = 0;

  for (size_t i = 0; i < decl.NumElements(); i++)
  {
    auto & field = decl.GetElement(i);

    // First round up to the alignment of the field
    auto fieldAlignment = isPacked ? 1 : GetLlvmTypeAlignment(field);
    offset = RoundUpToMultipleOf(offset, fieldAlignment);

    if (i == fieldIndex)
      return offset;

    // Add the size of the field
    offset += GetLlvmTypeSize(field);
  }

  JLM_UNREACHABLE("Invalid fieldIndex in GetStructFieldOffset");
}

}
