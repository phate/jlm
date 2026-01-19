/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_LOCALALIASANALYSIS_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_LOCALALIASANALYSIS_HPP

#include <cstddef>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/simple-node.hpp>

#include <optional>
#include <unordered_map>

namespace jlm::llvm::aa
{
/**
 * Class for making alias analysis queries using stateless ad hoc IR traversal.
 * It roughly corresponds to BasicAA from LLVM.
 * It is unable to trace via memory or across function calls.
 * It tries to keep track of pointer offsets when possible,
 * and can respond NoAlias when the queried pointers are based on distinct offsets
 * into the same base pointer. If the offsets are identical, MustAlias is returned.
 */
class LocalAliasAnalysis final : public AliasAnalysis
{

public:
  LocalAliasAnalysis();

  ~LocalAliasAnalysis() noexcept override;

  /**
   * Gets the maximum trace collection size before the analysis gives up with "MayAlias".
   * If set <= 1, tracing stops as soon as a value has multiple origins, or unknown offset.
   * @return the maximum number of outputs to trace before giving up on a pointer.
   */
  size_t
  getMaxTraceCollectionSize();

  /**
   * Sets the maximum number of outputs a pointer can be traced to before giving up with "MayAlias"
   * If set <= 1, tracing stops as soon as a value has multiple origins, or unknown offset.
   * @param maxTraceCollectionSize the number of outputs
   */
  void
  setMaxTraceCollectionSize(size_t maxTraceCollectionSize);

  std::string
  ToString() const override;

  AliasQueryResponse
  Query(const rvsdg::Output & p1, size_t s1, const rvsdg::Output & p2, size_t s2) override;

private:
  struct TracedPointerOrigin;
  struct TraceCollection;

  /**
   * Calculates the byte offset applied by the GEP, if the offset is static.
   * The offset is the number of bytes needed to satisfy
   *   output ptr = input ptr + offset in bytes
   *
   * @param gepNode the node representing the \ref GetElementPtrOperation
   * @return the offset applied by the GEP, if it is possible to determine at compile time
   */
  [[nodiscard]] static std::optional<int64_t>
  CalculateGepOffset(const rvsdg::SimpleNode & gepNode);

  /**
   * Returns the result of tracing the origin of p as far as possible, without tracing through
   * operations that add an unknown offsets, or add multiple possible origins.
   * Since the pointer has exactly one possible origin with a statically known offset,
   * it may be possible to give MustAlias responses.
   * If not, more extensive tracing can be performed using \ref TraceAllPointerOrigins.
   *
   * @param p the pointer value to trace
   * @return the TracedPointer for p, which is guaranteed to have a defined offset
   */
  [[nodiscard]] static TracedPointerOrigin
  TracePointerOriginPrecise(const rvsdg::Output & p);

  /**
   * Given two pointers with the same base pointer
   *  p1 = base + offset1
   *  p2 = base + offset2
   * Answers if the regions [p1, p1 + s1) and [p2, p2 + s2) may alias.
   * If p1 == p2, MustAlias is returned.
   * @param offset1 the offset of the first pointer. If it is nullopt, the offset is unknown.
   * @param s1 the size of the first memory region
   * @param offset2 the offset of the second pointer. If it is nullopt, the offset is unknown.
   * @param s2 the size of the second memory region
   */
  [[nodiscard]] static AliasQueryResponse
  QueryOffsets(
      std::optional<int64_t> offset1,
      size_t s1,
      std::optional<int64_t> offset2,
      size_t s2);

  /**
   * Traces to find all possible origins of the given pointer.
   * Traces through \ref GetElementPtrOperation, including those with offsets that are not known
   * at compile time. Also traces through gamma and theta nodes, building a set of multiple
   * possibilities. Tracing stops at "top origins", for example an \ref AllocaOperation, a \ref
   * LoadNonVolatileOperation, the return value of a \ref CallOperation etc.
   *
   * @param p the pointer to trace from
   * @param traceCollection the collection of trace points being created
   * @return false if the trace collection reached its maximum allowed size, and tracing aborted
   */
  [[nodiscard]] bool
  TraceAllPointerOrigins(TracedPointerOrigin p, TraceCollection & traceCollection);

  /**
   * Checks if the given pointer is the direct result of a memory location defining operation.
   * These operations are guaranteed to output pointers that do not alias any pointer,
   * except for those that are based on the original pointer itself.
   * The pointer is also guaranteed to be at the very beginning of the memory region.
   *
   * For example, the output of an \ref AllocaOperation, a \ref DeltaOperation,
   * or a \ref GraphImport, are such original origins.
   *
   * @param pointer the pointer value to check
   * @return true if the pointer is the original pointer to a memory location
   */
  [[nodiscard]] static bool
  IsOriginalOrigin(const rvsdg::Output & pointer);

  /**
   * Checks if all top origins in the trace collection are original.
   *
   * @param traces the trace collection
   * @return true if all top origins are original, false otherwise
   */
  [[nodiscard]] static bool
  HasOnlyOriginalTopOrigins(TraceCollection & traces);

  /**
   * Gets the size of the memory location(s) defined at the given output's operation.
   * If the output is not an original origin, or the size is unknown, nullopt is returned.
   * @param pointer a pointer output, should be from a memory location defining operation
   * @return the size of the defined memory location, or nullopt if it is unknown
   */
  [[nodiscard]] static std::optional<size_t>
  GetOriginalOriginSize(const rvsdg::Output & pointer);

  /**
   * Given a traced pointer origin like p, where
   *  b = alloca[3 x i32]
   *  p = b + 8
   *
   * we know that an operation starting at p can have a maximum size of 4 bytes.
   * This function attempts to calculate this size, known as the pointer's remaining size.
   *
   * If the offset is larger than the size of the target, the size 0 is returned.
   * If the offset is unknown, the size of the target is returned.
   *
   * @param trace the traced pointer
   * @return the number of bytes left after the given traced pointer, or nullopt if unknown.
   */
  [[nodiscard]] static std::optional<size_t>
  GetRemainingSize(TracedPointerOrigin trace);

  /**
   * For each top origin in the given trace collection, it is removed if it is deemed too small.
   * Let p be the traced pointer, and let the TraceCollection contain p = b + o,
   * where b is an original origin with a statically known size of S bytes.
   *
   * We know that the operation being performed at p has a length of s bytes.
   * If o + s > S, the operation would exceed the bounds of b.
   * The traced origin p = b + o can therefore be removed as an impossibility.
   *
   * Further, if we have a trace p = c + unknown offset,
   * where c is an original origin whose size is exactly equal to s,
   * then we replace the unknown offset with a 0, as that is the only possible offset
   * that leaves s bytes remaining in the trace.
   *
   * @param traces the trace collection
   * @param s the size of the operation being performed at the traced pointer
   */
  static void
  RemoveTopOriginsWithRemainingSizeBelow(TraceCollection & traces, size_t s);

  /**
   * Finds the minimum distance into some memory region the traced pointer is pointing.
   * For example, if the trace collection only contains the trace
   *  p = b + 12
   * we know that any operation on p will not touch the first 12 bytes of whatever region b is in.
   * The region must also be at least 12 + s bytes large.
   * @param traces
   * @return the minimum offset among all traces in the collection, or 0 if some are unknown
   */
  [[nodiscard]] static size_t
  GetMinimumOffsetFromStart(TraceCollection & traces);

  /**
   * For each top origin in the given trace collection, it is removed if it is too small.
   * This function considers the total size of the target, and ignores the offset.
   * @param traces the trace collection
   * @param s the minimum size of remaining top origins
   */
  static void
  RemoveTopOriginsSmallerThanSize(TraceCollection & traces, size_t s);

  /**
   * When a top origin represents an original memory location,
   * its offset indicates how far into the memory region the operation is accessing memory.
   * If this access only touches memory within the first N bytes, remove the top origin.
   *
   * @param traces the traces of some pointer p being accessed
   * @param s the size of the memory access performed at the traced pointer
   * @param N the number of bytes into its memory region an access must be to be kept.
   */
  static void
  RemoveTopOriginsWithinTheFirstNBytes(TraceCollection & traces, size_t s, size_t N);

  /**
   * Checks if any of the top origins in the two trace collections are the same,
   * and have overlapping offsets.
   * Only identical top origins are considered, so if two distinct top origins
   * point to the same memory, that aliasing will not be detected by this method.
   *
   * @param tc1 the collection of traces of p1
   * @param s1 the size of the operation performed at p1
   * @param tc2 the collection of traces of p2
   * @param s2 the size of the operation performed at p2
   * @return true if any top origins are shared, and possibly overlap
   */
  [[nodiscard]] static bool
  DoTraceCollectionsOverlap(TraceCollection & tc1, size_t s1, TraceCollection & tc2, size_t s2);

  /**
   * Checks if the given pointer is the output of an original memory location,
   * AND that the address of the memory location is never passed anywhere that is not
   * traceable back to the original operation.
   *
   * Only \ref AllocaOperation is fully traceable, and only when the address can be traced to all
   * uses, and all uses are loads and stores. If the address is passed to a function, or stored in
   * a variable, the \ref AllocaOperation is not fully traceable.
   * For a fully traceable \ref AllocaOperation, any use of its address can therefore be traced back
   * to the \ref AllocaOperation using the \ref TraceAllPointerOrigins function
   *
   * In summary: this function performs a simple, local, escape analysis for ALLOCAs.
   * Its result is cached for performance.
   *
   * @param pointer the pointer output to be analyzed
   * @return true if the output is an original pointer, and it can be fully traced
   */
  [[nodiscard]] bool
  IsOriginalOriginFullyTraceable(const rvsdg::Output & pointer);

  /**
   * Checks if the given trace collection only contains top origins that are fully traced.
   * @param traces the trace collection
   * @return true if all top origins are fully traceable
   */
  [[nodiscard]] bool
  HasOnlyFullyTraceableTopOrigins(TraceCollection & traces);

  /**
   * The number of outputs a pointer can be traced to before giving up
   * @see setMaxTraceCollectionSize
   */
  size_t maxTraceCollectionSize_ = 1000;

  /**
   * Memoization of "fully traceable" (escape analysis) queries.
   * It assumes that no changes are made to the underlying RVSDG between queries.
   */
  std::unordered_map<const rvsdg::Output *, bool> IsFullyTraceable_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_LOCALALIASANALYSIS_HPP
