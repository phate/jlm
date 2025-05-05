/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>

namespace jlm::llvm::aa
{

/**
 * Interface for making alias analysis queries about pairs of pointers p1 and p2.
 * Each pointer must also have an associated compile time byte size, s1 and s2.
 * The analysis response gives guarantees about the possibility of [p1, p1+s1) and [p2, p2+s2)
 * overlapping.
 *
 * If p1 and p2 are local, they must both be defined in the same function.
 * When both pointers are defined within a region, the alias query is made relative to an
 * execution of that region. Thus, a NoAlias response does not make any guarantees about aliasing
 * between different executions of the region.
 */
class AliasAnalysis
{
public:
  /**
   * The possible responses of an alias query about two memory regions (p1, s1) and (p2, s2)
   */
  enum AliasQueryResponse
  {
    // The analysis guarantees that [p1, p1+s1) and [p2, p2+s2) never overlap
    NoAlias,

    // The analysis is unable to determine any facts about p1 and p2
    MayAlias,

    // p1 and p2 always have identical values (s1 and s2 are ignored)
    MustAlias
  };

  AliasAnalysis();
  virtual ~AliasAnalysis();

  /**
   * @return a string description of the alias analysis
   */
  [[nodiscard]] virtual std::string
  ToString() const = 0;

  /**
   * Queries the alias analysis about two memory regions represented as pointer + size pairs.
   * @param p1 the first pointer value
   * @param s1 the byte size of the first pointer access
   * @param p2 the second pointer value
   * @param s2 the byte size of the second pointer access
   * @return the result of the alias query
   */
  virtual AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) = 0;
};

/**
 * Class for making alias analysis queries against a PointsToGraph
 */
class PointsToGraphAliasAnalysis final : public AliasAnalysis
{
public:
  explicit PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph);
  ~PointsToGraphAliasAnalysis() override;

  std::string
  ToString() const override;

  AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) override;

private:

  /**
   * Determines the size of the memory region represented by the given memory node, if possible.
   * If the memory node represents multiple regions of the same size,
   * (e.g., an ALLOCA[i32]), the size of each represented region (e.g., 4) is returned.
   * @param node the MemoryNode representing an abstract memory location.
   * @return the size of the memory region, in bytes
   */
  [[nodiscard]] static std::optional<size_t>
  GetMemoryNodeSize(const PointsToGraph::MemoryNode & node);

  /**
   * Determines if the given abstract memory location represent exactly one region in memory,
   * such as imports and global variables.
   * As a counterexample, an ALLOCA[i32] can represent multiple 4-byte locations.
   * @param node the MemoryNode for the abstract memory location in question
   * @return true if node represents a single location
   */
  [[nodiscard]] static bool
  IsRepresentingSingleMemoryLocation(const PointsToGraph::MemoryNode & node);

  // The PointsToGraph used to answer alias queries
  PointsToGraph & PointsToGraph_;
};

/**
 * Class using two instances of AliasAnalysis to answer alias analysis queries.
 * If the first analysis responds "May Alias", the second analysis is queried.
 */
class ChainedAliasAnalysis final : public AliasAnalysis
{
public:
  explicit ChainedAliasAnalysis(AliasAnalysis & first, AliasAnalysis & second);
  ~ChainedAliasAnalysis() override;

  std::string
  ToString() const override;

  AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) override;

private:
  AliasAnalysis & First_;
  AliasAnalysis & Second_;
};

/**
 * Class for making alias analysis queries using stateless ad hoc IR traversal
 */
class BasicAliasAnalysis final : public AliasAnalysis
{
public:
  BasicAliasAnalysis();
  ~BasicAliasAnalysis() override;

  std::string
  ToString() const override;

  AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) override;

private:
  struct TracedPointerOrigin;
  struct TraceCollection;

  /**
   * Calculates the byte offset applied by the GEP, if the offset is static.
   * The offset is the number of bytes needed to satisfy
   *   output ptr = input ptr + offset in bytes
   *
   * @param gepNode the node representing the GEP operation
   * @return the offset applied by the GEP, if it is possible to determine at compile time
   */
  [[nodiscard]] static std::optional<int64_t>
  CalculateGepOffset(const rvsdg::SimpleNode & gepNode);

  /**
   * Returns the result of tracing the origin of p as far as possible, without introducing any
   * uncertainty. By not introducing uncertainty, the result may be used to determine MustAlias
   * relations. p is normalized, and traced through GEPs with compile time known offsets, as far as
   * possible.
   *
   * @param p the pointer value to trace
   * @return the TracedPointer for p, which is guaranteed to have a defined offset
   */
  [[nodiscard]] static TracedPointerOrigin
  TracePointerOriginPrecise(const rvsdg::output & p);

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
   * Traces through GEP operations, including those with offsets that are not known at compile time.
   * Also traces through gamma and theta nodes, building a set of multiple possibilities.
   * Tracing stops at top origins, for example an ALLOCA, or the return value of a CALL.
   *
   * @param p the pointer to trace from
   * @param traceCollection the collection of trace points being created
   * @return false if the trace collection reached its maximum allowed size, and tracing aborted
   */
  [[nodiscard]] static bool
  TraceAllPointerOrigins(TracedPointerOrigin p, TraceCollection & traceCollection);

  /**
   * Checks if any of the top origins in the two trace collections are the same,
   * and have overlapping offsets.
   * Only identical top origins are considered, so if two distinct top origins
   * point to the same memory, that aliasing will not be detected by this method.
   *
   * @param tc1
   * @param s1
   * @param tc2
   * @param s2
   * @return
   */
  [[nodiscard]] static bool
  DoTraceCollectionsOverlap(TraceCollection & tc1, size_t s1, TraceCollection & tc2, size_t s2);

  /**
   * Checks if the given pointer is the direct result of a memory location defining operation.
   * These operations are guaranteed to output pointers that do not alias any pointer,
   * except for those that are based on the original pointer itself.
   *
   * For example, the output of an ALLOCA, a DELTA, or a GraphImport, are such original origins.
   *
   * @param pointer the pointer value to check
   * @return true if the pointer is the original pointer to a memory location
   */
  [[nodiscard]] static bool
  IsOriginalOrigin(const rvsdg::output & pointer);

  /**
   * Checks if all top origins in the trace collection are original.
   *
   * @param traces the trace collection
   * @return true if all top origins are original, false otherwise
   */
  [[nodiscard]] static bool
  HasOnlyOriginalTopOrigins(TraceCollection & traces);

  /**
   * Checks if the given pointer may have escaped to somewhere it cannot be traced.
   * The analysis is simple, and considers the output of ALLOCAs.
   * If the output ever reaches an operation that "takes the address" of the ALLOCA,
   * it will be marked as having escaped.
   *
   * This function has the following property:
   * For any original origin that has not escaped, any RVSDG output that holds a pointer to it,
   * can also be traced back to the origin using TraceAllPointerOrigins.
   *
   * @param pointer the output to be analyzed
   * @return false if no
   */
  [[nodiscard]] bool
  HasOriginEscaped(const rvsdg::output & pointer);

  /**
   * Checks if any top origin in the trace collection is defined as escaping.
   * @param traces the trace collection
   * @return true if any top origin escaped
   */
  [[nodiscard]] bool
  HasAnyTopOriginEscaped(TraceCollection & traces);

  /**
   * Memoization of escape analysis queries.
   * It assumes that no changes are made to the underlying RVSDG between queries.
   */
  std::unordered_map<const rvsdg::output *, bool> EscapeAnalysisResults_;
};

/**
 * Determines if the given value is regarded as representing a pointer
 * @param value the value
 * @return true if value represents a pointer, false otherwise
 */
[[nodiscard]] bool
IsPointerCompatible(const rvsdg::output & value);

/**
 * Follows the definition of the given \p output through operations that do not modify its value,
 * and out of / into regions when the value is guaranteed to be the same.
 * @param output the output to trace from
 * @return the most normalized source of the given output
 */
[[nodiscard]] const rvsdg::output &
NormalizeOutput(const rvsdg::output & output);

/**
 * Follows the definition of the given pointer value when it is a trivial copy of another pointer,
 * resulting in a possibly different rvsdg::output that produces exactly the same value.
 * Take for example a program like:
 *
 * p1 = alloca
 * p2, _ = IOBarrier(p1, _)
 * _ = Gamma(_, p2)
 *   [p3]{
 *     x = load p3
 *   }[x]
 *   ...
 *
 * Normalizing p3 yields p1
 *
 * @param pointer the pointer value to be normalized
 * @return a definition of pointer normalized as much as possible
 */
[[nodiscard]] const rvsdg::output &
NormalizePointerValue(const rvsdg::output & pointer);

/**
 * Gets the value of the given \p output as a compile time constant, if possible.
 * The constant is interpreted as a signed value, and sign extended to int64 if needed.
 * This function does not perform any constant folding.
 *
 * @param output the output whose constant value is requested
 * @return the value of the output, or nullopt if it could not be determined.
 */
[[nodiscard]] std::optional<int64_t>
GetConstantIntegerValue(const rvsdg::output & output);

/**
 * Returns the size of the given type's in-memory representation, in bytes.
 * The size must be a multiple of the alignment, just like the C operator sizeof().
 * @param type the ValueType
 * @return the byte size of the type
 */
[[nodiscard]] size_t
GetLlvmTypeSize(const rvsdg::ValueType & type);

/**
 * Returns the alignment of the given type's in-memory representation, in bytes.
 * @param type the ValueType
 * @return the byte alignment of the type
 */
[[nodiscard]] size_t
GetLlvmTypeAlignment(const rvsdg::ValueType & type);

/**
 * Gets the offset at which the given field is located in memory
 * @param structType the struct type
 * @param fieldIndex the index of the field, must be a valid field
 * @return the byte offset of the field, relative to the beginning of the struct.
 */
[[nodiscard]] size_t
GetStructFieldOffset(const StructType & structType, size_t fieldIndex);

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
