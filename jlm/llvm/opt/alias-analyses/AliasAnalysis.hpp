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
   * Queries the alias analysis about two memory regions represented as pointer + size pairs.
   * @param p1 the first pointer value
   * @param s1 the byte size of the first pointer access
   * @param p2 the second pointer value
   * @param s2 the byte size of the second pointer access
   * @return the result of the alias query
   */
  virtual AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) = 0;

  /**
   * @return a string description of the alias analysis
   */
  [[nodiscard]] virtual std::string
  ToString() const = 0;
};

/**
 * Class for making alias analysis queries against a PointsToGraph
 */
class PointsToGraphAliasAnalysis final : public AliasAnalysis
{
public:
  explicit PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph);
  ~PointsToGraphAliasAnalysis() override;

  AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) override;

  std::string
  ToString() const override;

private:
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

  AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) override;

  std::string
  ToString() const override;

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

  AliasQueryResponse
  Query(const rvsdg::output & p1, size_t s1, const rvsdg::output & p2, size_t s2) override;

  std::string
  ToString() const override;

private:
  /**
   * Checks if the given pointer is the result of a memory location defining operation.
   * These operations are guaranteed to output pointers that do not alias any pointer
   * that is not based on the pointer itself.
   *
   * For example, the output of an ALLOCA, or a GraphImport, are such pointer origins.
   *
   * @param pointer the pointer value to check
   * @return true if the pointer is the original pointer to a created memory location
   */
  [[nodiscard]] bool
  IsOriginalMemoryLocation(const rvsdg::output & pointer);
};

/**
 * Determines if the given value is regarded as representing a pointer
 * @param value the value
 * @return true if value represents a pointer, false otherwise
 */
[[nodiscard]] bool
IsPointerCompatible(const rvsdg::output & value);

/**
 * Follows the definition of the given pointer value when it is a trivial copy of another pointer,
 * resulting in a possibly different rvsdg::output that produces exactly the same value.
 * Take for example a program like:
 *
 * p1 = alloca
 * p2, _ = IOBarrier(p1, _)
 * _ = Delta(_, p2)
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
 * Returns the size of the given type's in-memory representation, in bytes.
 * The size must be a multiple of the alignment, just like the C operator sizeof().
 * @param type the ValueType
 * @return the byte size of the type
 */
[[nodiscard]] size_t
GetLlvmTypeSize(const rvsdg::ValueType & type);

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
