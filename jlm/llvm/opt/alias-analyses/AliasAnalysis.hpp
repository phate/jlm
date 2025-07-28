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
  virtual ~AliasAnalysis() noexcept;

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
  Query(const rvsdg::Output & p1, size_t s1, const rvsdg::Output & p2, size_t s2) = 0;
};

/**
 * Determines if the given value is regarded as representing a pointer
 * @param value the value
 * @return true if value represents a pointer, false otherwise
 */
[[nodiscard]] bool
IsPointerCompatible(const rvsdg::Output & value);

/**
 * Follows the definition of the given \p output through operations that do not modify its value,
 * and out of / into regions when the value is guaranteed to be the same.
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
 * @param output the output to trace from
 * @return the most normalized source of the given output
 */
[[nodiscard]] const rvsdg::Output &
NormalizeOutput(const rvsdg::Output & output);

/**
 * Follows the definition of the given pointer value when it is a trivial copy of another pointer,
 * resulting in a possibly different rvsdg::output that produces exactly the same value.
 *
 * @param pointer the pointer value to be normalized
 * @return a definition of pointer normalized as much as possible
 * @see NormalizeOutput
 */
[[nodiscard]] const rvsdg::Output &
NormalizePointerValue(const rvsdg::Output & pointer);

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
