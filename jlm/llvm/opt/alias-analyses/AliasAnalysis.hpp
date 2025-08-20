/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP

#include <jlm/rvsdg/node.hpp>

#include <optional>

namespace jlm::llvm::aa
{

/**
 * Interface for making alias analysis queries about pairs of pointers p1 and p2.
 * Each pointer must also have an associated compile time byte size, s1 and s2.
 * The analysis response gives guarantees about the possibility of [p1, p1+s1) and [p2, p2+s2)
 * overlapping.
 *
 * If p1 and p2 are both defined within a lambda region, it must be the same lambda region.
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
  Query(const rvsdg::Output & p1, size_t s1, const rvsdg::Output & p2, size_t s2) override;

private:
  AliasAnalysis & First_;
  AliasAnalysis & Second_;
};

/**
 * Determines if the given value is regarded as representing a pointer
 * @param value the value in question
 * @return true if value is or contains a pointer, false otherwise
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
 * Gets the value of the given \p output as a compile time constant, if possible.
 * The constant is interpreted as a signed value, and sign extended to int64 if needed.
 * This function does not perform any constant folding.
 * @return the integer value of the output, or nullopt if it could not be determined.
 */
std::optional<int64_t>
TryGetConstantSignedInteger(const rvsdg::Output & output);

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
