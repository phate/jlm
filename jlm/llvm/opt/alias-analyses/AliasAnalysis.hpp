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
 * The analysis response gives guarantees about the possibility of [p1, p1+s1) and [p2, p2+s2) overlapping.
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
   * Queries the alias analysis about two pointers or two memory regions.
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

}

#endif //JLM_LLVM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
