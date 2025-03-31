/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP

#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/rvsdg/RvsdgModule.hpp>
#include <jlm/util/Statistics.hpp>

#include <unordered_map>

namespace jlm::llvm::aa
{

/**
 * Interface for making alias analysis queries about pairs of pointers
 */
class PairwiseAliasAnalysis
{
public:
  PairwiseAliasAnalysis();
  virtual ~PairwiseAliasAnalysis();

  /**
   * If this alias analysis is unable to determine NoAlias, it queries the optional backup analysis.
   * @param backup the PairwiseAliasAnalysis instance to use as backup
   */
  void
  SetBackup(PairwiseAliasAnalysis * backup) noexcept
  {
    Backup_ = backup;
  }

  PairwiseAliasAnalysis *
  GetBackup() const noexcept
  {
    return Backup_;
  }

  /**
   * Checks if the pointers represented by p1 and p2 may alias.
   * If this analysis is unable to prove No Alias, and a backup is provided, the backup is asked.
   * @param p1 the first pointer value
   * @param p2 the second pointer value
   * @return false if the analysis is able to prove that they do not alias, true otherwise
   */
  bool
  MayAlias(const rvsdg::output & p1, const rvsdg::output & p2);

  /**
   * @return a string description of the pairwise alias analysis, including any backup instances
   */
  [[nodiscard]] std::string
  ToString() const;

protected:
  virtual bool
  MayAliasImpl(const rvsdg::output & p1, const rvsdg::output & p2) = 0;

  virtual std::string
  ToStringImpl() const = 0;

private:
  // Another instance of PairwiseAliasAnalysis, queried if the current analysis says "May Alias"
  PairwiseAliasAnalysis * Backup_ = nullptr;
};

/**
 * Interface for making alias pairwise analysis queries to a PointsToGraph
 */
class PointsToGraphAliasAnalysis final : public PairwiseAliasAnalysis
{
public:
  explicit PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph);
  ~PointsToGraphAliasAnalysis() override;

protected:
  bool
  MayAliasImpl(const rvsdg::output & p1, const rvsdg::output & p2) override;

  std::string
  ToStringImpl() const override;

private:
  // The PointsToGraph used to answer alias queries
  PointsToGraph & PointsToGraph_;
};

/**
 * Option configuring how pointers are collected and evaluated for may-alias precision
 */
enum class PrecisionEvaluationMode
{
  // In this mode, a pointer use is a store / load, and a clobber is a store
  ClobberingStores,

  // In this mode, all pointer outputs are both used, and considered to clobber.
  // This is equivalent to checking all pairs of pointers in each function.
  // Temporary pointers, such as the result of a pointer offset, are counted as separate pointers.
  // This is consistent with Lattner's 2007 DSA paper.
  AllPointerPairs
};

/**
 * Class for evaluating the precision of a PointsToGraph on an RVSDG module.
 * Uses a pairwise alias analysis to ask MayAlias queries on relevant pairs of pointers.
 */
class PrecisionEvaluator
{
  class PrecisionStatistics;

public:
  explicit PrecisionEvaluator(PrecisionEvaluationMode mode)
      : Mode_(mode)
  {}

  void
  SetMode(PrecisionEvaluationMode mode)
  {
    Mode_ = mode;
  }

  PrecisionEvaluationMode
  GetMode() const noexcept
  {
    return Mode_;
  }

  void
  EvaluateAliasAnalysisClient(
      const rvsdg::RvsdgModule & rvsdgModule,
      PairwiseAliasAnalysis & aliasAnalysis,
      util::StatisticsCollector & statisticsCollector);

private:
  void
  EvaluateAllFunctions(const rvsdg::Region & region, PairwiseAliasAnalysis & aliasAnalysis);

  void
  EvaluateFunction(const rvsdg::LambdaNode & function, PairwiseAliasAnalysis & aliasAnalysis);

  void
  CollectPointersFromFunctionArguments(const rvsdg::LambdaNode & function);

  void
  CollectPointersFromRegion(const rvsdg::Region & region);

  void
  CollectPointersFromSimpleNode(const rvsdg::SimpleNode & node);

  void
  CollectPointersFromStructuralNode(const rvsdg::StructuralNode & node);

  // Determines if the given value is regarded as representing a pointer
  bool
  IsPointerCompatible(const rvsdg::output * value);

  // Adds a value to the list of pointer uses and/or clobbers in the function being evaluated
  // currently
  void
  CollectPointer(const rvsdg::output * value, bool isUse, bool isClobber);

  // Called once all functions have been evaluated, to calculate and print averages
  void
  CalculateAverageMayAliasRate(const util::file & outputFile, PrecisionStatistics & statistics)
      const;

  // How pointers are counted in the precision evaluation
  PrecisionEvaluationMode Mode_;

  // Alias analysis precision info for a set of pointer usages
  struct PrecisionInfo
  {
    // The number of points in the function that are considered to be pointer clobbering
    uint64_t NumClobberingPointers;

    /**
     * When a pointer is used, how many of the clobbering pointers in the function may it alias?
     * Each value is a double between 0 and 1, where 1 means it may alias with every clobber.
     * If the use is itself a clobber, it only considers all other clobbers.
     */
    std::vector<double> UsedPointerMayAlias;

    /**
     * Adds a pointer use to the statistics.
     * Calculates the ratio:
     *  other clobbers I may alias / number of other clobbers in function
     * @param useIsClobber true if the pointer use is also a clobber
     * @param numClobbersMayAlias the number of clobber operations in the function the use may alias
     */
    void
    AddPointerUse(bool useIsClobber, uint64_t numClobbersMayAlias)
    {
      // If this use is itself a clobber, omit it from the ratio calculation
      numClobbersMayAlias -= useIsClobber;
      auto numOtherClobbers = NumClobberingPointers - useIsClobber;

      // Skip functions that do not have any clobbering points, to avoid division by zero
      if (numOtherClobbers == 0)
        return;

      auto ratio = numClobbersMayAlias / static_cast<double>(numOtherClobbers);
      UsedPointerMayAlias.push_back(ratio);
    }
  };

  struct Context
  {
    // Precision info per function in the evaluated module
    std::unordered_map<const rvsdg::LambdaNode *, PrecisionInfo> PerFunctionPrecision;

    /**
     * During traversal of the current function, which pointers are used.
     * The use consists of a pointer value, and a bool that is true if the use is also a clobber.
     * How pointers are counted depends on the configured mode.
     * The same pointer can also be used multiple times.
     * @see PrecisionEvaluationMode
     */
    std::vector<std::pair<const rvsdg::output *, bool>> PointerUses;
    std::vector<const rvsdg::output *> PointerClobbers;

    // Keeps count of the number of MayAlias queries made
    uint64_t NumMayAliasQueries = 0;
  };

  Context Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP
