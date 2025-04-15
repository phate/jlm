/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/util/Statistics.hpp>

#include <unordered_map>

namespace jlm::llvm::aa
{

/**
 * Class for evaluating the precision of a PointsToGraph on an RVSDG module.
 * Uses a pairwise alias analysis to ask MayAlias queries on relevant pairs of pointers.
 */
class PrecisionEvaluator
{
  class PrecisionStatistics;

public:
  /**
   * Option configuring how pointers are collected and evaluated for may-alias precision
   */
  enum class Mode
  {
    // Load operations are only uses, while store operations both use and clobber
    ClobberingStores,

    // In this mode, all loads and stores are both uses and clobbers
    AllLoadStorePairs
  };

  explicit PrecisionEvaluator(Mode mode)
      : Mode_(mode)
  {}

  void
  SetMode(Mode mode)
  {
    Mode_ = mode;
  }

  Mode
  GetMode() const noexcept
  {
    return Mode_;
  }

  void
  EvaluateAliasAnalysisClient(
      const rvsdg::RvsdgModule & rvsdgModule,
      AliasAnalysis & aliasAnalysis,
      util::StatisticsCollector & statisticsCollector);

private:
  void
  EvaluateAllFunctions(const rvsdg::Region & region, AliasAnalysis & aliasAnalysis);

  void
  EvaluateFunction(const rvsdg::LambdaNode & function, AliasAnalysis & aliasAnalysis);

  void
  CollectPointersFromRegion(const rvsdg::Region & region);

  void
  CollectPointersFromSimpleNode(const rvsdg::SimpleNode & node);

  void
  CollectPointersFromStructuralNode(const rvsdg::StructuralNode & node);

  // Adds a value to the list of pointer operations in the function being evaluated
  void
  CollectPointer(const rvsdg::output * value, size_t size, bool isUse, bool isClobber);

  /**
   * Updates the pointers collected in the current context by tracing their origin.
   * This is only done for pointers that are trivially passed directly through nodes or into regions.
   * @see NormalizePointerValue() in AliasAnalysis.hpp
   */
  void NormalizePointerValues();

  /**
   * Removes repeated instances of the same (pointer, size) pair in the current context
   */
  void RemoveDuplicates();

  // Called once all functions have been evaluated, to calculate and print averages
  void
  CalculateAverageMayAliasRate(const util::file & outputFile, PrecisionStatistics & statistics)
      const;

  // How pointers are counted in the precision evaluation
  Mode Mode_;

  // Alias analysis precision info for a set of pointer usages
  struct PrecisionInfo
  {
    struct UseInfo
    {
      uint64_t NumNoAlias = 0;
      uint64_t NumMayAlias = 0;
      uint64_t NumMustAlias = 0;
    };

    /**
     * For each pointer use, how it relates to all (other) clobbering operations in the function.
     * The relationships are represented as alias query results.
     */
    std::vector<UseInfo> UseOperations;

    /**
     * The number of operations classified as clobbers
     */
    uint64_t NumClobberingOperations = 0;

    /**
     * The number of operations that are either uses, clobbers, or both
     */
    uint64_t NumOperations = 0;
  };

  struct Context
  {
    // Precision info per function in the evaluated module
    std::unordered_map<const rvsdg::LambdaNode *, PrecisionInfo> PerFunctionPrecision;

    /**
     * During traversal of the current function, collects relevant operations on pointers.
     * Each operation is represented by a tuple (pointer value, byte size, isUse, isClobber).
     * All operations should be either a use, a clobber or both.
     * @see PrecisionEvaluationMode for setting which operations are counted and how
     */
    std::vector<std::tuple<const rvsdg::output *, size_t, bool, bool>> PointerOperations;
  };

  Context Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP
