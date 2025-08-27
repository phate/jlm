/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP

#include "jlm/rvsdg/lambda.hpp"
#include "jlm/rvsdg/RvsdgModule.hpp"
#include "jlm/util/GraphWriter.hpp"
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/util/Statistics.hpp>

#include <unordered_map>

namespace jlm::util
{
class GraphWriter;
}

namespace jlm::llvm::aa
{

/**
 * Class for evaluating the precision of a PointsToGraph on an RVSDG module.
 * Uses a pairwise alias analysis to ask MayAlias queries on relevant pairs of pointers.
 * Only considers load and store operations, and not function calls.
 */
class AliasAnalysisPrecisionEvaluator
{
  class PrecisionStatistics;

public:
  AliasAnalysisPrecisionEvaluator();

  ~AliasAnalysisPrecisionEvaluator() noexcept;

  /**
   * Makes loads count as clobbering operations.
   * Alias queries are only made between pairs operations if at least one of them is a clobber.
   * If alias queries between pairs of loads is desired, set this to true.
   *
   * Default: false
   * @param loadsConsideredClobbers if true, loads will be clobbers.
   */
  void
  SetLoadsConsideredClobbers(bool loadsConsideredClobbers) noexcept
  {
    LoadsConsideredClobbers_ = loadsConsideredClobbers;
  }

  bool
  AreLoadsConsideredClobbers() const noexcept
  {
    return LoadsConsideredClobbers_;
  }

  /**
   * Enables or disables the deduplication of pointers during precision evaluation.
   * Deduplication removes repeated instances of the same (pointer, size) pair in each function.
   * Iff any of the (pointer, size) pairs is a clobber, the deduplicated pair is a clobber.
   *
   * @param deduplicatePointers if true, pointers will be deduplicated.
   */
  void
  SetDeduplicatePointers(bool deduplicatePointers) noexcept
  {
    DeduplicatePointers_ = deduplicatePointers;
  }

  bool
  IsDeduplicatingPointers() const noexcept
  {
    return DeduplicatePointers_;
  }

  /**
   * Enables or disables the creation of an aliasing graph.
   * When enabled, the pass produces a GraphWriter output of the module,
   * with an additional graph containing all pointers, with edges connecting MayAlias and MustAlias.
   * The graph is written to an output file in the statistics directory.
   *
   * @param aliasingGraphEnabled if true, evaluation will produce an aliasing graph
   */
  void
  SetAliasingGraphEnabled(bool aliasingGraphEnabled) noexcept
  {
    AliasingGraphEnabled_ = aliasingGraphEnabled;
  }

  bool
  IsAliasingGraphEnabled() const noexcept
  {
    return AliasingGraphEnabled_;
  }

  /**
   * Enables or disables the creation of a file containing aliasing statistics per function.
   * When enabled, the file is created in the statistics output directory.
   *
   * @param perFunctionOutputEnabled if true, evaluation will print output per function to a file
   */
  void
  SetPerFunctionOutputEnabled(bool perFunctionOutputEnabled) noexcept
  {
    PerFunctionOutputEnabled_ = perFunctionOutputEnabled;
  }

  bool
  IsPerFunctionOutputEnabled() const noexcept
  {
    return PerFunctionOutputEnabled_;
  }

  /**
   * Performs alias analysis precision evaluation on the given \p rvsdgModule,
   * using the given \p aliasAnalysis instance.
   * If the given \p statisticsCollector does not demand
   * Statistics::Id::AliasAnalysisPrecisionEvaluation, this is a no-op.
   */
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
  CollectPointer(const rvsdg::Output * value, size_t size, bool isUse, bool isClobber);

  /**
   * Updates the pointers collected in the current context by tracing their origin.
   * This is only done for pointers that are trivially passed directly through nodes or into
   * regions.
   */
  void
  NormalizePointerValues();

  /**
   * Removes repeated instances of the same (pointer, size) pair in the current context.
   * If IsDeduplicatingPointers() is true, duplicate instances are discarded.
   * Otherwise, the duplicated instances are added up and given a multiplier.
   */
  void
  AggregateDuplicates();

  /**
   * Called once all functions have been evaluated, to calculate and print averages.
   * Adds up both the total number of responses, and calculates average response rates per clobber.
   *
   * @param perFunctionOutput if given, statistics about each function are printed to the file
   * @param statistics the statistics instance where the final module-level statistics are written
   */
  void
  CalculateAverageMayAliasRate(
      std::optional<util::File> perFunctionOutput,
      PrecisionStatistics & statistics) const;

  // Alias analysis precision info for a set of pointer usages, such as in a function
  struct PrecisionInfo
  {
    struct ClobberInfo
    {
      // A single ClobberInfo can represent multiple identical clobbers.
      // The multiplier should be applied to all the alias query response counters below,
      // and when calculating the average of all clobbers, the multiplier is the weight.
      uint64_t Multiplier = 1;

      uint64_t NumNoAlias = 0;
      uint64_t NumMayAlias = 0;
      uint64_t NumMustAlias = 0;
    };

    /**
     * For each pointer clobber, how it relates to all (other) use operations in the function.
     * The relationships are represented as alias query results.
     */
    std::vector<ClobberInfo> ClobberOperations;

    /**
     * The total number of clobber operations.
     * Equal to summing up the Multiplier of all ClobberInfos in the ClobberOperations list.
     */
    uint64_t NumClobberOperations = 0;

    /**
     * The total number of operations, including both clobbers and uses
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
     * All operations should be either a use, a clobber, or both.
     * @see PrecisionEvaluationMode for setting which operations are counted and how
     */
    std::vector<std::tuple<const rvsdg::Output *, size_t, bool, bool>> PointerOperations;
  };

  // Adds up a list of ClobberInfo structs, where each element represents N clobber operations
  // Returns results for the average clobber, as well as total alias query response counts
  static void
  AggregateClobberInfos(
      const std::vector<PrecisionInfo::ClobberInfo> & clobberInfos,
      double & clobberAverageNoAlias,
      double & clobberAverageMayAlias,
      double & clobberAverageMustAlias,
      uint64_t & totalNoAlias,
      uint64_t & totalMayAlias,
      uint64_t & totalMustAlias);

  // Whether to consider loads as clobbers
  bool LoadsConsideredClobbers_ = false;

  // Whether to deduplicate pointers
  bool DeduplicatePointers_ = false;

  // Whether to create an aliasing graph and write it to a file
  bool AliasingGraphEnabled_ = false;

  // Whether to create a file containing aliasing statistics per function
  bool PerFunctionOutputEnabled_ = false;

  Context Context_;

  // Output dot graph, only used if dumping a graph of alias analysis-response edges is enabled
  util::graph::Graph * AliasingGraph_ = {};
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP
