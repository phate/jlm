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
   * If alias queries between pairs of loads are desired, set this to true.
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
   * Deduplication removes duplicate instances of (pointer, size) operations in each function.
   * If any of the (pointer, size) operations is a clobber, the deduplicated operation is a clobber.
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
   * with an additional graph containing pointer values, and edges connecting pairs of pointers
   * where the alias analysis responded MayAlias and MustAlias.
   * The graph is written to an output file in the statistics directory.
   *
   * @param aliasingGraphEnabled if true, evaluation will produce an aliasing graph.
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
   * Enables or disables the creation of a file containing precision statistics per function.
   * When enabled, the file is created in the statistics output directory.
   *
   * @param perFunctionOutputEnabled if true, evaluation will print output per function to a file.
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
   * Statistics::Id::AliasAnalysisPrecisionEvaluation, evaluation is skipped.
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

  /**
   * Adds the given pointer + size to the list of operations in the function being evaluated
   */
  void
  CollectPointer(const rvsdg::Output * value, size_t size, bool isClobber);

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
   * @param perFunctionOutput if given, statistics about each function are written to the file
   * @param statistics the statistics instance where the final module-level statistics are written
   */
  void
  CalculateResults(
      std::optional<util::FilePath> perFunctionOutput,
      PrecisionStatistics & statistics) const;

  /**
   * Creates nodes corresponding to operations on p1 and p2 in the aliasing graph,
   * and adds an edge between them based on the alias response.
   * @param p1 the first pointer
   * @param s1 the size of the operation at the first pointer
   * @param p2 the second pointer
   * @param s2 the size of the operation at the second pointer
   * @param response the result of performing an alias query on the operation pair
   */
  void
  AddToAliasingGraph(
      const rvsdg::Output & p1,
      size_t s1,
      const rvsdg::Output & p2,
      size_t s2,
      AliasAnalysis::AliasQueryResponse response);

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

  /**
   * Struct containing the result of adding up and averaging across multiple ClobberInfo structs
   */
  struct AggregatedClobberInfos
  {
    uint64_t NumClobberOperations = 0;

    // Statistics about the average ClobberInfo. Sum to 1
    double ClobberAverageNoAlias = 0.0;
    double ClobberAverageMayAlias = 0.0;
    double ClobberAverageMustAlias = 0.0;

    // Total number of alias query responses
    uint64_t TotalNoAlias = 0;
    uint64_t TotalMayAlias = 0;
    uint64_t TotalMustAlias = 0;
  };

  /**
   * Adds up a list of ClobberInfo structs, where each element represents N clobber operations
   * Returns results for the average clobber, as well as total alias query response counts
   */
  static AggregatedClobberInfos
  AggregateClobberInfos(const std::vector<PrecisionInfo::ClobberInfo> & clobberInfos);

  /**
   * Outputs the aggregated clobber infos to the given output stream
   * @param clobberInfos the result of aggregating a set of clobbering operations
   * @param out the stream to write the results to
   */
  static void
  PrintAggregatedClobberInfos(const AggregatedClobberInfos & clobberInfos, std::ostream & out);

  struct Context
  {
    // Output dot graph, only used if dumping a graph of alias analysis-response edges is enabled
    util::graph::Graph * AliasingGraph_ = {};

    // Precision info per function in the evaluated module
    std::unordered_map<const rvsdg::LambdaNode *, PrecisionInfo> PerFunctionPrecision;

    /**
     * During traversal of the current function, collects relevant operations on pointers.
     * Each operation is represented by a tuple (pointer value, byte size, isClobber, multiplier).
     * The multiplier is used by \ref AggregateDuplicates() to represent duplicates efficiently.
     */
    std::vector<std::tuple<const rvsdg::Output *, size_t, bool, size_t>> PointerOperations;
  };

  // Whether to consider loads as clobbers
  bool LoadsConsideredClobbers_ = false;

  // Whether to deduplicate pointers
  bool DeduplicatePointers_ = false;

  // Whether to create an aliasing graph and write it to a file
  bool AliasingGraphEnabled_ = false;

  // Whether to create a file containing aliasing statistics per function
  bool PerFunctionOutputEnabled_ = false;

  // Data used during evaluation
  Context Context_;
};

}

#endif // JLM_LLVM_OPT_ALIAS_ANALYSES_PRECISIONEVALUATOR_HPP
