/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INLINE_HPP
#define JLM_LLVM_OPT_INLINE_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class SimpleNode;
class LambdaNode;
}

namespace jlm::llvm
{

/**
 * \brief Performs function inlining on functions that are determined to be good candidates,
 * such as private functions that are only called from a single call site.
 */
class FunctionInlining final : public rvsdg::Transformation
{
public:
  class Statistics;
  struct Context;

  ~FunctionInlining() noexcept override;

  FunctionInlining();

  FunctionInlining(const FunctionInlining &) = delete;

  FunctionInlining &
  operator=(const FunctionInlining &) = delete;

private:
  /**
   * Performs inlining of the given call
   * @param callNode the node containing the \ref CallOperation
   * @param caller the lambda node of the caller
   * @param callee the lambda node of the callee
   */
  static void
  inlineCall(
      rvsdg::SimpleNode & callNode,
      rvsdg::LambdaNode & caller,
      const rvsdg::LambdaNode & callee);

  /**
   * Determines if there is anything in the region that would prevent a function from being inlined.
   * Recurses into subregions.
   * @param region the region in question
   * @param topLevelRegion true if the region is the top level region in its lambda node
   * @return true if nothing disqualifying inlining was found in the region.
   */
  static bool
  canBeInlined(rvsdg::Region & region, bool topLevelRegion);

  /**
   * Determines if the given call should be inlined or not.
   * @param callNode the node containing the \ref CallOperation
   * @param caller the lambda node of the caller
   * @param callee the lambda node of the callee
   * @return
   */
  bool
  shouldInline(
      rvsdg::SimpleNode & callNode,
      rvsdg::LambdaNode & caller,
      rvsdg::LambdaNode & callee);

  /**
   * Determines if the given \p callNode is a call that can be inlined, and if it should be inlined.
   * If yes, inlining is performed.
   * @param callNode the node containing the CallOperation
   * @param callerLambda the lambda containing the call node, a.k.a. the caller
   */
  void
  considerCallForInlining(rvsdg::SimpleNode & callNode, rvsdg::LambdaNode & callerLambda);

  /**
   * Recursively visits all call operations in the region and its subregions,
   * and performs inlining if determined to be beneficial.
   * @param region the region being visited
   * @param lambda the function to which the region belongs
   */
  void
  visitIntraProceduralRegion(rvsdg::Region & region, rvsdg::LambdaNode & lambda);

  /**
   * Visits the given function, performing inlining of calls inside it, and storing facts about
   * it that may be relevant when considering inlining calls to the function.
   * @param lambda the function in question
   */
  void
  visitLambda(rvsdg::LambdaNode & lambda);

  /**
   * Visits all lambda nodes in the given \p region, including in subregions.
   * In each visited lambda, inlining transformations are applied if determined to be beneficial.
   * @param region the region in question
   */
  void
  visitInterProceduralRegion(rvsdg::Region & region);

public:
  /**
   * Performs inlining of the given call node, targeting the given callee function
   * @param callNode the call to inline
   * @param callee the function being inlined
   */
  static void
  inlineCall(rvsdg::SimpleNode & callNode, const rvsdg::LambdaNode & callee);

  /**
   * Determines if there is anything in the function \p callee that prevents it from being inlined.
   * @return true if nothing disqualifying inlining was found in the region.
   */
  static bool
  canBeInlined(const rvsdg::LambdaNode & callee);

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  std::unique_ptr<Context> context_;
};

}

#endif
