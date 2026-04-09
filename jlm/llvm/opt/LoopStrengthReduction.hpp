/*
 * Copyright 2026 Andreas Lilleby Hjulstad <andreas.lilleby.hjulstad@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_LOOP_STRENGTH_REDUCTION_HPP
#define JLM_LLVM_OPT_LOOP_STRENGTH_REDUCTION_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/llvm/opt/ScalarEvolution.hpp>
#include <jlm/rvsdg/Transformation.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm
{
class LoopStrengthReduction final : public jlm::rvsdg::Transformation
{
  class Statistics;
  class Context;

public:
  ~LoopStrengthReduction() noexcept override;

  LoopStrengthReduction();

  LoopStrengthReduction(const LoopStrengthReduction &) = delete;

  LoopStrengthReduction(LoopStrengthReduction &&) = delete;

  LoopStrengthReduction &
  operator=(const LoopStrengthReduction &) = delete;

  LoopStrengthReduction &
  operator=(LoopStrengthReduction &&) = delete;

  void
  Run(rvsdg::RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  void
  ProcessRegion(rvsdg::Region & region);

  void
  ReduceStrength(rvsdg::ThetaNode & thetaNode);

  /**
   * Checks if the RVSDG subtree of the output contains an IntegerMulOperation or
   * IntegerShlOperation (which we treat as multiplication by 2) somewhere in the tree.
   *
   * @param output The output to be checked
   * @return true if the subtree contains a multiplication operation, otherwise false.
   */
  bool
  ContainsMul(const rvsdg::Output & output);

  /**
   * Checks if the operation depends on an induction variable. By induction variable we mean a loop
   * variable that evolves in a predictable way, which is the same as checking if its chrec does not
   * contain any SCEVUnknown or SCEVInit elements.
   *
   * @param output The output to be checked
   * @return true if the output depends on an induction variable, otherwise false.
   */
  bool
  DependsOnInductionVariable(const rvsdg::Output & output);

  bool
  IsValidCandidateOperation(const rvsdg::Output & output);

  void
  ProcessOutput(
      rvsdg::Output & output,
      rvsdg::ThetaNode & thetaNode,
      util::HashSet<rvsdg::Output *> & candidateOperations,
      util::HashSet<rvsdg::Output *> & visited);

  void
  ReplaceCandidateOperation(rvsdg::Output & output, rvsdg::ThetaNode & thetaNode);

  /**
   * Finds a path of gamma/theta nodes from an outer loop to an inner loop (if one exists).
   *
   * The reason the path does not just contain theta nodes is that in the case of head-controlled
   * loops, we have gammas interleaving nested theta nodes.
   *
   *
   * @param from The loop we start in.
   * @param to The loop we want to find a path to.
   * @return a list of nodes needed to traverse from the starting loop to the target loop if one
   * exists, otherwise std::nullopt.
   */
  static std::optional<std::vector<rvsdg::StructuralNode *>>
  FindLoopPath(const rvsdg::ThetaNode & from, const rvsdg::ThetaNode & to);

  /**
   * Tries to route a value down from an outer loop to an inner loop.
   *
   * Uses FindLoopPath() to try to find a path from the starting loop to
   * the target loop.
   *
   * @param origin The output of the value we want to route.
   * @param from The outermost loop (tne loop which contains origin).
   * @param to The innermost loop.
   * @return the output of the routed value in the parent region of the innermost loop if a path
   * exists, otherwise std::nullopt;
   */
  static std::optional<rvsdg::Output *>
  RouteValueThroughLoops(
      rvsdg::Output & origin,
      const rvsdg::ThetaNode & from,
      const rvsdg::ThetaNode & to);

  std::optional<rvsdg::Output *>
  HoistChrec(const SCEVChainRecurrence & chrec, const rvsdg::ThetaNode & thetaNode, size_t numBits);

  std::optional<rvsdg::Output *>
  HoistSCEVExpresssion(const SCEV & scev, rvsdg::ThetaNode & thetaNode, size_t numBits);

  void
  ReplaceGEPOperation(
      std::unique_ptr<SCEVChainRecurrence> & chrec,
      rvsdg::Output & output,
      rvsdg::ThetaNode & thetaNode,
      const std::shared_ptr<const PointerType> & pointerType);

  void
  ReplaceArithmeticOperation(
      std::unique_ptr<SCEVChainRecurrence> & chrec,
      rvsdg::Output & output,
      rvsdg::ThetaNode & thetaNode,
      const std::shared_ptr<const rvsdg::BitType> & bitType);

  std::optional<rvsdg::ThetaNode::LoopVar>
  CreateNewArithmeticInductionVariable(
      const SCEVChainRecurrence & chrec,
      rvsdg::ThetaNode & thetaNode,
      size_t numBits);

  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEVChainRecurrence>> ChrecMap_;
  std::unordered_map<const rvsdg::Output *, std::unique_ptr<SCEV>> SCEVMap_;
  std::unordered_map<const rvsdg::Output *, bool> DependsOnIVMemo_;
  std::unordered_map<const rvsdg::Output *, bool> ContainsMulMemo_;

  std::unique_ptr<Context> Context_;
};

}

#endif
