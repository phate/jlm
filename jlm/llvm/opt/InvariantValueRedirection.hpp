/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INVARIANTVALUEREDIRECTION_HPP
#define JLM_LLVM_OPT_INVARIANTVALUEREDIRECTION_HPP

#include <jlm/rvsdg/Transformation.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class Graph;
class Region;
class StructuralNode;
class ThetaNode;
class SimpleNode;
}

namespace jlm::llvm
{

class ThetaGammaPredicateCorrelation;

/** \brief Invariant Value Redirection Optimization
 *
 * Invariant Value Redirection (IVR) redirects invariant edges around gamma, theta, and call nodes.
 * It does this by diverting all users of invariant gamma and theta outputs to the origin of the
 * respective gamma and theta inputs. For call nodes, it redirects all users of invariant call
 * outputs to the origin of the respective call input. In case of nested nodes, the optimization
 * processes the innermost nodes first to ensure that the outputs of outer nodes are correctly
 * identified as invariant. Moreover, IVR processes a lambda node before all the lambda's call nodes
 * to ensure that the outputs of call nodes are correctly identified as invariant.
 *
 * ### Theta nodes
 * A loop variable is considered invariant if its post value is connected to its corresponding pre
 * value. All the users of the loop variables' output are diverted to the origin of the
 * corresponding input. See rvsdg::ThetaLoopVarIsInvariant() for more details.
 *
 * ### Gamma nodes
 * The output of a gamma node is considered invariant if all the corresponding region results are
 * connected to the arguments of a single gamma input. All the users of a gamma output are diverted
 * to the origin of this gamma input.
 *
 * ### Call nodes
 * The output of a call node is considered invariant if the respective result of the corresponding
 * lambda is connected to an argument of the lambda. All the users of a call output are diverted to
 * the origin of the call input corresponding to the lambda argument. Invariant Value Redirection
 * for call nodes works only on non-recursive direct calls as IVR needs to inspect the lambda body
 * in order to determine whether a value is simply routed through the lambda.
 *
 * ### Theta nodes with a predicate that correlates with a gamma node
 * If the theta node has a gamma node in its subregion and for both nodes the predicates correlate,
 * then the theta node's loop variables can be redirected under certain conditions. The
 * correlation of predicates means that either:
 *
 * 1. The theta node's predicate origin is an exit variable of the gamma node and the respective
 * producers of the subregion results are control constants.
 * 2. The producer of the theta node's predicate origin and the gamma node's predicate origin are
 * the same \ref rvsdg::MatchOperation node.
 *
 * These conditions are sufficient to statically determine that either one of the two gamma node's
 * subregions is only executed on loop repetition or loop exit.
 *
 * A loop variable can now be redirected in the following cases:
 *
 * 1. If the loop variable's output is dead, then only the repetition value of the loop variable
 * is of interest. This means that we can redirect the value from the respective entry variable of
 * the gamma node's repetition subregion to the post value.
 *
 * 2. If the loop variable's pre value is dead, then only the exit value of the loop variable is of
 * interest. This means that we can redirect the value from the respective entry variable of the
 * gamma node's exit subregion to the post value.
 */
class InvariantValueRedirection final : public rvsdg::Transformation
{
  class Statistics;

public:
  ~InvariantValueRedirection() override;

  InvariantValueRedirection()
      : Transformation("InvariantValueRedirection")
  {}

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  RedirectInRootRegion(rvsdg::Graph & rvsdg);

  static void
  RedirectInRegion(rvsdg::Region & region);

  static void
  RedirectInSubregions(rvsdg::StructuralNode & structuralNode);

  static void
  RedirectGammaOutputs(rvsdg::GammaNode & gammaNode);

  static void
  RedirectThetaOutputs(rvsdg::ThetaNode & thetaNode);

  /**
   * Redirects invariant loop variables from theta nodes of which thet predicate statically
   * correlates with the predicate of a gamma node.
   *
   * @param thetaNode The \ref rvsdg::ThetaNode of which the loop variables are redirected.
   */
  static void
  redirectThetaGammaOutputs(rvsdg::ThetaNode & thetaNode);

  typedef struct
  {
    rvsdg::Region * repetitionSubregion;
    rvsdg::Region * exitSubregion;
  } GammaSubregionRoles;

  /**
   * Tries to assign the respective roles (exit or repetition) to the subregions of a gamma node
   * that statically determines the predicate of a theta node.
   *
   * @param correlation The predicate correlation between a theta and gamma node.
   * @return The roles of the gamma subregions, otherwise std::nullopt.
   */
  static std::optional<GammaSubregionRoles>
  determineGammaSubregionRoles(const ThetaGammaPredicateCorrelation & correlation);

  static void
  RedirectCallOutputs(rvsdg::SimpleNode & callNode);
};

}

#endif
