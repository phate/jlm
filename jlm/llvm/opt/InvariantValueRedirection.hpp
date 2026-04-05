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

/** \brief Invariant Value Redirection
 *
 * Invariant Value Redirection (IVR) redirects invariant edges around gamma, theta, call, and load
 * nodes. It does this by diverting all users of invariant outputs to the origin of the respective
 * inputs. In case of nested nodes, the transformation processes the innermost nodes first to ensure
 * that the outputs of outer nodes are correctly identified as invariant. Moreover, IVR processes a
 * lambda node before all the lambda's call nodes to ensure that the outputs of call nodes are
 * correctly identified as invariant. Nodes that become dead throughout the transformation are
 * pruned from a region.
 *
 * ### Gamma output redirection
 * The output of a gamma node is considered invariant if all the corresponding region results are
 * connected to the arguments of a single gamma input. All the users of a gamma output are diverted
 * to the origin of this gamma input.
 *
 * ### Theta output redirection
 * A loop variable is considered invariant if its post value is connected to its corresponding pre
 * value. All the users of the loop variables' output are diverted to the origin of the
 * corresponding input. See rvsdg::ThetaLoopVarIsInvariant() for more details.
 *
 * ### Theta/gamma correlation redirection
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
 *
 * ### Call output redirection
 * The output of a call node is considered invariant if the respective result of the corresponding
 * lambda is connected to an argument of the lambda. All the users of a call output are diverted to
 * the origin of the call input corresponding to the lambda argument. Invariant Value Redirection
 * for call nodes works only on non-recursive direct calls as IVR needs to inspect the lambda body
 * in order to determine whether a value is simply routed through the lambda.
 *
 * ### Load memory state redirection
 * The memory states of a load node can be diverted to their respective input's origin if the loaded
 * value output of the load node is dead.
 *
 */
class InvariantValueRedirection final : public rvsdg::Transformation
{
  class Statistics;

public:
  struct Configuration
  {
    bool enableGammaOutputRedirection = true;
    bool enableThetaOutputRedirection = true;
    bool enableThetaGammaCorrelationRedirection = true;
    bool enableCallOutputRedirection = true;
    bool enableLoadMemoryStateRedirection = true;
  };

  ~InvariantValueRedirection() override;

  explicit InvariantValueRedirection(Configuration configuration)
      : Transformation("InvariantValueRedirection"),
        configuration_(std::move(configuration))
  {}

  void
  Run(rvsdg::RvsdgModule & module, util::StatisticsCollector & statisticsCollector) override;

  /**
   * Creates an instance of \ref InvariantValueRedirection and invokes its \ref Run() method.
   *
   * @param rvsdgModule The RVSDG on which invariant value redirection is invoked on.
   * @param configuration The configuration for the instance.
   */
  static void
  createAndRun(rvsdg::RvsdgModule & rvsdgModule, Configuration configuration);

private:
  void
  redirectInRootRegion(rvsdg::Graph & rvsdg);

  void
  redirectInRegion(rvsdg::Region & region);

  void
  redirectInSubregions(rvsdg::StructuralNode & structuralNode);

  static void
  redirectGammaOutputs(rvsdg::GammaNode & gammaNode);

  static void
  redirectThetaOutputs(rvsdg::ThetaNode & thetaNode);

  /**
   * Redirects invariant loop variables from theta nodes of which thet predicate statically
   * correlates with the predicate of a gamma node.
   *
   * @param thetaNode The \ref rvsdg::ThetaNode of which the loop variables are redirected.
   */
  static void
  redirectThetaGammaOutputs(rvsdg::ThetaNode & thetaNode);

  static void
  redirectCallOutputs(rvsdg::SimpleNode & callNode);

  /**
   * Redirects the load node's users of the memory state outputs to the origins' of the respective
   * memory state inputs, if the value output of the load node is dead.
   *
   * @param loadNode The load node for which the memory states are redirected.
   */
  static void
  redirectLoadMemoryStates(rvsdg::SimpleNode & loadNode);

  Configuration configuration_{};
};

}

#endif
