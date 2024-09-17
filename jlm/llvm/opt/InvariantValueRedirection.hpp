/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INVARIANTVALUEREDIRECTION_HPP
#define JLM_LLVM_OPT_INVARIANTVALUEREDIRECTION_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class ThetaNode;
}

namespace jlm::llvm
{

class RvsdgModule;

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
 * ### Theta Nodes
 * The output of a theta node is considered invariant if the corresponding region result is
 * connected to the corresponding region argument. All the users of a theta output are diverted to
 * the origin of the corresponding theta input.
 *
 * ### Gamma Nodes
 * The output of a gamma node is considered invariant if all the corresponding region results are
 * connected to the arguments of a single gamma input. All the users of a gamma output are diverted
 * to the origin of this gamma input.
 *
 * ### Call Nodes
 * The output of a call node is considered invariant if the respective result of the corresponding
 * lambda is connected to an argument of the lambda. All the users of a call output are diverted to
 * the origin of the call input corresponding to the lambda argument. Invariant Value Redirection
 * for call nodes works only on non-recursive direct calls as IVR needs to inspect the lambda body
 * in order to determine whether a value is simply routed through the lambda.
 */
class InvariantValueRedirection final : public optimization
{
  class Statistics;

public:
  ~InvariantValueRedirection() override;

  void
  run(RvsdgModule & rvsdgModule, util::StatisticsCollector & statisticsCollector) override;

private:
  static void
  RedirectInRootRegion(rvsdg::graph & rvsdg);

  static void
  RedirectInRegion(rvsdg::Region & region);

  static void
  RedirectInSubregions(rvsdg::structural_node & structuralNode);

  static void
  RedirectGammaOutputs(rvsdg::GammaNode & gammaNode);

  static void
  RedirectThetaOutputs(rvsdg::ThetaNode & thetaNode);

  static void
  RedirectCallOutputs(CallNode & callNode);
};

}

#endif
