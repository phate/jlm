/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_INVARIANTVALUEREDIRECTION_HPP
#define JLM_LLVM_OPT_INVARIANTVALUEREDIRECTION_HPP

#include <jlm/llvm/opt/optimization.hpp>

namespace jive {
class theta_node;
}

namespace jlm {

class RvsdgModule;

/** \brief Invariant Value Redirection Optimization
 *
 * Invariant Value Redirection (IVR) redirects invariant edges around gamma and theta nodes. It does this by diverting
 * all users of invariant gamma and theta outputs to the origin of the respective gamma and theta inputs. In case
 * of nested nodes, the optimization processes the innermost nodes first to ensure that the outputs
 * of outer nodes are correctly identified as invariant.
 *
 * ### Theta Nodes
 * The output of a theta node is considered invariant if the corresponding region result is connected to the
 * corresponding region argument. All the users of a theta output are diverted to the origin of the corresponding
 * theta input.
 *
 * ### Gamma Nodes
 * The output of a gamma node is considered invariant if all the corresponding region results are connected to the
 * arguments of a single gamma input. All the users of a theta output are diverted to the origin of this gamma input.
 */
class InvariantValueRedirection final : public optimization {
public:
  ~InvariantValueRedirection() override;

  void
  run(
    RvsdgModule & rvsdgModule,
    StatisticsCollector & statisticsCollector) override;

private:
  static void
  RedirectInvariantValues(jive::region & region);

  static void
  RedirectInvariantGammaOutputs(jive::gamma_node & gammaNode);

  static void
  RedirectInvariantThetaOutputs(jive::theta_node & thetaNode);
};

}

#endif
