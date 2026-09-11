/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TRACE_HPP
#define JLM_RVSDG_TRACE_HPP

#include <jlm/rvsdg/node.hpp>

namespace jlm::rvsdg
{
class GammaNode;
class ThetaNode;

/**
 * Helper class for tracing through RVSDG graphs to find the origins of outputs.
 * Traces through simple nodes that do not affect the value,
 * through structural nodes when the value is invariant,
 * and out of structural nodes when the value is passed in.
 *
 * It supports caching of traced values at gamma and theta outputs to avoid the retracing of a value
 * through these structural nodes.
 */
class OutputTracer
{
public:
  /**
   * Different policies for how to handle outputs of structural nodes during tracing.
   * The policies are numbered such that a greater number means more effort is spent.
   *
   * These policies only apply to gamma and theta nodes.
   * For phi nodes, see \ref isEnteringPhiNodes()
   */
  enum class StructuralNodePolicy
  {
    // Perform a quick check to see if the structural output is trivially invariant,
    // i.e., gets its value directly from a single subregion argument in all subregions.
    // If so, tracing continues from the corresponding input to the structural node.
    traceThroughTriviallyInvariant = 0,

    // Performs the same check as above, but tries harder to determine invariance.
    // Tracing continues inside the subregion, to check if it can reach a subregion argument.
    // If all subregions reach the same argument, the structural node is invariant,
    // and tracing continue from the input of the structural node.
    traceThroughIfDetectedInvariant = 1,

    // Performs the same checks as above, but can also trace outputs that are not invariant.
    // Tracing can continue inside subregions, such as inside a theta subregion,
    // or inside a gamma subregion when the other subregions are unreachable or
    // provide undefined values.
    // Unlike the previous policies, this means the final return value of the tracer can be
    // inside a region that is not an ancestor of the region where tracing started.
    traceIntoSubregions = 2
  };

  virtual ~OutputTracer();

  /**
   * Creates an OutputTracer with the default configuration
   */
  OutputTracer() noexcept;

  /**
   * When tracing reaches the output of a structural node, how should the tracer continue.
   * @see StructuralNodePolicy
   * @return the current structural node policy
   */
  [[nodiscard]] StructuralNodePolicy
  getStructuralNodePolicy() const noexcept
  {
    return structuralNodePolicy_;
  }

  /**
   * Checks if the current \ref StructuralNodePolicy allows checking for invariant outputs
   * of structural nodes by performing tracing inside the subregion(s) of the node.
   *
   * @return true if the policy allows deep invariance checking, false otherwise
   */
  [[nodiscard]] bool
  structuralNodePolicyAllowsDeepInvarianceChecking() const noexcept
  {
    return structuralNodePolicy_ >= StructuralNodePolicy::traceThroughIfDetectedInvariant;
  }

  /**
   * Checks if the current \ref StructuralNodePolicy allows tracing to enter subregions,
   * and returning value origins that are inside subregions or sibling regions of the region
   * where tracing started.
   *
   * @return true if the policy allows tracing to enter subregions, false otherwise.
   */
  [[nodiscard]] bool
  structuralNodePolicyAllowsTracingIntoSubregions() const noexcept
  {
    return structuralNodePolicy_ >= StructuralNodePolicy::traceIntoSubregions;
  }

  /**
   * Sets the policy for how to trace when reaching outputs of structural nodes.
   * @param value the new value
   */
  void
  setStructuralNodePolicy(StructuralNodePolicy value) noexcept
  {
    structuralNodePolicy_ = value;
  }

  /**
   * Controls if tracing is allowed to leave functions.
   * If true, outputs can be traced out of functions via context arguments.
   * If false, tracing stops if it reaches lambda context arguments.
   * @return true if interprocedural tracing is enabled.
   */
  [[nodiscard]] bool
  isInterprocedural() const noexcept
  {
    return isInterprocedural_;
  }

  /**
   * Enables or disables interprocedural tracing.
   * @see isInterprocedural()
   * @param value the new value
   */
  void
  setInterprocedural(bool value) noexcept
  {
    isInterprocedural_ = value;
  }

  /**
   * Controls if tracing is allowed to enter the subregion of a phi node from its outputs.
   * If true, tracing can go further and reach lambda or delta nodes inside of phi nodes.
   * It does, however, mean that the result of tracing can end up inside a region that is not
   * an ancestor of the starting region in the region tree.
   * @return true if tracing may enter phi nodes.
   */
  [[nodiscard]] bool
  isEnteringPhiNodes() const noexcept
  {
    return enterPhiNodes_;
  }

  /**
   * Enables or disables tracing into phi nodes from the outside.
   * @see isEnteringPhiNodes()
   * @param value the new value
   */
  void
  setEnterPhiNodes(bool value) noexcept
  {
    enterPhiNodes_ = value;
  }

  /**
   * Controls whether caching of structural output invariance is enabled,
   * which can speed up tracing through deeply nested graphs.
   *
   * It only caches outputs that are confirmed to be invariant,
   * so outputs that change from variant to invariant will still be picked up.
   * Outputs that change from invariant to variant need to trigger a manual cache invalidation.
   * @see clearInvarianceCache().
   *
   * @return true if structutal output invariance is cached, false otherwise.
   */
  [[nodiscard]] bool
  isInvarianceCachingEnabled() const noexcept
  {
    return enableInvarianceCaching_;
  }

  /**
   * Enables or disables invariant structural output caching.
   * @see isInvarianceCachingEnabled()
   *
   * @param value the new value
   */
  void
  setInvarianceCaching(bool value) noexcept
  {
    enableInvarianceCaching_ = value;
  }

  /**
   * Clears the invariance cache.
   *
   * This only needs to done if invariance caching is enabled.
   * @see isInvarianceCachingEnabled()
   *
   * It only needs to be done if either of the following happens:
   *  - structural outputs that were once invariant have their origin changed
   *    inside the structural node to make them no longer be invariant,
   *    or be invariant copies of a different structural input.
   *  - structural input and outputs are removed from the graph.
   */
  void
  clearInvarianceCache()
  {
    invariantOutputCache_.clear();
  }

  /**
   * Traces from the given \p output to find the source of the output's value.
   * @param output the output to trace from.
   */
  [[nodiscard]] Output &
  trace(Output & output);

  /**
   * Traces from the given \p output to find the source of the output's value.
   * The optional parameter \p withinRegion prevents values from being traced out of the region. If
   * \p withinRegion is a nullptr, the tracing will continue until the output no longer changes.
   *
   * @param output the output to trace from.
   * @param withinRegion the region where we stop tracing.
   */
  [[nodiscard]] Output &
  trace(Output & output, const rvsdg::Region * withinRegion);

protected:
  /**
   * Attempts to trace the output of a gamma node through the node.
   * This is only possible if the output can be traced to a gamma entry variable in all subregions,
   * and these entry variables all share the same origin outside the gamma.
   *
   * @pre the \p output is an output of the given \p gammaNode
   *
   * @param gammaNode the gamma node to trace through
   * @param output an output of the given gamma node
   * @return the origin of the output value on the input side of the gamma, or nullptr.
   */
  [[nodiscard]] Output *
  tryTraceThroughGamma(GammaNode & gammaNode, Output & output);

  /**
   * Trace from the the given loop output.
   * If the loop output is found to be loop invariant, the origin of the theta input is returned.
   * Otherwise, if the tracer is allowed to trace into the subregion of structural nodes,
   * the traced origin of the loop variable post inside the theta is returned.
   * Otherwise, nullopt is returned.
   *
   * @pre the \p output is an output of the given \p thetaNode
   *
   * @param thetaNode the theta node to trace through
   * @param output an output of the given theta node
   * @return the origin of the output value on the input side of the theta,
   *         the origin of the loop variable post, or nullopt
   */
  [[nodiscard]] Output *
  traceThetaOutput(ThetaNode & thetaNode, Output & output);

  /**
   * The innermost body of the tracing loop. Should trace at least one step, if possible.
   * If it is not possible to trace further, the same output is returned.
   * @param output the output to trace from.
   * @param withinRegion if not nullptr, tracing stops if it reaches an argument of the region.
   * @return the result of tracing from the given output, if possible. Otherwise, \p output.
   */
  [[nodiscard]] virtual Output &
  traceStep(Output & output, const rvsdg::Region * withinRegion);

  /**
   * Inserts the given \p structuralOutput in the invariance cache.
   * Invariance means that the structural output gets its value from one of the inputs
   * of the structural node, so tracing can pass through the structural node without
   * looking inside its subregions.
   *
   * @param structuralOutput The structural output that was traced.
   * @param structuralInput The corresponding structural input.
   * @return The origin of \p structuralInput for convenience.
   */
  Output *
  insertInInvarianceCache(const Output & structuralOutput, Input & structuralInput);

  /**
   * Performs a lookup in the invariance cache.
   *
   * @param structuralOutput the output to look up in the cache.
   * @return the corresponding structural input the invariant output gets its value from.
   */
  Input *
  lookupInInvarianceCache(const Output & structuralOutput);

  // The policy determining how tracing handles outputs of gamma and theta nodes
  StructuralNodePolicy structuralNodePolicy_ =
      StructuralNodePolicy::traceThroughIfDetectedInvariant;

  // When true, tracing is allowed to continue outside of lambda nodes.
  // When false, tracing will stop at the lambda's context arguments.
  bool isInterprocedural_ = true;

  // When true, tracing can go from the output of a \ref PhiNode into its subregion.
  // When false, tracing will stop at the output of the phi node.
  bool enterPhiNodes_ = true;

  // When true, the tracer can cache the fact that outputs of structural nodes are invariant.
  // Enabling caching means the user of the tracer is responsible for cache invalidation.
  // @see clearInvarianceCache() for details
  bool enableInvarianceCaching_ = false;
  std::unordered_map<const Output *, Input *> invariantOutputCache_{};
};

/**
 * Traces \p output intra-procedurally through the RVSDG. The function is capable of tracing:
 *
 * 1. Through gamma nodes if the exit variable is invariant
 * 2. Out of gamma nodes from entry variable arguments
 * 3. Through theta nodes if the loop variable is invariant
 * 4. Out of theta nodes from the arguments, if the loop variable is invariant
 *
 * Tracing stops when a lambda function argument or context argument is reached.
 *
 * The parameter \p mayEnterSubregions controls if the return value is allowed to
 * be an output located in a subregion or sibling region of the \p output.
 * If false, the return value is guaranteed to be in the same region, or in an ancestor region.
 *
 * @param output The \ref Output that needs to be traced.
 * @param mayEnterSubregions if true, the result can be in a sub/sibling region
 * @return The final value of the tracing.
 */
Output &
traceOutputIntraProcedurally(Output & output, bool mayEnterSubregions);

inline const Output &
traceOutputIntraProcedurally(const Output & output, bool mayEnterSubregions)
{
  return traceOutputIntraProcedurally(const_cast<Output &>(output), mayEnterSubregions);
}

/**
 * Traces \p output through the RVSDG. The optional parameter \p withinRegion prevents values from
 * being traced out of the region. If it is a nullptr, tracing will continue until the output no
 * longer changes. The function is capable of tracing through everything \ref
 * traceOutputIntraProcedurally is, in addition to:
 *
 * 1. From lambda context variables out of the lambda
 * 2. From delta context variables out of the delta
 * 3. From phi context variables out of the phi
 * 4. From phi outputs into the phi subregion
 *
 * It will not trace through phi recursion variables
 *
 * @param output the output to trace.
 * @param mayEnterSubregions if true, the result can be in a sub/sibling region
 * @param withinRegion the region to stop at (if any).
 * @return the final value of the tracing
 */
Output &
traceOutput(Output & output, bool mayEnterSubregions, const rvsdg::Region * withinRegion = nullptr);

inline const Output &
traceOutput(
    const Output & output,
    bool mayEnterSubregions,
    const rvsdg::Region * withinRegion = nullptr)
{
  return traceOutput(const_cast<Output &>(output), mayEnterSubregions, withinRegion);
}

}

#endif // JLM_RVSDG_TRACE_HPP
