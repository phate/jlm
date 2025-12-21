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
 */
class OutputTracer
{
public:
  virtual ~OutputTracer();

  /**
   * Creates an OutputTracer with the default configuration
   */
  OutputTracer();

  /**
   * When tracing reaches the output of a structural node, how much effort should be made to
   * check if the output is an invariant copy of one of the node's inputs.
   * If true, tracing is performed inside the subregion(s) of the node.
   * If false, only a simple invariant check is performed, which only detects invariance if
   * the region result is directly connected to a region argument.
   * @return true if tracing though the subregions of structural nodes is enabled.
   */
  [[nodiscard]] bool
  isTracingThroughStructuralNodes() const noexcept
  {
    return traceThroughStrucutalNodes_;
  }

  /**
   * Enables or disables tracing through the subregions of structural nodes.
   * @see isTracingThroughStructuralNodes
   * @param value the new value
   */
  void
  setTraceThroughStructuralNodes(bool value) noexcept
  {
    traceThroughStrucutalNodes_ = value;
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
   * Traces from the given \p output to find the source of the output's value.
   * @param output the output to trace from.
   */
  [[nodiscard]] Output &
  trace(Output & output);

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
   * Attempts to trace the output of a theta node through the node.
   * This is only possible if the output is invariant within the theta node.
   *
   * @pre the \p output is an output of the given \p thetaNode
   *
   * @param thetaNode the theta node to trace through
   * @param output an output of the given theta node
   * @return the origin of the output value on the input side of the theta, or nullptr.
   */
  [[nodiscard]] Output *
  tryTraceThroughTheta(ThetaNode & thetaNode, Output & output);

protected:
  /**
   * Traces from the given \p output to find the source of the output's value.
   * @param output the output to trace from.
   * @param mayLeaveRegion if false, tracing stops if it reaches a region argument
   */
  [[nodiscard]] Output &
  trace(Output & output, bool mayLeaveRegion);

  /**
   * The innermost body of the tracing loop. Should trace at least one step, if possible.
   * If it is not possible to trace further, the same output is returned.
   * @param output the output to trace from
   * @param mayLeaveRegion if false, tracing stops if it reaches a region argument
   * @return the result of tracing from the given output, if possible. Otherwise, \p output.
   */
  [[nodiscard]] virtual Output &
  traceStep(Output & output, bool mayLeaveRegion);

  // When true, tracing enters subregions of structural nodes to check if the value is invariant.
  // When false, values are only considered invariant if they are directly connected to arguments.
  bool traceThroughStrucutalNodes_ = true;

  // When true, tracing can go from the output of a Phi node into its subregion.
  // When false, tracing will stop at the Phi output.
  bool enterPhiNodes_ = true;

  // When true, tracing is allowed to continue outside of lambda nodes.
  // When false, tracing will stop at the lambda's context arguments.
  bool isInterprocedural_ = true;
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
 * @param output The \ref Output that needs to be traced.
 * @return The final value of the tracing.
 */
Output &
traceOutputIntraProcedurally(Output & output);

inline const Output &
traceOutputIntraProcedurally(const Output & output)
{
  return traceOutputIntraProcedurally(const_cast<Output &>(output));
}

/**
 * Traces \p output through the RVSDG.
 * The function is capable of tracing through everything \ref traceOutputIntraProcedurally is,
 * in addition to:
 *
 * 1. From lambda context variables out of the lambda
 * 2. From delta context variables out of the delta
 * 3. From phi context variables out of the phi
 * 4. From phi outputs into the phi subregion
 *
 * It will not trace through phi recursion variables
 *
 * @param output the output to trace.
 * @return the final value of the tracing
 */
Output &
traceOutput(Output & output);

inline const Output &
traceOutput(const Output & output)
{
  return traceOutput(const_cast<Output &>(output));
}

}

#endif // JLM_RVSDG_TRACE_HPP
