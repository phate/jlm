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
 * and through structural nodes when the value is invariant.
 * Traces out of regions
 */
class OutputTracer
{
public:
  virtual ~OutputTracer();

  /**
   * Creates an OutputTracer with the given configuration
   * @param isDeep if true, tracing goes into subregions, trying to trace through structural nodes.
   *        Otherwise, outputs of structural nodes use a simple invariant check.
   * @param isInterprocedural allow tracing out of loops
   */
  OutputTracer(bool isDeep, bool isInterprocedural);

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
   * @param gammaNode the gamma node to trace through
   * @param output an output of the given gamma node
   * @return the origin of the output value on the input side of the gamma, or nullptr.
   */
  [[nodiscard]] Output *
  tryTraceThroughGamma(GammaNode & gammaNode, Output & output);

  /**
   * Attempts to trace the output of a theta node through the node.
   * This is only possible if the output is invariant within the theta node.
   * @param thetaNode the gamma node to trace through
   * @param output an output of the given gamma node
   * @return the origin of the output value on the input side of the gamma, or nullptr.
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
  bool isDeep_;

  // When true, tracing is allowed to continue outside of lambda nodes.
  // When false, tracing will stop at the lambda's context arguments
  bool isInterprocedural_;
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
 *
 * It will not trace through phi recursion variables.
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
