/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_TRACE_HPP
#define JLM_LLVM_IR_TRACE_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/Trace.hpp>

#include <optional>

namespace jlm::llvm
{

class OutputTracer : public rvsdg::OutputTracer
{
public:
  explicit OutputTracer(bool enableCaching);

  /**
   * When enabled, tracing can continue through the memory state output of a load,
   * and keep tracing from the load's memory state input.
   * @return true if tracing states through loads is enabled, false otherwise
   */
  bool
  isTracingThroughLoadedStates()
  {
    return traceThroughLoadedStates_;
  }

  /**
   * Enables or disables tracing state edges through load operation nodes.
   * @param traceThroughLoadedStates the new value
   */
  void
  setTraceThroughLoadedStates(bool traceThroughLoadedStates)
  {
    traceThroughLoadedStates_ = traceThroughLoadedStates;
  }

protected:
  [[nodiscard]] rvsdg::Output &
  traceStep(rvsdg::Output & output, const rvsdg::Region * withinRegion) override;

private:
  bool traceThroughLoadedStates_ = false;
};

/**
 * Traces the origin of the given \p output to find the origin of the value. The optional parameter
 * \p withinRegion prevents values from being traced out of the region. If it is a nullptr, tracing
 * will continue until the output no longer changes.
 * Traces through everything handled by \ref jlm::rvsdg::traceOutput, with the addition of
 * LLVM-specific operations.
 *
 * @param output the output to start tracing from
 * @param withinRegion the region to stop at (if any).
 * @return the maximally traced output
 */
rvsdg::Output &
traceOutput(rvsdg::Output & output, const rvsdg::Region * withinRegion = nullptr);

inline const rvsdg::Output &
traceOutput(const rvsdg::Output & output, const rvsdg::Region * withinRegion = nullptr)
{
  return llvm::traceOutput(const_cast<rvsdg::Output &>(output), withinRegion);
}

/**
 * Attempts to find the constant integer value of a given \p output,
 * by normalizing it back to its source operation.
 * If it is a constant integer operation, the constant is returned as a signed 64-bit integer.
 * @param output an output that may be a constant integer.
 * @return the constant integer value if found, otherwise nullopt.
 */
std::optional<int64_t>
tryGetConstantSignedInteger(const rvsdg::Output & output);

}

#endif // JLM_LLVM_IR_TRACE_HPP
