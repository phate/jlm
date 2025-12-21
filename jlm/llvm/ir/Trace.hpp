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
  OutputTracer();

protected:
  [[nodiscard]] rvsdg::Output &
  traceStep(rvsdg::Output & output, bool mayLeaveRegion) override;
};

/**
 * Traces the origin of the given \p output to find the origin of the value.
 * Traces through everything handled by \ref jlm::rvsdg::traceOutput,
 * with the addition of LLVM-specific operations.
 *
 * @param output the output to start tracing from
 * @return the maximally traced output
 */
rvsdg::Output &
traceOutput(rvsdg::Output & output);

inline const rvsdg::Output &
traceOutput(const rvsdg::Output & output)
{
  return llvm::traceOutput(const_cast<rvsdg::Output &>(output));
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
