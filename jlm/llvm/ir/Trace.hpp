/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_TRACE_HPP
#define JLM_LLVM_IR_TRACE_HPP

#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
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

// FIXME: update documentation

/**
 * Represents the result of tracing a pointer p to some origin,
 * as a traced base pointer value plus an optional byte offset.
 *
 * If the offset is present, that means
 *  p = base pointer + offset
 *
 * If the offset is not present, that means
 *  p = base pointer + [unknown offset]
 */
struct TracedPointerOrigin
{
  const rvsdg::Output * BasePointer = nullptr;
  std::optional<int64_t> Offset;
  std::optional<std::vector<GetElementPtrOperation::Constant>> gepConstants{};
};

/**
 * Returns the result of tracing the origin of p as far as possible, without tracing through
 * operations that add an unknown offsets, or add multiple possible origins.
 * Since the pointer has exactly one possible origin with a statically known offset,
 * it may be possible to give MustAlias responses.
 * If not, more extensive tracing can be performed using \ref TraceAllPointerOrigins.
 *
 * @param p the pointer value to trace
 * @return the TracedPointer for p, which is guaranteed to have a defined offset
 */
[[nodiscard]] TracedPointerOrigin
TracePointerOriginPrecise(const rvsdg::Output & p);

/**
 * Represents a collection of possible origins of a pointer value.
 */
struct TraceCollection
{
  /**
   * Contains all outputs visited while tracing, to avoid re-visiting.
   * If an output is visited first with a known offset, and later with a different offset,
   * the offset is collapsed to an unknown offset, and tracing continues with that.
   */
  std::unordered_map<const rvsdg::Output *, std::optional<int64_t>> AllTracedOutputs{};

  /**
   * Contains the outputs that have been reached through tracing, which can not be traced further.
   * For example, the output of an \ref AllocaOperation, the result of a \ref
   * LoadNonVolatileOperation, or the return value of a \ref CallOperation.
   */
  std::unordered_map<const rvsdg::Output *, std::optional<int64_t>> TopOrigins{};
};

/**
 * Traces to find all possible origins of the given pointer.
 * Traces through \ref GetElementPtrOperation, including those with offsets that are not known
 * at compile time. Also traces through gamma and theta nodes, building a set of multiple
 * possibilities. Tracing stops at "top origins", for example an \ref AllocaOperation, a \ref
 * LoadNonVolatileOperation, the return value of a \ref CallOperation etc.
 *
 * @param p the pointer to trace from
 * @param traceCollection the collection of trace points being created
 * @param maxTraceCollectionSize the number of outputs a pointer can be traced to before giving up
 * @return false if the trace collection reached its maximum allowed size, and tracing aborted
 */
[[nodiscard]] bool
TraceAllPointerOrigins(
    TracedPointerOrigin p,
    TraceCollection & traceCollection,
    size_t maxTraceCollectionSize);

}

#endif // JLM_LLVM_IR_TRACE_HPP
